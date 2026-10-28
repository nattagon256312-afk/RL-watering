#include <Arduino.h>
#include "rl_model.h"  
#include <math.h>
#include <DHT.h>
#include <WiFi.h>
#include <time.h>


#include "all_ops_resolver.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>


#define DHTPIN 4         
#define DHTTYPE DHT22     
DHT dht(DHTPIN, DHTTYPE);

#define SLEEP_US (3600ULL * 1000000ULL)

const int humidity_pin = 34; 
const int relay_pin = 18;    

const char* ssid = "Anong_2.4GHz";
const char* password = "Aa0821588287";


const float MAX_CAP = 1000.0f;                  
const int TOTAL_EPISODE_DAYS = 3;              
const int MAX_STEPS = TOTAL_EPISODE_DAYS * 24; // 72 steps
const float water_actions[6] = {0, 20, 40, 60, 80, 100}; // Action 


float max_water = 1000.0f;         
float water_remaining = 1000.0f;  
int steps_remaining = MAX_STEPS;  
int current_hour = 0;     



const int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
static tflite::AllOpsResolver resolver;


void update_and_set_observations(TfLiteTensor* input, float raw_temp, int raw_soil_adc);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connected WiFi...");
  }

   configTime(7 * 3600, 0, "pool.ntp.org", "time.google.com");
   struct tm timeinfo;

  if (getLocalTime(&timeinfo, 10000)) {
    Serial.println("Time synchronized!");
  } else {
    Serial.println("Failed to get time");
  }

  pinMode(relay_pin, OUTPUT);
  digitalWrite(relay_pin, LOW); 
  dht.begin();

  
  model = tflite::GetModel(rl_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, nullptr);
  interpreter = &static_interpreter;

  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }

  
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("RL Model Loaded Successfully!");
  delay(2000);
}


void loop() {

  struct tm timeinfo;
  if (getLocalTime(&timeinfo, 10000)) {
    current_hour = timeinfo.tm_hour;
  }
  

  int raw_soil_adc = analogRead(humidity_pin);
  float raw_temp = dht.readTemperature();

  if (isnan(raw_temp)) {
    raw_temp = 25.0f; 
  }

  
  update_and_set_observations(input, raw_temp, raw_soil_adc);

  
  Serial.print("[ADC: "); Serial.print(raw_soil_adc);
  Serial.print(" | Soil Norm: "); Serial.print(input->data.f[0]);
  Serial.print(" | Water Rem: "); Serial.print(input->data.f[1]);
  Serial.print("] ");

  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }
  
  TfLiteTensor* output = interpreter->output(0);

  
  int best_action = 0;
  float max_logit = output->data.f[0];

  for (int i = 1; i < 6; i++) {
    if (output->data.f[i] > max_logit) {
      max_logit = output->data.f[i];
      best_action = i;
    }
  }

  
  float water_to_give = water_actions[best_action];
  Serial.print("[Hour "); Serial.print(current_hour); Serial.print("] ");
  Serial.print("Action: "); Serial.print(best_action);
  Serial.print(" -> Water: "); Serial.print(water_to_give); Serial.println(" ml");

  
  if (water_to_give > 0 && water_remaining > 0) {
    float actual_water = min(water_to_give, water_remaining);
    digitalWrite(relay_pin, LOW);
    
    // อัตราการไหล 7.7 ml/s
    unsigned long pump_duration_ms = (unsigned long)((actual_water / 7.7f) * 1000.0f);
    delay(pump_duration_ms); 
    
    digitalWrite(relay_pin, HIGH);
    water_remaining -= actual_water; 
  } else {
    digitalWrite(relay_pin, HIGH);
  }


  current_hour = (current_hour + 1) % 24;
  steps_remaining--;

  
  if (steps_remaining <= 0) {
    steps_remaining = MAX_STEPS;
    water_remaining = max_water;
    Serial.println("--- Episode Reset ---");
  }

digitalWrite(relay_pin, HIGH);

gpio_hold_en((gpio_num_t)relay_pin); 
esp_sleep_enable_timer_wakeup(SLEEP_US);
Serial.flush();
esp_light_sleep_start(); 
gpio_hold_dis((gpio_num_t)relay_pin);
}

void update_and_set_observations(TfLiteTensor* input, float raw_temp, int raw_soil_adc) {
  //  ความชื้นในดิน Normalized (0.0 = แห้ง, 1.0 = เปียก)
  raw_soil_adc = constrain(raw_soil_adc, 1361, 4095);
  float humidity_norm = (float)(4095 - raw_soil_adc) / (4095 - 1361);
  input->data.f[0] = humidity_norm;

  // สัดส่วนน้ำที่เหลืออยู่
  float water_remaining_ratio = water_remaining / MAX_CAP;
  input->data.f[1] = constrain(water_remaining_ratio, 0.0f, 1.0f);

  //  สัดส่วนความจุเริ่มต้น
  float max_water_ratio = max_water / MAX_CAP;
  input->data.f[2] = constrain(max_water_ratio, 0.0f, 1.0f);

  //  สัดส่วนเวลาที่เหลือใน Episode
  float time_remaining_ratio = (float)steps_remaining / (float)MAX_STEPS;
  input->data.f[3] = constrain(time_remaining_ratio, 0.0f, 1.0f);

  //  Sin/Cos of Hour
  input->data.f[4] = sin(2.0f * M_PI * current_hour / 24.0f);
  input->data.f[5] = cos(2.0f * M_PI * current_hour / 24.0f);

  //  อัตราการระเหย e_t
  float daily_factor = 1.0f + 0.5f * cos(2.0f * M_PI * (current_hour - 12.0f) / 24.0f);
  float evap_scale = raw_temp / 25.0f;
  float e_t = max(0.0f, evap_scale * daily_factor * 6.0f);
  
  float evap_input = min(e_t / 20.0f, 1.0f);
  input->data.f[6] = evap_input;
}






