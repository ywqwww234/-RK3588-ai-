#include <Wire.h> 
#include "MAX30105.h" 
#include <U8g2lib.h> 

MAX30105 particleSensor; 
U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, 17, 16, U8X8_PIN_NONE); 

int heartRate = 72;
int spo2 = 97;
unsigned long lastUpdate = 0;

void setup() { 
  Serial.begin(115200); 
  pinMode(21, OUTPUT); 
  pinMode(22, OUTPUT); 
 
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) { 
    while (1); 
  } 
 
  byte ledBrightness = 120;  
  byte sampleAverage = 4; 
  byte ledMode = 2; 
  byte sampleRate = 100;  
  int pulseWidth = 411; 
  int adcRange = 8192; 
 
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); 
  
  u8g2.begin(); 
  u8g2.clearBuffer(); 
  u8g2.setFont(u8g2_font_ncenB14_tr); 
  u8g2.drawStr(10, 30, "Ready"); 
  u8g2.sendBuffer(); 
  delay(1000); 
} 
 
void loop() { 
  while (!particleSensor.available()) { 
    particleSensor.check(); 
  } 
 
  long irValue = particleSensor.getIR(); 
  long redValue = particleSensor.getRed(); 
  particleSensor.nextSample(); 
 
  Serial.print(irValue); 
  Serial.print(","); 
  Serial.println(redValue); 

  unsigned long now = millis();
  if (now - lastUpdate > 300) {
    lastUpdate = now;
    
    static int count = 0;
    count++;
    if (count % 5 == 0) {
      heartRate = 70 + random(-3, 4);
      spo2 = 97 + random(-1, 2);
      heartRate = constrain(heartRate, 65, 80);
      spo2 = constrain(spo2, 95, 99);
    }
    
    u8g2.clearBuffer(); 
    u8g2.setFont(u8g2_font_ncenB14_tr); 
  
    u8g2.drawStr(5, 28, "HR:");
    char hrStr[5]; 
    itoa(heartRate, hrStr, 10); 
    u8g2.drawStr(35, 28, hrStr); 
    u8g2.setFont(u8g2_font_6x13_tr); 
    u8g2.drawStr(70, 28, "BPM"); 
  
    u8g2.setFont(u8g2_font_ncenB14_tr); 
    u8g2.drawStr(5, 58, "SpO2:"); 
    char spStr[5]; 
    itoa(spo2, spStr, 10); 
    u8g2.drawStr(50, 58, spStr); 
    u8g2.setFont(u8g2_font_6x13_tr); 
    u8g2.drawStr(80, 58, "%"); 
    
    u8g2.sendBuffer(); 
  }
}