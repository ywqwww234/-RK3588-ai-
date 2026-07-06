#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

#define OLED_ADDR 0x3C
#define OLED_SDA 16
#define OLED_SCL 17

TwoWire Wire_OLED = TwoWire(0);

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire_OLED, OLED_RESET);

void setup() {
  Serial.begin(115200);
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102传感器初始化失败！");
    while (1);
  }

  byte ledBrightness = 200;
  byte sampleAverage = 4;
  byte ledMode = 2;
  byte sampleRate = 400;
  int pulseWidth = 411;
  int adcRange = 8192;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  Wire_OLED.begin(OLED_SDA, OLED_SCL, 400000);

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR, false, false)) {
    Serial.println("OLED显示初始化失败！");
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("MAX30102 Ready");
  display.display();
}

void loop() {
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  Serial.print(irValue);
  Serial.print(",");
  Serial.println(redValue);

  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(2);
  display.print("IR:");
  display.println(irValue);
  display.print("RED:");
  display.println(redValue);
  display.display();

  delay(10);
}