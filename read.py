#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102传感器初始化失败！");
    while (1);
  }

  // 调整传感器参数，避免饱和，获得更稳定的PPG波形
  byte ledBrightness = 120;   // 降低亮度，减少饱和尖峰
  byte sampleAverage = 4;
  byte ledMode = 2;
  byte sampleRate = 100;      // 与115200串口更匹配，降低丢包概率
  int pulseWidth = 411;
  int adcRange = 8192;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}

void loop() {
  // 等待新数据可用
  while (particleSensor.available() == false) {
    particleSensor.check();
  }

  // 正确顺序：先读当前样本，再移动FIFO指针
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();
  particleSensor.nextSample();

  // 输出格式：ir,red\n
  Serial.print(irValue);
  Serial.print(",");
  Serial.println(redValue);
}
