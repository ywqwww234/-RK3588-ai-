/*
 * MindBand · ESP32-C3 边缘智能手环 (TFT版)
 * - MAX30102 PPG 传感器（I2C, SDA=8, SCL=9）
 * - ST7789 1.54" TFT 240x240 (SPI)
 * - 三色 LED 风险指示
 */

#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <TFT_eSPI.h>

// ============================================================
// GPIO 分配
// ============================================================
// MAX30102：I2C (SDA=8, SCL=9)
// TFT：SPI（在 TFT_eSPI User_Setup.h 中配置）
// LED：使用安全GPIO
// ESP32-C3 SuperMini 可用 GPIO: 0-10, 20, 21
// 已占用: 0(BL) 1(RST) 2(DC) 6(MOSI) 7(SCLK) 10(CS) 8(SDA) 9(SCL) 20/21(UART)
// LED 改到 3/4/5（原 10/18/19 冲突且 18/19 在 C3 上不存在，导致芯片卡死）
#define LED_GREEN   3
#define LED_YELLOW  4
#define LED_RED     5

// ============================================================
// 驱动对象
// ============================================================
TFT_eSPI tft = TFT_eSPI();
MAX30105 particleSensor;

// ============================================================
// 心率算法状态
// ============================================================
const byte RATE_SIZE = 8;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

uint32_t irBuffer[100];
uint32_t redBuffer[100];
int32_t bufferLength = 100;
int32_t spo2 = 0;
int8_t  validSPO2 = 0;
int32_t heartRate = 0;
int8_t  validHeartRate = 0;

// ============================================================
// 显示 & 状态机
// ============================================================
enum DisplayMode {
  MODE_STARTUP,
  MODE_WAITING,
  MODE_MONITOR,
  MODE_RISK_CARD
};
DisplayMode currentMode = MODE_STARTUP;

enum RiskLevel {
  RISK_NONE = 0,
  RISK_LOW  = 1,
  RISK_MID  = 2,
  RISK_HIGH = 3
};
RiskLevel currentRisk = RISK_NONE;
RiskLevel pendingRisk = RISK_NONE;

// 时间与限流
unsigned long bootTime = 0;
unsigned long lastDisplayTime = 0;
unsigned long lastBeatVisual = 0;
unsigned long riskCardStart = 0;
unsigned long riskPendingStart = 0;

const unsigned long DISPLAY_INTERVAL = 100;
const unsigned long RISK_CARD_DURATION = 2800;
const unsigned long RISK_HYSTERESIS = 2500;

bool fingerDetected = false;
int  displayBPM = 0;
int  displaySPO2 = 0;
long lastIR = 0;

// ============================================================
// 颜色定义（RGB565）
// ============================================================
#define COLOR_BG       0x0000
#define COLOR_PRIMARY  0x07FF
#define COLOR_TEXT     0xFFFF
#define COLOR_GRAY     0x7BEF
#define COLOR_GREEN    0x07E0
#define COLOR_YELLOW   0xFFE0
#define COLOR_RED      0xF800

// ============================================================
// 工具函数
// ============================================================
void setLEDs(bool g, bool y, bool r) {
  digitalWrite(LED_GREEN, g ? HIGH : LOW);
  digitalWrite(LED_YELLOW, y ? HIGH : LOW);
  digitalWrite(LED_RED, r ? HIGH : LOW);
}

void updateLEDs(RiskLevel level) {
  switch (level) {
    case RISK_LOW:  setLEDs(true,  false, false); break;
    case RISK_MID:  setLEDs(false, true,  false); break;
    case RISK_HIGH: setLEDs(false, false, true ); break;
    default:        setLEDs(false, false, false); break;
  }
}

RiskLevel evaluateRisk(int bpm, int spo2, bool finger) {
  if (!finger || bpm <= 0) return RISK_NONE;

  int score = 0;
  if      (bpm < 45 || bpm > 130) score += 3;
  else if (bpm < 55 || bpm > 110) score += 2;
  else if (bpm < 60 || bpm > 100) score += 1;

  if (spo2 > 0) {
    if      (spo2 < 90) score += 3;
    else if (spo2 < 93) score += 2;
    else if (spo2 < 95) score += 1;
  }

  if (score >= 4) return RISK_HIGH;
  if (score >= 2) return RISK_MID;
  return RISK_LOW;
}

// ============================================================
// 简化显示函数
// ============================================================
void showSimpleText(const char* line1, const char* line2 = "", const char* line3 = "") {
  tft.fillScreen(COLOR_BG);
  tft.setTextColor(COLOR_TEXT);
  tft.setTextSize(2);

  tft.setCursor(20, 80);
  tft.println(line1);

  if (strlen(line2) > 0) {
    tft.setCursor(20, 110);
    tft.println(line2);
  }

  if (strlen(line3) > 0) {
    tft.setCursor(20, 140);
    tft.println(line3);
  }
}

// ============================================================
// 屏幕显示（简化版）
// ============================================================
void showStartupScreen(int progress) {
  tft.fillScreen(COLOR_BG);
  tft.setTextColor(COLOR_PRIMARY);
  tft.setTextSize(3);
  tft.setCursor(40, 80);
  tft.println("MindBand");

  tft.setTextSize(2);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(60, 120);
  tft.print(progress);
  tft.println("%");
}

void showWaitingScreen() {
  tft.fillScreen(COLOR_BG);
  tft.setTextColor(COLOR_PRIMARY);
  tft.setTextSize(3);
  tft.setCursor(30, 80);
  tft.println("Place");
  tft.setCursor(30, 120);
  tft.println("Finger");

  tft.setTextSize(1);
  tft.setTextColor(COLOR_GRAY);
  tft.setCursor(10, 200);
  tft.print("IR: ");
  tft.print(lastIR);
}

void showMonitorScreen(int bpm, int spo2, RiskLevel risk) {
  tft.fillScreen(COLOR_BG);

  // 心率
  tft.setTextSize(2);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(20, 60);
  tft.println("BPM");
  tft.setTextSize(4);
  tft.setTextColor(COLOR_GREEN);
  tft.setCursor(20, 90);
  if (bpm > 0) {
    tft.println(bpm);
  } else {
    tft.println("--");
  }

  // SpO2
  tft.setTextSize(2);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(140, 60);
  tft.println("SpO2");
  tft.setTextSize(4);
  tft.setTextColor(COLOR_GREEN);
  tft.setCursor(140, 90);
  if (spo2 > 0) {
    tft.println(spo2);
  } else {
    tft.println("--");
  }

  // 状态
  tft.setTextSize(2);
  tft.setCursor(20, 180);
  switch (risk) {
    case RISK_LOW:  tft.setTextColor(COLOR_GREEN); tft.println("OK"); break;
    case RISK_MID:  tft.setTextColor(COLOR_YELLOW); tft.println("WARN"); break;
    case RISK_HIGH: tft.setTextColor(COLOR_RED); tft.println("ALERT"); break;
    default: tft.setTextColor(COLOR_GRAY); tft.println("WAIT"); break;
  }
}

void showRiskCard(RiskLevel risk, int bpm, int spo2) {
  tft.fillScreen(COLOR_BG);

  const char* tag = "LOW";
  uint16_t color = COLOR_GREEN;

  if (risk == RISK_MID) {
    tag = "MID";
    color = COLOR_YELLOW;
  } else if (risk == RISK_HIGH) {
    tag = "HIGH";
    color = COLOR_RED;
  }

  tft.setTextSize(5);
  tft.setTextColor(color);
  tft.setCursor(40, 100);
  tft.println(tag);

  tft.setTextSize(2);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(20, 180);
  tft.print("HR:");
  tft.print(bpm);
  tft.print(" SpO2:");
  tft.print(spo2);
}

// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n\n[INFO] === MindBand Booting ===");

  // === LED 初始化并测试 ===
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);

  Serial.println("[INFO] Testing LEDs...");
  digitalWrite(LED_GREEN, HIGH);
  delay(300);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_YELLOW, HIGH);
  delay(300);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED, HIGH);
  delay(300);
  digitalWrite(LED_RED, LOW);
  Serial.println("[INFO] LED test done");

  // === TFT 初始化 ===
  Serial.println("[INFO] Initializing TFT...");
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(COLOR_BG);

  // 测试显示
  tft.setTextColor(COLOR_PRIMARY);
  tft.setTextSize(2);
  tft.setCursor(50, 100);
  tft.println("TFT OK!");
  delay(1000);
  Serial.println("[INFO] TFT initialized");

  // 启动动画
  for (int p = 0; p <= 100; p += 10) {
    showStartupScreen(p);
    delay(100);
  }

  // === I2C 初始化 ===
  Serial.println("[INFO] Initializing I2C...");
  Wire.begin(8, 9);
  delay(100);

  // === MAX30102 初始化 ===
  Serial.println("[INFO] Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERR] MAX30102 init failed!");
    showSimpleText("ERROR", "Sensor Fail");
    digitalWrite(LED_RED, HIGH);
    while (1) { delay(100); }
  }

  // 传感器参数
  particleSensor.setup(200, 4, 2, 400, 411, 8192);
  Serial.println("[INFO] MAX30102 ready");

  // === 预填充 buffer ===
  currentMode = MODE_WAITING;
  showWaitingScreen();

  bufferLength = 100;
  Serial.println("[INFO] Calibrating...");
  unsigned long calibStart = millis();
  for (byte i = 0; i < bufferLength; i++) {
    unsigned long timeout = millis();
    while (!particleSensor.available()) {
      particleSensor.check();
      if (millis() - timeout > 100) break;
    }
    if (millis() - calibStart > 5000) {
      Serial.println("[WARN] Calibration timeout");
      break;
    }
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  bootTime = millis();
  currentMode = MODE_MONITOR;
  digitalWrite(LED_GREEN, HIGH);
  Serial.println("[INFO] === System Ready ===\n");
}

// ============================================================
// loop
// ============================================================
void loop() {
  // === 实时读取 IR ===
  long irValue = particleSensor.getIR();
  lastIR = irValue;
  fingerDetected = (irValue >= 50000);

  if (checkForBeat(irValue) == true) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    lastBeatVisual = millis();

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 220 && beatsPerMinute > 30) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;

      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // === SpO2 滚动 buffer ===
  for (byte i = 25; i < 100; i++) {
    redBuffer[i - 25] = redBuffer[i];
    irBuffer[i - 25] = irBuffer[i];
  }
  for (byte i = 75; i < 100; i++) {
    while (!particleSensor.available()) {
      particleSensor.check();
    }
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  // === 显示限流 ===
  unsigned long now = millis();
  if (now - lastDisplayTime < DISPLAY_INTERVAL) return;
  lastDisplayTime = now;

  displayBPM = (validHeartRate && heartRate > 0 && heartRate < 220)
                   ? heartRate : beatAvg;
  displaySPO2 = (validSPO2 && spo2 > 0 && spo2 <= 100) ? spo2 : 0;

  // === 风险评估 + 迟滞 ===
  RiskLevel newRisk = evaluateRisk(displayBPM, displaySPO2, fingerDetected);

  if (newRisk != currentRisk) {
    if (newRisk != pendingRisk) {
      pendingRisk = newRisk;
      riskPendingStart = now;
    } else if (now - riskPendingStart >= RISK_HYSTERESIS) {
      RiskLevel prev = currentRisk;
      currentRisk = newRisk;
      pendingRisk = newRisk;

      if (fingerDetected && currentRisk != RISK_NONE
          && !(prev == RISK_NONE && currentRisk == RISK_LOW)) {
        currentMode = MODE_RISK_CARD;
        riskCardStart = now;
      }
    }
  } else {
    pendingRisk = newRisk;
    riskPendingStart = now;
  }

  // === LED 同步 ===
  updateLEDs(currentRisk);

  // === 串口协议 ===
  Serial.print("[DATA] ir=");
  Serial.print(irValue);
  Serial.print(", bpm=");
  Serial.print(displayBPM);
  Serial.print(", spo2=");
  Serial.print(displaySPO2);
  Serial.print(", finger=");
  Serial.print(fingerDetected ? 1 : 0);
  Serial.print(", risk=");
  Serial.println((int)currentRisk);

  // === 画面调度 ===
  if (!fingerDetected) {
    currentMode = MODE_WAITING;
  } else if (currentMode == MODE_RISK_CARD) {
    if (now - riskCardStart > RISK_CARD_DURATION) {
      currentMode = MODE_MONITOR;
    }
  } else {
    currentMode = MODE_MONITOR;
  }

  switch (currentMode) {
    case MODE_WAITING:
      showWaitingScreen();
      break;
    case MODE_RISK_CARD:
      showRiskCard(currentRisk, displayBPM, displaySPO2);
      break;
    case MODE_MONITOR:
    default:
      showMonitorScreen(displayBPM, displaySPO2, currentRisk);
      break;
  }
}