/*
 * MindBand · ESP32 边缘智能手环
 * - MAX30102 PPG 传感器（GPIO 21/22 - 默认 Wire，用户验证启动方式不可改）
 * - SSD1306 OLED 显示屏 (GPIO 16/17 - 第二条 I2C 总线)
 * - 三色 LED 风险指示 (GPIO 25/26/27 - 绿/黄/红)
 * - 4 个 UI 模板：启动 / 等待 / 监测 / 风险卡（LOW/MID/HIGH 三种）
 * - 与 PC 端 1.py 兼容的 [DATA] 串口协议
 */

#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ============================================================
// GPIO 分配（每个设备独立 - 不冲突）
// ============================================================
// MAX30102：使用默认 Wire（I2C0, SDA=21, SCL=22）—— 用户指定锁定
// OLED：独立 I2C1 外设（TwoWire(1)），引脚 16/17 —— 切勿改回 TwoWire(0)，
//       那会与 Wire 共用同一硬件外设，重映射引脚后 MAX30102 无法通信
#define OLED_ADDR    0x3C
#define OLED_SDA     16
#define OLED_SCL     17
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1

// 三色 LED（共阴极，HIGH=点亮）
#define LED_GREEN   25
#define LED_YELLOW  26
#define LED_RED     27

// 调试 LED（板载或外接）
#define PIN_PULSE_LED 13
#define PIN_READ_LED  2

// ============================================================
// 驱动对象
// ============================================================
TwoWire Wire_OLED = TwoWire(1);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire_OLED, OLED_RESET);
MAX30105 particleSensor;

// ============================================================
// 心率算法状态
// ============================================================
const byte RATE_SIZE = 8;       // 提升平均窗口至 8，工业级稳定性
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;

#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
  uint16_t irBuffer[100];
  uint16_t redBuffer[100];
#else
  uint32_t irBuffer[100];
  uint32_t redBuffer[100];
#endif

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

// 时间
unsigned long bootTime = 0;
unsigned long lastDisplayTime = 0;
unsigned long lastBeatVisual = 0;
unsigned long riskCardStart = 0;
unsigned long riskPendingStart = 0;

const unsigned long DISPLAY_INTERVAL = 150;   // ms
const unsigned long RISK_CARD_DURATION = 2800; // ms
const unsigned long RISK_HYSTERESIS = 2500;    // 等级稳定 2.5s 才切换

bool fingerDetected = false;
int  displayBPM = 0;
int  displaySPO2 = 0;
long lastIR = 0;

// ============================================================
// 位图图标（PROGMEM 节省 RAM）
// ============================================================
// 16×14 大心形
const unsigned char PROGMEM bmp_heart16[] = {
  0x0E,0x70, 0x1F,0xF8, 0x3F,0xFC, 0x7F,0xFE,
  0x7F,0xFE, 0x7F,0xFE, 0x7F,0xFE, 0x3F,0xFC,
  0x1F,0xF8, 0x0F,0xF0, 0x07,0xE0, 0x03,0xC0,
  0x01,0x80, 0x00,0x00
};
// 16×14 心形外框（脉冲间隔时显示）
const unsigned char PROGMEM bmp_heart16_outline[] = {
  0x0E,0x70, 0x11,0x88, 0x20,0x04, 0x40,0x02,
  0x40,0x02, 0x40,0x02, 0x40,0x02, 0x20,0x04,
  0x10,0x08, 0x08,0x10, 0x04,0x20, 0x02,0x40,
  0x01,0x80, 0x00,0x00
};
// 8×10 SpO2 水滴
const unsigned char PROGMEM bmp_drop[] = {
  0x10, 0x38, 0x38, 0x7C, 0x7C, 0xFE, 0xFE, 0x7C, 0x38, 0x00
};
// 8×8 警告三角
const unsigned char PROGMEM bmp_warn[] = {
  0x18, 0x18, 0x3C, 0x3C, 0x66, 0x66, 0xFF, 0xFF
};
// 8×8 OK 对勾
const unsigned char PROGMEM bmp_ok[] = {
  0x00, 0x03, 0x07, 0x8E, 0xDC, 0x78, 0x30, 0x00
};

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

// 工业级风险评估（对标小米/华为手环阈值）
RiskLevel evaluateRisk(int bpm, int spo2, bool finger) {
  if (!finger || bpm <= 0) return RISK_NONE;

  int score = 0;
  // 心率分数（静息正常 60-100 bpm）
  if      (bpm < 45 || bpm > 130) score += 3;
  else if (bpm < 55 || bpm > 110) score += 2;
  else if (bpm < 60 || bpm > 100) score += 1;

  // SpO2 分数（正常 ≥95%）
  if (spo2 > 0) {
    if      (spo2 < 90) score += 3;
    else if (spo2 < 93) score += 2;
    else if (spo2 < 95) score += 1;
  }

  if (score >= 4) return RISK_HIGH;
  if (score >= 2) return RISK_MID;
  return RISK_LOW;
}

void printCentered(const char* text, int y, int textSize) {
  display.setTextSize(textSize);
  int w = strlen(text) * 6 * textSize;
  int x = (SCREEN_WIDTH - w) / 2;
  if (x < 0) x = 0;
  display.setCursor(x, y);
  display.print(text);
}

// ============================================================
// 屏幕 1：启动
// ============================================================
void showStartupScreen(int progress) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // 标题
  printCentered("MindBand", 6, 2);
  display.setTextSize(1);
  printCentered("Edge AI Wearable", 26, 1);

  // 进度条外框
  display.drawRect(14, 40, 100, 8, SSD1306_WHITE);
  int barW = constrain(progress, 0, 100);
  display.fillRect(16, 42, barW, 4, SSD1306_WHITE);

  // 状态文本
  display.setTextSize(1);
  display.setCursor(14, 54);
  if (progress < 30)      display.print("Init I2C bus...");
  else if (progress < 60) display.print("Calibrating...");
  else if (progress < 95) display.print("Loading model...");
  else                    display.print("Ready");

  display.display();
}

// ============================================================
// 屏幕 2：等待手指
// ============================================================
void showWaitingScreen() {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // 顶部状态栏
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("MindBand");
  display.setCursor(94, 0);
  display.print("WAIT");
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  // 中央提示
  display.setTextSize(2);
  display.setCursor(8, 18);
  display.print("Place");
  display.setCursor(8, 36);
  display.print("Finger");

  // 闪烁箭头提示
  static bool blink = false;
  blink = !blink;
  if (blink) {
    display.fillTriangle(100, 28, 100, 44, 116, 36, SSD1306_WHITE);
  } else {
    display.drawTriangle(100, 28, 100, 44, 116, 36, SSD1306_WHITE);
  }

  // 底部数据状态
  display.setTextSize(1);
  display.setCursor(0, 56);
  display.print("IR: ");
  display.print(lastIR);

  display.display();
}

// ============================================================
// 屏幕 3：主监测（默认运行）
// ============================================================
void showMonitorScreen(int bpm, int spo2, RiskLevel risk) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  // === 顶部状态栏 ===
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("MindBand");
  // 运行时间
  unsigned long t = (millis() - bootTime) / 1000;
  char timeStr[12];
  sprintf(timeStr, "%02lu:%02lu:%02lu", t / 3600, (t / 60) % 60, t % 60);
  display.setCursor(72, 0);
  display.print(timeStr);
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  // === 左侧：心率（大字）===
  // 心形动画：心跳后 150ms 内显示实心，否则空心
  bool beating = (millis() - lastBeatVisual) < 150;
  if (beating) {
    display.drawBitmap(0, 16, bmp_heart16, 16, 14, SSD1306_WHITE);
  } else {
    display.drawBitmap(0, 16, bmp_heart16_outline, 16, 14, SSD1306_WHITE);
  }

  display.setTextSize(3);
  display.setCursor(20, 14);
  if (bpm > 0) {
    if (bpm < 100) display.print(" ");
    display.print(bpm);
  } else {
    display.print(" --");
  }
  display.setTextSize(1);
  display.setCursor(20, 36);
  display.print("BPM");

  // 分隔竖线
  display.drawLine(80, 14, 80, 44, SSD1306_WHITE);

  // === 右侧：SpO2 ===
  display.drawBitmap(86, 14, bmp_drop, 8, 10, SSD1306_WHITE);
  display.setTextSize(2);
  display.setCursor(98, 14);
  if (spo2 > 0) {
    display.print(spo2);
  } else {
    display.print("--");
  }
  display.setTextSize(1);
  display.setCursor(86, 32);
  display.print("SpO2 %");

  // === 底部：风险等级条 ===
  display.drawLine(0, 47, 128, 47, SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 52);
  const char* statusText = "WAITING";
  int barWidth = 0;
  switch (risk) {
    case RISK_LOW:
      statusText = "OK     LOW";
      barWidth = 36;
      display.drawBitmap(58, 52, bmp_ok, 8, 8, SSD1306_WHITE);
      break;
    case RISK_MID:
      statusText = "WARN   MID";
      barWidth = 78;
      display.drawBitmap(58, 52, bmp_warn, 8, 8, SSD1306_WHITE);
      break;
    case RISK_HIGH:
      statusText = "ALERT HIGH";
      barWidth = 124;
      display.drawBitmap(58, 52, bmp_warn, 8, 8, SSD1306_WHITE);
      break;
    default:
      statusText = "NO SIGNAL";
      barWidth = 0;
      break;
  }
  display.setCursor(0, 52);
  display.print(statusText);
  display.fillRect(2, 61, barWidth, 3, SSD1306_WHITE);

  display.display();
}

// ============================================================
// 屏幕 4：风险等级卡（LOW / MID / HIGH 三种）
// ============================================================
void showRiskCard(RiskLevel risk, int bpm, int spo2) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  const char* tag = "LOW";
  const char* line1 = "Heart steady";
  const char* line2 = "All good";
  const unsigned char* icon = bmp_ok;

  if (risk == RISK_MID) {
    tag = "MID";
    line1 = "Watch out";
    line2 = "Slow & breathe";
    icon = bmp_warn;
  } else if (risk == RISK_HIGH) {
    tag = "HIGH";
    line1 = "Attention!";
    line2 = "Rest now";
    icon = bmp_warn;
  }

  // 外双框（强视觉冲击）
  display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
  display.drawRect(2, 2, 124, 60, SSD1306_WHITE);

  // 顶部小字 + 图标
  display.setTextSize(1);
  display.setCursor(10, 6);
  display.print("RISK LEVEL");
  display.drawBitmap(108, 5, icon, 8, 8, SSD1306_WHITE);

  // 大字等级
  display.setTextSize(3);
  int textWidth = strlen(tag) * 18;
  display.setCursor((SCREEN_WIDTH - textWidth) / 2, 18);
  display.print(tag);

  // 闪烁效果（高危等级时）
  if (risk == RISK_HIGH) {
    static bool flash = false;
    flash = !flash;
    if (flash) {
      display.drawRect(4, 4, 120, 56, SSD1306_WHITE);
    }
  }

  // 提示文字
  display.setTextSize(1);
  printCentered(line1, 44, 1);

  // 底部数据
  char hr_spo2[24];
  sprintf(hr_spo2, "HR %d   SpO2 %d%%", bpm, spo2);
  printCentered(hr_spo2, 54, 1);

  display.display();
}

// ============================================================
// 错误屏
// ============================================================
void showErrorScreen(const char* msg) {
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(2);
  printCentered("! ERROR", 8, 2);
  display.drawLine(0, 28, 128, 28, SSD1306_WHITE);
  display.setTextSize(1);
  printCentered(msg, 38, 1);
  printCentered("Reboot device", 52, 1);
  display.display();
}

// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);

  // === 关键启动顺序（来自 2.txt，已验证）===
  // 不要删除这两行 pinMode！
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);

  // === LED ===
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(PIN_READ_LED, OUTPUT);
  setLEDs(false, false, false);

  // === OLED 初始化（第二条 I2C 总线）===
  Wire_OLED.begin(OLED_SDA, OLED_SCL, 400000);
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR, false, false)) {
    Serial.println("[ERR] OLED init failed");
  }

  // 启动动画：进度从 0 到 100
  for (int p = 0; p <= 100; p += 5) {
    showStartupScreen(p);
    delay(60);
  }

  // === MAX30102 初始化（默认 Wire = GPIO 21/22）===
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERR] MAX30102 init failed");
    showErrorScreen("Sensor Fail");
    // 亮红灯告警
    setLEDs(false, false, true);
    while (1) { delay(100); }
  }

  // 工业级传感器参数
  byte ledBrightness = 200;
  byte sampleAverage = 4;
  byte ledMode = 2;          // Red + IR (SpO2 模式)
  byte sampleRate = 400;     // 400Hz 采样
  int pulseWidth = 411;
  int adcRange = 8192;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode,
                       sampleRate, pulseWidth, adcRange);

  Serial.println("[INFO] Sensor ready, waiting for finger");

  // === 等待手指 + 预填充 buffer ===
  currentMode = MODE_WAITING;
  showWaitingScreen();

  bufferLength = 100;
  for (byte i = 0; i < bufferLength; i++) {
    while (!particleSensor.available()) {
      particleSensor.check();
    }
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();

    // 进度反馈到 OLED（每 10 个采样刷新一次）
    if (i % 10 == 0) {
      showWaitingScreen();
    }

    Serial.print("[CAL] red=");
    Serial.print(redBuffer[i]);
    Serial.print(", ir=");
    Serial.println(irBuffer[i]);
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  bootTime = millis();
  currentMode = MODE_MONITOR;
  Serial.println("[INFO] Monitoring started");
}

// ============================================================
// loop
// ============================================================
void loop() {
  // === 实时读取 IR（心跳检测）===
  long irValue = particleSensor.getIR();
  lastIR = irValue;
  fingerDetected = (irValue >= 50000);

  if (checkForBeat(irValue) == true) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    lastBeatVisual = millis();  // 心形动画触发
    digitalWrite(PIN_READ_LED, !digitalRead(PIN_READ_LED));

    beatsPerMinute = 60 / (delta / 1000.0);

    if (beatsPerMinute < 220 && beatsPerMinute > 30) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;

      beatAvg = 0;
      for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  // === SpO2 滚动 buffer（每次替换最旧 25 个）===
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

  // === 显示节流 ===
  unsigned long now = millis();
  if (now - lastDisplayTime < DISPLAY_INTERVAL) return;
  lastDisplayTime = now;

  displayBPM = (validHeartRate && heartRate > 0 && heartRate < 220)
                   ? heartRate : beatAvg;
  displaySPO2 = (validSPO2 && spo2 > 0 && spo2 <= 100) ? spo2 : 0;

  // === 风险评估 + 等级稳定（迟滞）===
  RiskLevel newRisk = evaluateRisk(displayBPM, displaySPO2, fingerDetected);

  if (newRisk != currentRisk) {
    if (newRisk != pendingRisk) {
      pendingRisk = newRisk;
      riskPendingStart = now;
    } else if (now - riskPendingStart >= RISK_HYSTERESIS) {
      // 稳定足够长 → 触发等级切换
      RiskLevel prev = currentRisk;
      currentRisk = newRisk;
      pendingRisk = newRisk;

      // 等级变化时弹出风险卡（除非从 NONE 进入 LOW，无需打扰）
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

  // === 串口协议（兼容 1.py 解析器）===
  Serial.print("[DATA] ir=");
  Serial.print(irValue);
  Serial.print(", bpm=");
  Serial.print(displayBPM);
  Serial.print(", avg=");
  Serial.print(beatAvg);
  Serial.print(", spo2=");
  Serial.print(displaySPO2);
  Serial.print(", finger=");
  Serial.print(fingerDetected ? 1 : 0);
  Serial.print(", risk=");
  Serial.print((int)currentRisk);
  if (!fingerDetected) Serial.print(" NoFinger");
  Serial.println();

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
