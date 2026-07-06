/*
 * MindBand · ESP32-C3 边缘智能手环 (TFT版)
 * - MAX30102 PPG 传感器（GPIO 21/22 - 默认 Wire）
 * - ST7789 1.54" TFT 显示屏 240x240 (SPI)
 * - 三色 LED 风险指示 (GPIO 3/4/5 - 绿/黄/红)
 * - 4 个 UI 模板：启动 / 等待 / 监测 / 风险卡
 * - 优化：DMA加速、限流机制、防SPI堵塞
 */

#include <Wire.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"
#include <TFT_eSPI.h>

// ============================================================
// GPIO 分配（ESP32-C3-supermini）
// ============================================================
// MAX30102：默认 Wire（I2C, SDA=8, SCL=9）
// TFT：SPI (MOSI=6, SCK=7, CS=10, DC=2, RST=-1, BL=1)
// RGB LED：GPIO 3/4/5

#define LED_GREEN   3
#define LED_YELLOW  4
#define LED_RED     5
#define LED_RGB     0   // WS2812 RGB LED

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

const unsigned long DISPLAY_INTERVAL = 100;    // 100ms限流，防SPI堵塞
const unsigned long RISK_CARD_DURATION = 2800;
const unsigned long RISK_HYSTERESIS = 2500;

bool fingerDetected = false;
int  displayBPM = 0;
int  displaySPO2 = 0;
long lastIR = 0;

// 波形缓冲（用于实时显示）
#define WAVE_POINTS 60
int waveBuffer[WAVE_POINTS];
int waveIndex = 0;

// ============================================================
// 颜色定义（RGB565）
// ============================================================
#define COLOR_BG       0x0000  // 黑色
#define COLOR_PRIMARY  0x07FF  // 青色
#define COLOR_TEXT     0xFFFF  // 白色
#define COLOR_GRAY     0x7BEF  // 灰色
#define COLOR_GREEN    0x07E0  // 绿色
#define COLOR_YELLOW   0xFFE0  // 黄色
#define COLOR_RED      0xF800  // 红色
#define COLOR_ORANGE   0xFD20  // 橙色

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

void drawCenteredText(const char* text, int y, uint8_t size, uint16_t color) {
  tft.setTextSize(size);
  tft.setTextColor(color);
  int16_t x1, y1;
  uint16_t w, h;
  tft.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  tft.setCursor((240 - w) / 2, y);
  tft.print(text);
}

// ============================================================
// 屏幕 1：启动界面（工业级设计）
// ============================================================
void showStartupScreen(int progress) {
  tft.fillScreen(COLOR_BG);

  // 顶部装饰线
  tft.fillRect(0, 0, 240, 3, COLOR_PRIMARY);

  // 品牌名称
  tft.setTextSize(3);
  tft.setTextColor(COLOR_PRIMARY);
  drawCenteredText("MindBand", 40, 3, COLOR_PRIMARY);

  // 副标题
  tft.setTextSize(1);
  drawCenteredText("Edge AI Wearable", 75, 1, COLOR_GRAY);

  // 进度条外框
  int barX = 30, barY = 110, barW = 180, barH = 20;
  tft.drawRoundRect(barX, barY, barW, barH, 10, COLOR_PRIMARY);

  // 进度填充（渐变效果）
  int fillW = map(progress, 0, 100, 0, barW - 4);
  if (fillW > 0) {
    tft.fillRoundRect(barX + 2, barY + 2, fillW, barH - 4, 8, COLOR_PRIMARY);
  }

  // 百分比
  char pct[8];
  sprintf(pct, "%d%%", progress);
  drawCenteredText(pct, 145, 2, COLOR_TEXT);

  // 状态文本
  const char* status = "Initializing...";
  if (progress < 30)      status = "Init I2C bus...";
  else if (progress < 60) status = "Calibrating...";
  else if (progress < 95) status = "Loading model...";
  else                    status = "Ready!";

  drawCenteredText(status, 180, 1, COLOR_GRAY);

  // 底部装饰
  tft.fillRect(0, 237, 240, 3, COLOR_PRIMARY);
}

// ============================================================
// 屏幕 2：等待手指
// ============================================================
void showWaitingScreen() {
  tft.fillScreen(COLOR_BG);

  // 顶部状态栏
  tft.fillRect(0, 0, 240, 30, 0x1082);
  tft.setTextSize(1);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(10, 10);
  tft.print("MindBand");
  tft.setCursor(180, 10);
  tft.print("WAIT");

  // 中央提示
  tft.setTextSize(3);
  drawCenteredText("Place", 80, 3, COLOR_PRIMARY);
  drawCenteredText("Finger", 115, 3, COLOR_PRIMARY);

  // 闪烁箭头（简化为三角形）
  static bool blink = false;
  blink = !blink;
  uint16_t arrowColor = blink ? COLOR_PRIMARY : COLOR_GRAY;
  tft.fillTriangle(200, 90, 200, 130, 220, 110, arrowColor);

  // 底部IR值
  tft.setTextSize(1);
  tft.setTextColor(COLOR_GRAY);
  tft.setCursor(10, 220);
  tft.print("IR: ");
  tft.print(lastIR);
}

// ============================================================
// 屏幕 3：主监测界面（工业级设计）
// ============================================================
void showMonitorScreen(int bpm, int spo2, RiskLevel risk) {
  tft.fillScreen(COLOR_BG);

  // === 顶部状态栏 ===
  tft.fillRect(0, 0, 240, 35, 0x1082);
  tft.setTextSize(1);
  tft.setTextColor(COLOR_TEXT);
  tft.setCursor(8, 10);
  tft.print("MindBand");

  // 运行时间
  unsigned long t = (millis() - bootTime) / 1000;
  char timeStr[12];
  sprintf(timeStr, "%02lu:%02lu:%02lu", t / 3600, (t / 60) % 60, t % 60);
  tft.setCursor(165, 10);
  tft.print(timeStr);

  // 电池图标（简化）
  tft.drawRect(210, 8, 20, 12, COLOR_GREEN);
  tft.fillRect(230, 11, 2, 6, COLOR_GREEN);
  tft.fillRect(212, 10, 14, 8, COLOR_GREEN);

  // === 心率大圆环 ===
  int cx1 = 70, cy1 = 110;
  uint16_t hrColor = COLOR_GREEN;
  if (bpm > 100 || bpm < 60) hrColor = COLOR_YELLOW;
  if (bpm > 120 || bpm < 50) hrColor = COLOR_RED;

  // 圆环背景
  tft.drawCircle(cx1, cy1, 50, COLOR_GRAY);
  tft.drawCircle(cx1, cy1, 48, hrColor);

  // 心率数值
  tft.setTextSize(4);
  tft.setTextColor(hrColor);
  char bpmStr[4];
  sprintf(bpmStr, "%d", bpm > 0 ? bpm : 0);
  int16_t x1, y1;
  uint16_t w, h;
  tft.getTextBounds(bpmStr, 0, 0, &x1, &y1, &w, &h);
  tft.setCursor(cx1 - w / 2, cy1 - 15);
  tft.print(bpmStr);

  tft.setTextSize(1);
  tft.setTextColor(COLOR_GRAY);
  tft.setCursor(cx1 - 10, cy1 + 15);
  tft.print("BPM");

  // === SpO2圆环 ===
  int cx2 = 170, cy2 = 110;
  uint16_t spo2Color = COLOR_GREEN;
  if (spo2 < 95 && spo2 > 0) spo2Color = COLOR_YELLOW;
  if (spo2 < 90 && spo2 > 0) spo2Color = COLOR_RED;

  tft.drawCircle(cx2, cy2, 50, COLOR_GRAY);
  tft.drawCircle(cx2, cy2, 48, spo2Color);

  // SpO2数值
  tft.setTextSize(4);
  tft.setTextColor(spo2Color);
  char spo2Str[4];
  sprintf(spo2Str, "%d", spo2 > 0 ? spo2 : 0);
  tft.getTextBounds(spo2Str, 0, 0, &x1, &y1, &w, &h);
  tft.setCursor(cx2 - w / 2, cy2 - 15);
  tft.print(spo2Str);

  tft.setTextSize(1);
  tft.setTextColor(COLOR_GRAY);
  tft.setCursor(cx2 - 15, cy2 + 15);
  tft.print("SpO2%");

  // === 底部波形 ===
  int waveY = 190;
  for (int i = 1; i < WAVE_POINTS; i++) {
    int x1 = (i - 1) * 4;
    int x2 = i * 4;
    int y1 = waveY - waveBuffer[i - 1] / 10;
    int y2 = waveY - waveBuffer[i] / 10;
    tft.drawLine(x1, y1, x2, y2, COLOR_PRIMARY);
  }

  // === 风险状态条 ===
  const char* statusText = "NO SIGNAL";
  uint16_t statusColor = COLOR_GRAY;
  switch (risk) {
    case RISK_LOW:  statusText = "OK"; statusColor = COLOR_GREEN; break;
    case RISK_MID:  statusText = "WARN"; statusColor = COLOR_YELLOW; break;
    case RISK_HIGH: statusText = "ALERT"; statusColor = COLOR_RED; break;
  }

  tft.fillRect(0, 215, 240, 25, statusColor);
  tft.setTextSize(2);
  tft.setTextColor(COLOR_BG);
  drawCenteredText(statusText, 220, 2, COLOR_BG);
}

// ============================================================
// 屏幕 4：风险等级卡
// ============================================================
void showRiskCard(RiskLevel risk, int bpm, int spo2) {
  tft.fillScreen(COLOR_BG);

  const char* tag = "LOW";
  const char* line1 = "Heart steady";
  const char* line2 = "All good";
  uint16_t cardColor = COLOR_GREEN;

  if (risk == RISK_MID) {
    tag = "MID";
    line1 = "Watch out";
    line2 = "Slow & breathe";
    cardColor = COLOR_YELLOW;
  } else if (risk == RISK_HIGH) {
    tag = "HIGH";
    line1 = "Attention!";
    line2 = "Rest now";
    cardColor = COLOR_RED;
  }

  // 外框（双层）
  tft.drawRect(10, 10, 220, 220, cardColor);
  tft.drawRect(12, 12, 216, 216, cardColor);

  // 顶部标签
  tft.setTextSize(1);
  tft.setTextColor(COLOR_GRAY);
  tft.setCursor(20, 25);
  tft.print("RISK LEVEL");

  // 大字等级
  tft.setTextSize(5);
  drawCenteredText(tag, 80, 5, cardColor);

  // 闪烁效果（高危）
  if (risk == RISK_HIGH) {
    static bool flash = false;
    flash = !flash;
    if (flash) {
      tft.drawRect(14, 14, 212, 212, cardColor);
    }
  }

  // 提示文字
  tft.setTextSize(2);
  drawCenteredText(line1, 150, 2, COLOR_TEXT);

  // 底部数据
  tft.setTextSize(1);
  char dataStr[32];
  sprintf(dataStr, "HR %d   SpO2 %d%%", bpm, spo2);
  drawCenteredText(dataStr, 200, 1, COLOR_GRAY);
}

// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);

  // === 关键启动顺序（保持不变）===
  pinMode(21, OUTPUT);
  pinMode(22, OUTPUT);

  // === LED ===
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  setLEDs(false, false, false);

  // === TFT 初始化 ===
  tft.init();
  tft.setRotation(0);  // 竖屏
  tft.fillScreen(COLOR_BG);

  // 启动动画
  for (int p = 0; p <= 100; p += 5) {
    showStartupScreen(p);
    delay(50);
  }

  // === MAX30102 初始化 ===
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERR] MAX30102 init failed");
    tft.fillScreen(COLOR_BG);
    tft.setTextSize(2);
    drawCenteredText("ERROR", 100, 2, COLOR_RED);
    drawCenteredText("Sensor Fail", 130, 1, COLOR_TEXT);
    setLEDs(false, false, true);
    while (1) { delay(100); }
  }

  // 传感器参数
  particleSensor.setup(200, 4, 2, 400, 411, 8192);

  Serial.println("[INFO] Sensor ready");

  // === 等待手指 + 预填充 ===
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

    if (i % 10 == 0) showWaitingScreen();
  }

  maxim_heart_rate_and_oxygen_saturation(
      irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  bootTime = millis();
  currentMode = MODE_MONITOR;
  Serial.println("[INFO] Monitoring started");
}

// ============================================================
// loop（优化：限流防SPI堵塞）
// ============================================================
void loop() {
  // === 实时读取 IR ===
  long irValue = particleSensor.getIR();
  lastIR = irValue;
  fingerDetected = (irValue >= 50000);

  // 心跳检测
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

  // === 波形缓冲更新 ===
  waveBuffer[waveIndex] = map(irValue, 50000, 100000, 0, 50);
  waveIndex = (waveIndex + 1) % WAVE_POINTS;

  // === 显示限流（100ms，防SPI堵塞）===
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