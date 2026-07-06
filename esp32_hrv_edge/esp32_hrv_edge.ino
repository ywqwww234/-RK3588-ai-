/*
 * MindBand · ESP32-C3 边缘智能手环 (TFT版)
 * - MAX30102 PPG 传感器（I2C, SDA=8, SCL=9）
 * - ST7789 1.54" TFT 240x240 (软件SPI)
 * - 三色 LED 风险指示
 *
 * 界面完全参考 d:\0\src\1 中的设计
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"

// ===== WiFi / UDP 配置（按实际网段修改）=====
static const char* WIFI_SSID = "YOUR_SSID";
static const char* WIFI_PASS = "YOUR_PASS";
static const uint16_t UDP_PORT = 5005;
WiFiUDP udp;
IPAddress udpTarget(255, 255, 255, 255);
bool wifiReady = false;

void pcPrint(const char* s) {
  Serial.print(s);
}

void pcPrint(long v) {
  Serial.print(v);
}

void pcPrint(unsigned long v) {
  Serial.print(v);
}

void pcPrint(int v) {
  Serial.print(v);
}

void pcPrintln() {
  Serial.println();
}

void pcPrintln(const char* s) {
  Serial.println(s);
}

// ============================================================
// GPIO 分配
// ============================================================
#define TFT_BL     0
#define TFT_CS     10
#define TFT_DC     3
#define TFT_RST    2
#define TFT_MOSI   4
#define TFT_SCLK   5

#define LED_GREEN   6
#define LED_YELLOW  7
#define LED_RED     1

// ============================================================
// 工业级配色 (RGB565) —— 对标 RUGGED 三防仪表盘
// ============================================================
#define COLOR_BG       0x10A2  // 深邃背景
#define COLOR_PANEL    0x2124  // 面板灰
#define COLOR_PANEL_HI 0x3186  // 面板高亮（顶/底状态栏）
#define COLOR_PANEL_LO 0x18C3  // 面板更暗（凹槽）
#define COLOR_ORANGE   0xFB80  // 工业橙（主数据色）
#define COLOR_CYAN     0x07FF  // 警示青（弧形仪表 / 副数据）
#define COLOR_TEXT     0xFFFF  // 主文字
#define COLOR_DIM      0x8410  // 次级文字
#define COLOR_GREEN    0x07E0
#define COLOR_YELLOW   0xFFE0
#define COLOR_RED      0xF800
// 兼容旧引用
#define COLOR_PRIMARY  COLOR_TEXT
#define COLOR_GRAY     COLOR_PANEL_HI

// ============================================================
// 驱动对象
// ============================================================
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

// Arduino IDE 会自动生成函数声明；显式声明可避免自定义枚举类型
// RiskLevel 在自动声明阶段不可见导致的编译错误。
void showMonitorScreen(int bpm, int spo2, RiskLevel risk);
void showRiskCard(RiskLevel risk, int bpm, int spo2);
void updateLEDs(RiskLevel level);
RiskLevel evaluateRisk(int bpm, int spo2, bool finger);

RiskLevel currentRisk = RISK_NONE;
RiskLevel pendingRisk = RISK_NONE;

unsigned long bootTime = 0;
unsigned long lastDisplayTime = 0;
unsigned long lastRawPrintTime = 0;
unsigned long lastBeatVisual = 0;
unsigned long riskCardStart = 0;
unsigned long riskPendingStart = 0;

const unsigned long DISPLAY_INTERVAL = 100;
const unsigned long RAW_OUTPUT_INTERVAL = 10;  // 100 Hz raw IR for PC-side HRV
const long IR_FINGER_THRESHOLD = 3000;
const long IR_SATURATION_LEVEL = 260000;
const unsigned long RISK_CARD_DURATION = 2800;
const unsigned long RISK_HYSTERESIS = 2500;

bool fingerDetected = false;
bool lastFingerState = false;
int  displayBPM = 0;
int  displaySPO2 = 0;
long lastIR = 0;
uint16_t alertCount = 0;  // 中/高危事件累计（底部 ALERT 徽标）
bool cardDrawn = false;   // 风险卡片是否已绘制

// ============================================================
// 8x8 像素字体定义
// ============================================================
const uint8_t font5x7[][5] PROGMEM = {
  {0x00, 0x00, 0x00, 0x00, 0x00}, // 空格
  {0x00, 0x00, 0x5F, 0x00, 0x00}, // !
  {0x00, 0x07, 0x00, 0x07, 0x00}, // "
  {0x14, 0x7F, 0x7F, 0x14, 0x14}, // #
  {0x24, 0x2E, 0x6B, 0x32, 0x00}, // $
  {0x46, 0x26, 0x10, 0x4D, 0x00}, // %
  {0x30, 0x4D, 0x5B, 0x26, 0x46}, // &
  {0x00, 0x00, 0x07, 0x00, 0x00}, // '
  {0x00, 0x1C, 0x22, 0x41, 0x00}, // (
  {0x00, 0x41, 0x22, 0x1C, 0x00}, // )
  {0x14, 0x08, 0x3E, 0x08, 0x14}, // *
  {0x08, 0x08, 0x3E, 0x08, 0x08}, // +
  {0x00, 0xA0, 0x60, 0x00, 0x00}, // ,
  {0x08, 0x08, 0x08, 0x08, 0x08}, // -
  {0x00, 0x60, 0x60, 0x00, 0x00}, // .
  {0x20, 0x10, 0x08, 0x04, 0x02}, // /
  {0x3E, 0x51, 0x49, 0x45, 0x3E}, // 0
  {0x00, 0x42, 0x7F, 0x40, 0x00}, // 1
  {0x42, 0x61, 0x51, 0x49, 0x46}, // 2
  {0x21, 0x41, 0x45, 0x4B, 0x31}, // 3
  {0x18, 0x14, 0x12, 0x7F, 0x10}, // 4
  {0x27, 0x45, 0x45, 0x45, 0x39}, // 5
  {0x3C, 0x4A, 0x49, 0x49, 0x30}, // 6
  {0x01, 0x71, 0x09, 0x05, 0x03}, // 7
  {0x36, 0x49, 0x49, 0x49, 0x36}, // 8
  {0x06, 0x49, 0x49, 0x29, 0x1E}, // 9
  {0x00, 0x36, 0x36, 0x00, 0x00}, // :
  {0x00, 0xB6, 0xB6, 0x00, 0x00}, // ;
  {0x08, 0x14, 0x22, 0x41, 0x00}, // <
  {0x14, 0x14, 0x14, 0x14, 0x14}, // =
  {0x00, 0x41, 0x22, 0x14, 0x08}, // >
  {0x02, 0x01, 0x51, 0x09, 0x06}, // ?
  {0x3E, 0x41, 0x5D, 0x55, 0x5E}, // @
  {0x7E, 0x11, 0x11, 0x11, 0x7E}, // A
  {0x7F, 0x49, 0x49, 0x49, 0x36}, // B
  {0x3E, 0x41, 0x41, 0x41, 0x22}, // C
  {0x7F, 0x41, 0x41, 0x41, 0x3E}, // D
  {0x7F, 0x49, 0x49, 0x49, 0x41}, // E
  {0x7F, 0x09, 0x09, 0x09, 0x01}, // F
  {0x3E, 0x41, 0x49, 0x49, 0x7A}, // G
  {0x7F, 0x08, 0x08, 0x08, 0x7F}, // H
  {0x00, 0x41, 0x7F, 0x41, 0x00}, // I
  {0x20, 0x40, 0x41, 0x3F, 0x01}, // J
  {0x7F, 0x08, 0x14, 0x22, 0x41}, // K
  {0x7F, 0x40, 0x40, 0x40, 0x40}, // L
  {0x7F, 0x02, 0x0C, 0x02, 0x7F}, // M
  {0x7F, 0x04, 0x08, 0x10, 0x7F}, // N
  {0x3E, 0x41, 0x41, 0x41, 0x3E}, // O
  {0x7F, 0x09, 0x09, 0x09, 0x06}, // P
  {0x3E, 0x41, 0x51, 0x21, 0x5E}, // Q
  {0x7F, 0x09, 0x19, 0x29, 0x46}, // R
  {0x26, 0x49, 0x49, 0x49, 0x32}, // S
  {0x01, 0x01, 0x7F, 0x01, 0x01}, // T
  {0x3F, 0x40, 0x40, 0x40, 0x3F}, // U
  {0x1F, 0x20, 0x40, 0x20, 0x1F}, // V
  {0x3F, 0x40, 0x38, 0x40, 0x3F}, // W
  {0x63, 0x14, 0x08, 0x14, 0x63}, // X
  {0x07, 0x08, 0x70, 0x08, 0x07}, // Y
  {0x61, 0x51, 0x49, 0x45, 0x43}, // Z
  {0x00, 0x7F, 0x41, 0x41, 0x00}, // [
  {0x02, 0x04, 0x08, 0x10, 0x20}, // 反斜杠
  {0x00, 0x41, 0x41, 0x7F, 0x00}, // ]
  {0x04, 0x02, 0x01, 0x02, 0x04}, // ^
  {0x40, 0x40, 0x40, 0x40, 0x40}, // _
  {0x00, 0x00, 0x07, 0x00, 0x00}, // `
  {0x20, 0x54, 0x54, 0x54, 0x78}, // a
  {0x7F, 0x48, 0x44, 0x44, 0x38}, // b
  {0x38, 0x44, 0x44, 0x44, 0x28}, // c
  {0x38, 0x44, 0x44, 0x48, 0x7F}, // d
  {0x38, 0x54, 0x54, 0x54, 0x18}, // e
  {0x08, 0x7E, 0x09, 0x01, 0x02}, // f
  {0x0C, 0x52, 0x52, 0x52, 0x3E}, // g
  {0x7F, 0x08, 0x04, 0x04, 0x78}, // h
  {0x00, 0x44, 0x7D, 0x40, 0x00}, // i
  {0x20, 0x40, 0x44, 0x3D, 0x00}, // j
  {0x7F, 0x10, 0x28, 0x44, 0x00}, // k
  {0x00, 0x41, 0x7F, 0x40, 0x00}, // l
  {0x7C, 0x04, 0x78, 0x04, 0x78}, // m
  {0x7C, 0x08, 0x04, 0x04, 0x78}, // n
  {0x38, 0x44, 0x44, 0x44, 0x38}, // o
  {0x7C, 0x14, 0x14, 0x14, 0x08}, // p
  {0x08, 0x14, 0x14, 0x18, 0x7C}, // q
  {0x7C, 0x08, 0x04, 0x04, 0x08}, // r
  {0x48, 0x54, 0x54, 0x54, 0x20}, // s
  {0x04, 0x3F, 0x44, 0x40, 0x20}, // t
  {0x3C, 0x40, 0x40, 0x20, 0x7C}, // u
  {0x1C, 0x20, 0x40, 0x20, 0x1C}, // v
  {0x3C, 0x40, 0x30, 0x40, 0x3C}, // w
  {0x44, 0x28, 0x10, 0x28, 0x44}, // x
  {0x0C, 0x50, 0x50, 0x50, 0x3C}, // y
  {0x44, 0x64, 0x54, 0x4C, 0x44}, // z
  {0x00, 0x08, 0x36, 0x41, 0x00}, // {
  {0x00, 0x00, 0x7F, 0x00, 0x00}, // |
  {0x00, 0x41, 0x36, 0x08, 0x00}, // }
  {0x08, 0x08, 0x2A, 0x1C, 0x08}, // ~
};

// ============================================================
// TFT 软件SPI函数
// ============================================================
void LCD_Writ_Bus(uint8_t dat) {
  uint8_t i;
  digitalWrite(TFT_CS, LOW);
  for(i = 0; i < 8; i++) {
    digitalWrite(TFT_SCLK, LOW);
    digitalWrite(TFT_MOSI, (dat & 0x80) ? HIGH : LOW);
    digitalWrite(TFT_SCLK, HIGH);
    dat <<= 1;
  }
  digitalWrite(TFT_CS, HIGH);
}

void LCD_WR_DATA8(uint8_t dat) {
  LCD_Writ_Bus(dat);
}

void LCD_WR_DATA(uint16_t dat) {
  LCD_Writ_Bus(dat >> 8);
  LCD_Writ_Bus(dat);
}

void LCD_WR_REG(uint8_t dat) {
  digitalWrite(TFT_DC, LOW);
  LCD_Writ_Bus(dat);
  digitalWrite(TFT_DC, HIGH);
}

void LCD_Address_Set(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2) {
  LCD_WR_REG(0x2a);
  LCD_WR_DATA(x1);
  LCD_WR_DATA(x2);
  LCD_WR_REG(0x2b);
  LCD_WR_DATA(y1 + 80);
  LCD_WR_DATA(y2 + 80);
  LCD_WR_REG(0x2c);
}

void LCD_Fill(uint16_t color) {
  uint16_t i, j;
  LCD_Address_Set(0, 0, 239, 239);
  for(i = 0; i < 240; i++) {
    for(j = 0; j < 240; j++) {
      LCD_WR_DATA(color);
    }
    if((i & 0x0F) == 0) yield();  // ★ 每 16 行喂狗, 防止单核 WDT 复位
  }
}

void LCD_SetPixel(uint16_t x, uint16_t y, uint16_t color) {
  LCD_Address_Set(x, y, x, y);
  LCD_WR_DATA(color);
}

void LCD_Init(void) {
  pinMode(TFT_CS, OUTPUT);
  pinMode(TFT_DC, OUTPUT);
  pinMode(TFT_RST, OUTPUT);
  pinMode(TFT_MOSI, OUTPUT);
  pinMode(TFT_SCLK, OUTPUT);
  pinMode(TFT_BL, OUTPUT);

  digitalWrite(TFT_CS, HIGH);
  digitalWrite(TFT_DC, HIGH);
  digitalWrite(TFT_SCLK, HIGH);
  digitalWrite(TFT_MOSI, LOW);

  digitalWrite(TFT_RST, LOW);
  delay(100);
  digitalWrite(TFT_RST, HIGH);
  delay(100);

  digitalWrite(TFT_BL, HIGH);
  delay(100);

  LCD_WR_REG(0x11);
  delay(120);

  LCD_WR_REG(0x36);
  LCD_WR_DATA8(0xC0);

  LCD_WR_REG(0x3A);
  LCD_WR_DATA8(0x05);

  LCD_WR_REG(0xB2);
  LCD_WR_DATA8(0x0C);
  LCD_WR_DATA8(0x0C);
  LCD_WR_DATA8(0x00);
  LCD_WR_DATA8(0x33);
  LCD_WR_DATA8(0x33);

  LCD_WR_REG(0xB7);
  LCD_WR_DATA8(0x35);

  LCD_WR_REG(0xBB);
  LCD_WR_DATA8(0x32);

  LCD_WR_REG(0xC2);
  LCD_WR_DATA8(0x01);

  LCD_WR_REG(0xC3);
  LCD_WR_DATA8(0x15);

  LCD_WR_REG(0xC4);
  LCD_WR_DATA8(0x20);

  LCD_WR_REG(0xC6);
  LCD_WR_DATA8(0x0F);

  LCD_WR_REG(0xD0);
  LCD_WR_DATA8(0xA4);
  LCD_WR_DATA8(0xA1);

  LCD_WR_REG(0xE0);
  uint8_t gammaE0[] = {0xD0,0x08,0x0E,0x09,0x09,0x05,0x31,0x33,0x48,0x17,0x14,0x15,0x31,0x34};
  for(int i = 0; i < 14; i++) LCD_WR_DATA8(gammaE0[i]);

  LCD_WR_REG(0xE1);
  uint8_t gammaE1[] = {0xD0,0x08,0x0E,0x09,0x09,0x15,0x31,0x33,0x48,0x17,0x14,0x15,0x31,0x34};
  for(int i = 0; i < 14; i++) LCD_WR_DATA8(gammaE1[i]);

  LCD_WR_REG(0x21);
  LCD_WR_REG(0x29);
}

// ============================================================
// 文字绘制函数
// ============================================================
void drawChar(uint16_t x, uint16_t y, char c, uint16_t color, uint16_t bg) {
  if(c < ' ' || c > 'z') c = ' ';
  c -= ' ';
  for(uint8_t i = 0; i < 5; i++) {
    uint8_t line = pgm_read_byte(&font5x7[(uint8_t)c][i]);
    for(uint8_t j = 0; j < 7; j++) {
      if(line & (1 << j)) {
        LCD_SetPixel(x + i, y + j, color);
      }
    }
  }
}

void drawString(uint16_t x, uint16_t y, const char* str, uint16_t color, uint16_t bg) {
  while(*str) {
    drawChar(x, y, *str++, color, bg);
    x += 6;
  }
}

void drawNumber(uint16_t x, uint16_t y, int num, uint16_t color, uint16_t bg) {
  char buf[8];
  itoa(num, buf, 10);
  drawString(x, y, buf, color, bg);
}

// ============================================================
// 矩形绘制
// ============================================================
void drawRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color) {
  for(uint16_t x = x1; x <= x2; x++) {
    LCD_SetPixel(x, y1, color);
    LCD_SetPixel(x, y2, color);
  }
  for(uint16_t y = y1; y <= y2; y++) {
    LCD_SetPixel(x1, y, color);
    LCD_SetPixel(x2, y, color);
  }
}

void fillRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t color) {
  // 软件SPI下 LCD_SetPixel 每像素都重设地址非常慢；
  // 改为一次性 Address_Set + 连续推数据，提速 ~10x
  if (x2 < x1 || y2 < y1) return;
  LCD_Address_Set(x1, y1, x2, y2);
  uint32_t n = (uint32_t)(x2 - x1 + 1) * (uint32_t)(y2 - y1 + 1);
  for(uint32_t i = 0; i < n; i++) {
    LCD_WR_DATA(color);
    if((i & 0x0FFF) == 0) yield();  // ★ 每 4096 像素喂狗
  }
}

void drawHLine(uint16_t x1, uint16_t x2, uint16_t y, uint16_t color) {
  if (x2 < x1) return;
  LCD_Address_Set(x1, y, x2, y);
  for(uint16_t x = x1; x <= x2; x++) LCD_WR_DATA(color);
}

void drawVLine(uint16_t x, uint16_t y1, uint16_t y2, uint16_t color) {
  if (y2 < y1) return;
  LCD_Address_Set(x, y1, x, y2);
  for(uint16_t y = y1; y <= y2; y++) LCD_WR_DATA(color);
}

// ============================================================
// 界面显示函数（工业级优化版本）
// ============================================================

// 绘制圆角矩形
void drawRoundRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t r, uint16_t color) {
  // 上边
  for(uint16_t x = x1 + r; x <= x2 - r; x++) {
    LCD_SetPixel(x, y1, color);
    LCD_SetPixel(x, y2, color);
  }
  // 左边
  for(uint16_t y = y1 + r; y <= y2 - r; y++) {
    LCD_SetPixel(x1, y, color);
    LCD_SetPixel(x2, y, color);
  }
  // 圆角
  for(uint16_t i = 0; i <= r; i++) {
    uint16_t j = sqrt(r * r - i * i);
    LCD_SetPixel(x1 + i, y1 + r - j, color);
    LCD_SetPixel(x2 - i, y1 + r - j, color);
    LCD_SetPixel(x1 + i, y2 - r + j, color);
    LCD_SetPixel(x2 - i, y2 - r + j, color);
  }
}

// 绘制填充圆角矩形
void fillRoundRect(uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2, uint16_t r, uint16_t color) {
  // 填充内部
  for(uint16_t y = y1 + r; y <= y2 - r; y++) {
    for(uint16_t x = x1; x <= x2; x++) {
      LCD_SetPixel(x, y, color);
    }
  }
  // 填充圆角区域
  for(uint16_t i = 0; i <= r; i++) {
    uint16_t j = sqrt(r * r - i * i);
    for(uint16_t y = y1 + r - j; y <= y1 + r + j; y++) {
      LCD_SetPixel(x1 + i, y, color);
      LCD_SetPixel(x2 - i, y, color);
    }
  }
}

// 绘制心形图标
void drawHeart(uint16_t x, uint16_t y, uint16_t size, uint16_t color) {
  int w = size / 2;
  int h = size / 2;
  
  for(int yy = 0; yy <= h; yy++) {
    for(int xx = -w; xx <= w; xx++) {
      float x2 = abs(xx) * 1.0;
      float y2 = yy * 1.0;
      float d = (x2 / w) * (x2 / w) + (y2 / h) * (y2 / h);
      if(d <= 1.0) {
        LCD_SetPixel(x + xx, y + yy, color);
        if(yy > 0) LCD_SetPixel(x + xx, y - yy, color);
      }
    }
  }
}

// 绘制水滴图标（SpO2）
void drawDroplet(uint16_t x, uint16_t y, uint16_t size, uint16_t color) {
  int h = size;
  int w = size / 2;
  
  for(int yy = 0; yy < h; yy++) {
    float ratio = 1.0 - (yy * 1.0 / h);
    int ww = (int)(w * sqrt(ratio));
    for(int xx = -ww; xx <= ww; xx++) {
      LCD_SetPixel(x + xx, y + yy, color);
    }
  }
  // 底部尖端
  LCD_SetPixel(x, y + h - 1, color);
}

// 绘制警告三角图标
void drawWarning(uint16_t x, uint16_t y, uint16_t size, uint16_t color) {
  // 三角形
  for(int yy = 0; yy < size; yy++) {
    int w = yy + 1;
    for(int xx = -w; xx <= w; xx++) {
      LCD_SetPixel(x + xx, y + yy, color);
    }
  }
  // 感叹号
  LCD_SetPixel(x, y + size/3, COLOR_BG);
  LCD_SetPixel(x, y + size/2, COLOR_BG);
}

// 绘制圆形
void drawCircle(uint16_t x, uint16_t y, uint16_t r, uint16_t color) {
  for(int dy = -r; dy <= r; dy++) {
    int dx = sqrt(r * r - dy * dy);
    LCD_SetPixel(x + dx, y + dy, color);
    LCD_SetPixel(x - dx, y + dy, color);
  }
}

// 绘制填充圆形
void fillCircle(uint16_t x, uint16_t y, uint16_t r, uint16_t color) {
  for(int dy = -r; dy <= r; dy++) {
    int dx = sqrt(r * r - dy * dy);
    for(int xx = -dx; xx <= dx; xx++) {
      LCD_SetPixel(x + xx, y + dy, color);
    }
  }
}

// 绘制进度条
void drawProgressBar(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint8_t percent, uint16_t color) {
  fillRoundRect(x, y, x + w, y + h, h/2, COLOR_PANEL_LO);
  int barWidth = (percent * (w - 4)) / 100;
  if (barWidth > 0)
    fillRoundRect(x + 2, y + 2, x + 2 + barWidth, y + h - 2, (h-4)/2, color);
}

// ============================================================
// 工业级 UI 助手
// ============================================================

// 缩放字体绘制（基于 5x7 位图，scale 倍数放大）
void drawCharScaled(uint16_t x, uint16_t y, char c, uint8_t scale, uint16_t color) {
  if(c < ' ' || c > 'z') c = ' ';
  c -= ' ';
  for(uint8_t cx = 0; cx < 5; cx++) {
    uint8_t line = pgm_read_byte(&font5x7[(uint8_t)c][cx]);
    for(uint8_t cy = 0; cy < 7; cy++) {
      if(line & (1 << cy)) {
        if(scale == 1) {
          LCD_SetPixel(x + cx, y + cy, color);
        } else {
          fillRect(x + cx*scale, y + cy*scale,
                   x + cx*scale + scale - 1, y + cy*scale + scale - 1, color);
        }
      }
    }
  }
}

void drawStringScaled(uint16_t x, uint16_t y, const char* s, uint8_t scale, uint16_t color) {
  while(*s) {
    drawCharScaled(x, y, *s++, scale, color);
    x += 6 * scale;
  }
}

uint16_t textWidthScaled(const char* s, uint8_t scale) {
  uint16_t n = 0;
  while(*s++) n++;
  return n * 6 * scale;
}

// 绘制弧形仪表盘环段（degrees: 0 = 顶部，顺时针）
// rOuter / rInner 描边粗细；start/end 角度区间
void drawArc(int cx, int cy, int rOuter, int rInner,
             int startAngle, int endAngle, uint16_t color) {
  if(endAngle <= startAngle) return;
  for(int a = startAngle; a <= endAngle; a++) {
    float rad = (a - 90) * 0.017453293f;
    float c_ = cosf(rad);
    float s_ = sinf(rad);
    for(int r = rInner; r <= rOuter; r++) {
      int x = cx + (int)(c_ * r);
      int y = cy + (int)(s_ * r);
      if(x >= 0 && x < 240 && y >= 0 && y < 240)
        LCD_SetPixel(x, y, color);
    }
  }
}

// 电池图标（pct 0..100）
void drawBattery(int x, int y, int pct) {
  drawRect(x, y, x+22, y+11, COLOR_TEXT);
  fillRect(x+22, y+3, x+24, y+8, COLOR_TEXT);
  int fillW = (pct * 18) / 100;
  if(fillW < 0) fillW = 0;
  if(fillW > 18) fillW = 18;
  uint16_t c = (pct > 30) ? COLOR_GREEN : (pct > 15 ? COLOR_YELLOW : COLOR_RED);
  if(fillW > 0) fillRect(x+2, y+2, x+2+fillW, y+9, c);
}

// 信号格条（4 格）
void drawSignalBars(int x, int y, int level, uint16_t color) {
  for(int i = 0; i < 4; i++) {
    int h = 2 + i * 2;
    if(i < level) fillRect(x + i*3, y + 8 - h, x + i*3 + 2, y + 8, color);
    else drawRect(x + i*3, y + 8 - h, x + i*3 + 2, y + 8, COLOR_DIM);
  }
}

// 蓝牙小图标（10x12 简化版）
void drawBT(int x, int y, uint16_t color) {
  drawVLine(x+4, y, y+11, color);
  // 上三角
  for(int i = 0; i < 4; i++) {
    LCD_SetPixel(x + 4 + i, y + i, color);
    LCD_SetPixel(x + 4 + i, y + 11 - i, color);
  }
  // 下三角交点
  LCD_SetPixel(x+1, y+3, color);
  LCD_SetPixel(x+2, y+4, color);
  LCD_SetPixel(x+3, y+5, color);
  LCD_SetPixel(x+1, y+8, color);
  LCD_SetPixel(x+2, y+7, color);
  LCD_SetPixel(x+3, y+6, color);
}

// 带括号的工业标签 "[LABEL]"
void drawBracketLabel(int cx, int y, const char* label, uint16_t color, uint8_t scale) {
  uint16_t w = textWidthScaled(label, scale);
  int x = cx - (w + 12*scale) / 2;
  drawStringScaled(x, y, "[", scale, color);
  drawStringScaled(x + 6*scale, y, label, scale, color);
  drawStringScaled(x + 6*scale + w, y, "]", scale, color);
}

// 工业切角矩形（右下角斜切，参考图风格）
void drawAngleCutPanel(int x1, int y1, int x2, int y2, int cut, uint16_t fill) {
  fillRect(x1, y1, x2, y2 - cut, fill);
  for(int i = 0; i < cut; i++) {
    drawHLine(x1, x2 - i, y2 - cut + i, fill);
  }
}

// 绘制一个工业仪表盘 (icon + value + label) at center cx, cy with radius r
// arcPct: 0..100 决定亮弧覆盖角度；color = 主色
void drawGauge(int cx, int cy, int r, int arcPct,
               uint16_t arcColor, const char* value, const char* label,
               uint8_t valScale) {
  // 暗底环
  drawArc(cx, cy, r, r - 4, -120, 120, COLOR_PANEL);
  // 亮弧（从 -120 度起 240 度全角范围按 pct 切分）
  int sweep = (arcPct * 240) / 100;
  if(sweep > 240) sweep = 240;
  if(sweep > 0) drawArc(cx, cy, r, r - 4, -120, -120 + sweep, arcColor);

  // 中心数值
  uint16_t vw = textWidthScaled(value, valScale);
  drawStringScaled(cx - vw/2, cy - 4*valScale, value, valScale, COLOR_TEXT);

  // 底部标签
  uint16_t lw = textWidthScaled(label, 1);
  drawStringScaled(cx - lw/2, cy + r - 10, label, 1, COLOR_DIM);
}

// ============================================================
// 屏幕渲染（工业级）
// ============================================================

// 启动画面 (工业风)
void showStartupScreen(int progress) {
  LCD_Fill(COLOR_BG);

  // 顶部状态条
  fillRect(0, 0, 239, 18, COLOR_PANEL_HI);
  drawString(6, 6, "ESP-FIT / BOOT", COLOR_CYAN, COLOR_PANEL_HI);
  drawString(170, 6, "X-1000 V1.2", COLOR_DIM, COLOR_PANEL_HI);

  // 主标题面板
  fillRect(10, 36, 229, 110, COLOR_PANEL);
  drawHLine(10, 229, 36, COLOR_ORANGE);
  drawHLine(10, 229, 110, COLOR_PANEL_HI);

  // 大标题
  uint16_t tw = textWidthScaled("MINDBAND", 3);
  drawStringScaled(120 - tw/2, 52, "MINDBAND", 3, COLOR_TEXT);
  uint16_t sw = textWidthScaled("EDGE AI WEARABLE", 1);
  drawStringScaled(120 - sw/2, 90, "EDGE AI WEARABLE", 1, COLOR_DIM);

  // 进度条
  fillRect(20, 140, 219, 162, COLOR_PANEL_LO);
  int barW = (progress * 195) / 100;
  if(barW > 0) fillRect(22, 142, 22 + barW, 160, COLOR_ORANGE);

  // 进度数字
  char buf[8];
  itoa(progress, buf, 10);
  strcat(buf, "%");
  uint16_t pw = textWidthScaled(buf, 2);
  drawStringScaled(120 - pw/2, 174, buf, 2, COLOR_CYAN);

  // 阶段标签
  const char* status;
  if(progress < 30) status = ">> INIT I2C BUS";
  else if(progress < 60) status = ">> CALIBRATING SENSOR";
  else if(progress < 95) status = ">> LOADING EDGE MODEL";
  else status = ">> SYSTEM READY";
  uint16_t stw = textWidthScaled(status, 1);
  drawStringScaled(120 - stw/2, 208, status, 1, COLOR_GREEN);

  // 底部装饰条
  drawHLine(0, 239, 228, COLOR_PANEL_HI);
  drawString(6, 232, "[POWER ON SELF TEST]", COLOR_DIM, COLOR_BG);
}

// 等待画面 (工业风)
void showWaitingScreen() {
  LCD_Fill(COLOR_BG);

  // === 顶部状态栏 ===
  fillRect(0, 0, 239, 22, COLOR_PANEL_HI);
  drawBT(4, 5, COLOR_CYAN);
  drawString(18, 8, "ESP-FIT", COLOR_CYAN, COLOR_PANEL_HI);
  drawString(60, 8, "/ RUGGED", COLOR_DIM, COLOR_PANEL_HI);
  drawSignalBars(116, 6, 3, COLOR_TEXT);
  // 简单运行计数
  unsigned long t = millis() / 1000;
  char tStr[10];
  sprintf(tStr, "%02lu:%02lu", (t / 60) % 60, t % 60);
  drawString(140, 8, tStr, COLOR_TEXT, COLOR_PANEL_HI);
  drawBattery(210, 6, 96);

  // === 中央等待面板 ===
  fillRect(10, 32, 229, 152, COLOR_PANEL);
  drawHLine(10, 229, 32, COLOR_ORANGE);
  drawHLine(10, 229, 152, COLOR_PANEL_HI);

  // 大字提示
  uint16_t w = textWidthScaled("PLACE FINGER", 2);
  drawStringScaled(120 - w/2, 50, "PLACE FINGER", 2, COLOR_TEXT);
  uint16_t w2 = textWidthScaled("ON SENSOR", 2);
  drawStringScaled(120 - w2/2, 76, "ON SENSOR", 2, COLOR_ORANGE);

  // IR 信号条（实时反馈）
  drawString(20, 110, "IR SIGNAL", COLOR_DIM, COLOR_PANEL);
  // IR 范围 ~5000..200000 → 映射 0..100
  long ir = lastIR;
  int pct = (int)((ir - 5000L) * 100L / 195000L);
  if(pct < 0) pct = 0; if(pct > 100) pct = 100;
  fillRect(20, 124, 219, 138, COLOR_PANEL_LO);
  if(pct > 0) fillRect(22, 126, 22 + (pct*195)/100, 136,
                      (ir >= 50000) ? COLOR_GREEN : COLOR_ORANGE);
  // 数值
  char irBuf[16];
  ltoa(ir, irBuf, 10);
  drawString(20, 140, "VAL:", COLOR_DIM, COLOR_PANEL);
  drawString(48, 140, irBuf, COLOR_CYAN, COLOR_PANEL);

  // === 4 个图标瓦片 ===
  const char* tiles[4] = {"TRAIN", "SENSE", "MSG", "SYS"};
  for(int i = 0; i < 4; i++) {
    int tx = 4 + i * 59;
    fillRect(tx, 162, tx + 55, 200, COLOR_PANEL);
    drawHLine(tx, tx + 55, 162, COLOR_PANEL_HI);
    // 小图标占位（简单方块 + 标签）
    fillRect(tx + 24, 170, tx + 32, 178, (i == 1) ? COLOR_ORANGE : COLOR_DIM);
    uint16_t lw = textWidthScaled(tiles[i], 1);
    drawBracketLabel(tx + 28, 188, tiles[i], COLOR_DIM, 1);
    (void)lw;
  }

  // === 底部状态栏 ===
  static bool blink = false;
  blink = !blink;
  fillRect(0, 208, 239, 239, COLOR_PANEL_HI);
  drawBT(6, 218, blink ? COLOR_CYAN : COLOR_DIM);
  drawString(20, 220, "[BLE 5.0]", blink ? COLOR_CYAN : COLOR_DIM, COLOR_PANEL_HI);
  drawString(96, 220, "[NO SIGNAL]", COLOR_YELLOW, COLOR_PANEL_HI);
  drawString(180, 220, "ALERT", COLOR_DIM, COLOR_PANEL_HI);
  fillRect(218, 218, 234, 232, COLOR_ORANGE);
  drawString(223, 221, "0", COLOR_BG, COLOR_ORANGE);
}

// 监测画面 (工业仪表盘)
//   y=0..22   顶部状态栏（蓝牙/品牌/信号/时间/电池）
//   y=24..86  时钟面板（大时间 + 型号信息）
//   y=88..156 三仪表盘（BPM / SpO2 / RISK）
//   y=158..200 四个 [BRACKETED] 瓦片
//   y=202..239 底部状态条（BLE/手指/告警）
void showMonitorScreen(int bpm, int spo2, RiskLevel risk) {
  LCD_Fill(COLOR_BG);

  // === 顶部状态栏 ===
  fillRect(0, 0, 239, 22, COLOR_PANEL_HI);
  drawBT(4, 5, COLOR_CYAN);
  drawString(18, 8, "ESP-FIT", COLOR_CYAN, COLOR_PANEL_HI);
  drawString(60, 8, "/ RUGGED", COLOR_DIM, COLOR_PANEL_HI);
  drawSignalBars(116, 6, 4, COLOR_TEXT);
  unsigned long t = (millis() - bootTime) / 1000;
  char hms[10];
  sprintf(hms, "%02lu:%02lu:%02lu", t / 3600, (t / 60) % 60, t % 60);
  drawString(132, 8, hms, COLOR_ORANGE, COLOR_PANEL_HI);
  drawBattery(210, 6, 96);

  // === 时钟面板 ===
  fillRect(0, 24, 239, 86, COLOR_PANEL);
  drawHLine(0, 239, 24, COLOR_ORANGE);
  drawHLine(0, 239, 86, COLOR_PANEL_HI);
  // 大时间
  char bigT[8];
  sprintf(bigT, "%02lu:%02lu", (t / 3600), (t / 60) % 60);
  drawStringScaled(20, 34, bigT, 4, COLOR_TEXT);
  drawStringScaled(20, 70, "RUN", 1, COLOR_DIM);
  drawStringScaled(46, 70, hms, 1, COLOR_DIM);
  // 右侧型号
  drawString(160, 32, "MODEL", COLOR_DIM, COLOR_PANEL);
  drawStringScaled(160, 44, "X-1000", 1, COLOR_CYAN);
  drawString(160, 58, "FW v1.2", COLOR_DIM, COLOR_PANEL);
  drawString(160, 70, "ESP32-C3", COLOR_DIM, COLOR_PANEL);

  // === 三仪表盘 ===
  // 弧形仪表盘中心 y=125，半径 30；x 中心 = 44, 120, 196
  // BPM 仪表
  int bpmPct = (bpm > 0) ? (bpm - 40) * 100 / 100 : 0;  // 40..140 → 0..100
  if(bpmPct < 0) bpmPct = 0; if(bpmPct > 100) bpmPct = 100;
  char bpmS[8]; if(bpm > 0) itoa(bpm, bpmS, 10); else strcpy(bpmS, "--");
  bool beating = (millis() - lastBeatVisual) < 200;
  uint16_t bpmCol = beating ? COLOR_ORANGE : COLOR_CYAN;
  drawGauge(44, 125, 32, bpmPct, bpmCol, bpmS, "BPM", 2);
  // 心形小图标
  if(beating) drawHeart(44, 100, 8, COLOR_RED);

  // SpO2 仪表
  int spo2Pct = (spo2 > 0) ? (spo2 - 70) * 100 / 30 : 0;  // 70..100 → 0..100
  if(spo2Pct < 0) spo2Pct = 0; if(spo2Pct > 100) spo2Pct = 100;
  char spS[8]; if(spo2 > 0) itoa(spo2, spS, 10); else strcpy(spS, "--");
  uint16_t spCol = (spo2 >= 95) ? COLOR_GREEN : (spo2 >= 90 ? COLOR_YELLOW : COLOR_RED);
  if(spo2 <= 0) spCol = COLOR_DIM;
  drawGauge(120, 125, 32, spo2Pct, spCol, spS, "SpO2", 2);
  drawDroplet(120, 100, 8, COLOR_CYAN);

  // RISK 仪表
  int riskPct = (int)risk * 33;
  uint16_t rCol;
  const char* rTag;
  switch(risk) {
    case RISK_LOW:  rCol = COLOR_GREEN;  rTag = "OK";   break;
    case RISK_MID:  rCol = COLOR_YELLOW; rTag = "WARN"; break;
    case RISK_HIGH: rCol = COLOR_RED;    rTag = "HIGH"; break;
    default:        rCol = COLOR_DIM;    rTag = "--";   break;
  }
  drawGauge(196, 125, 32, riskPct, rCol, rTag, "RISK", 2);

  // === 4 个工业瓦片 ===
  const char* tiles[4] = {"TRAIN", "SENSE", "MSG", "SYS"};
  uint16_t tileColors[4] = {COLOR_ORANGE, COLOR_CYAN, COLOR_DIM, COLOR_DIM};
  for(int i = 0; i < 4; i++) {
    int tx = 4 + i * 59;
    fillRect(tx, 162, tx + 55, 200, COLOR_PANEL);
    drawHLine(tx, tx + 55, 162, tileColors[i]);
    fillRect(tx + 24, 170, tx + 32, 178, tileColors[i]);
    drawBracketLabel(tx + 28, 188, tiles[i], tileColors[i], 1);
  }

  // === 底部状态栏 ===
  fillRect(0, 202, 239, 239, COLOR_PANEL_HI);
  drawHLine(0, 239, 202, COLOR_ORANGE);
  drawBT(6, 212, COLOR_CYAN);
  drawString(20, 215, "[BLE 5.0]", COLOR_CYAN, COLOR_PANEL_HI);
  // 手指/信号状态
  const char* fingerS = fingerDetected ? "[FINGER OK]" : "[NO FINGER]";
  uint16_t fc = fingerDetected ? COLOR_GREEN : COLOR_DIM;
  drawString(86, 215, fingerS, fc, COLOR_PANEL_HI);
  // 告警计数
  drawString(170, 215, "ALERT", COLOR_DIM, COLOR_PANEL_HI);
  fillRect(208, 213, 232, 227, COLOR_ORANGE);
  char abuf[6];
  itoa(alertCount > 99 ? 99 : alertCount, abuf, 10);
  uint16_t aw = textWidthScaled(abuf, 1);
  drawString(220 - aw/2, 217, abuf, COLOR_BG, COLOR_ORANGE);
  // 风险横条
  drawString(6, 230, ">>", rCol, COLOR_PANEL_HI);
  uint16_t rw = textWidthScaled(rTag, 1);
  drawString(20, 230, rTag, rCol, COLOR_PANEL_HI);
}

// 风险告警卡 (工业风)
void showRiskCard(RiskLevel risk, int bpm, int spo2) {
  LCD_Fill(COLOR_BG);

  uint16_t accent;
  const char* tag;
  const char* line1;
  const char* line2;
  switch(risk) {
    case RISK_MID:
      accent = COLOR_YELLOW; tag = "MID";
      line1 = "ELEVATED RISK"; line2 = "SLOW & BREATHE";
      break;
    case RISK_HIGH:
      accent = COLOR_RED; tag = "HIGH";
      line1 = "ATTENTION REQUIRED"; line2 = "REST IMMEDIATELY";
      break;
    default:
      accent = COLOR_GREEN; tag = "LOW";
      line1 = "STATUS STEADY"; line2 = "ALL SYSTEMS OK";
      break;
  }

  // 外双框（工业告警感）
  drawRect(4, 4, 235, 235, accent);
  drawRect(7, 7, 232, 232, accent);

  // 顶部告警标签条
  fillRect(10, 10, 229, 38, accent);
  drawStringScaled(20, 18, "RISK LEVEL", 1, COLOR_BG);
  drawStringScaled(165, 18, "ALERT", 1, COLOR_BG);
  // 闪烁圆点
  static bool flash = false;
  flash = !flash;
  if(flash && risk == RISK_HIGH) fillCircle(212, 24, 6, COLOR_BG);
  else drawCircle(212, 24, 6, COLOR_BG);

  // 大字等级标识
  uint16_t tw = textWidthScaled(tag, 5);
  fillRect(10, 48, 229, 132, COLOR_PANEL);
  drawHLine(10, 229, 48, accent);
  drawHLine(10, 229, 132, COLOR_PANEL_HI);
  drawStringScaled(120 - tw/2, 64, tag, 5, accent);

  // 闪烁外框（HIGH）
  if(risk == RISK_HIGH && flash) {
    drawRect(0, 0, 239, 239, COLOR_RED);
    drawRect(1, 1, 238, 238, COLOR_RED);
    drawRect(2, 2, 237, 237, COLOR_RED);
  }

  // 提示文字
  uint16_t lw1 = textWidthScaled(line1, 1);
  drawStringScaled(120 - lw1/2, 142, line1, 1, COLOR_TEXT);
  uint16_t lw2 = textWidthScaled(line2, 1);
  drawStringScaled(120 - lw2/2, 158, line2, 1, COLOR_DIM);

  // 数据卡（HR + SpO2 一组）
  fillRect(10, 176, 119, 222, COLOR_PANEL);
  fillRect(121, 176, 229, 222, COLOR_PANEL);
  drawHLine(10, 119, 176, COLOR_ORANGE);
  drawHLine(121, 229, 176, COLOR_CYAN);
  drawString(20, 184, "HR", COLOR_ORANGE, COLOR_PANEL);
  drawString(131, 184, "SpO2", COLOR_CYAN, COLOR_PANEL);
  char b1[8], b2[8];
  if(bpm > 0) itoa(bpm, b1, 10); else strcpy(b1, "--");
  if(spo2 > 0) sprintf(b2, "%d%%", spo2); else strcpy(b2, "--");
  drawStringScaled(20, 198, b1, 2, COLOR_TEXT);
  drawStringScaled(131, 198, b2, 2, COLOR_TEXT);

  // 底部说明
  drawString(8, 228, "[AUTO RECOVERY IN 2.8s]", COLOR_DIM, COLOR_BG);
}

// ============================================================
// 诊断 & 日志
// ============================================================
unsigned long lastLoopLogMs = 0;
unsigned long maxLoopUs = 0;
unsigned long loopCount = 0;
unsigned long wdtHeartbeat = 0;  // 喂狗计数, PC 端可据此判断 ESP32 存活

void logResetReason() {
  int reason = (int)esp_reset_reason();
  Serial.printf("[BOOT] Reset reason = %d: ", reason);
  switch(reason) {
    case 1:  Serial.println("POWERON_RESET"); break;
    case 2:  Serial.println("EXT_RESET (pin)"); break;
    case 3:  Serial.println("SW_RESET"); break;
    case 4:  Serial.println("OWDT_RESET (panic)"); break;
    case 5:  Serial.println("DEEPSLEEP_RESET"); break;
    case 6:  Serial.println("SDIO_RESET"); break;
    case 7:  Serial.println("TG0WDT_SYS_RESET (task WDT!)"); break;
    case 8:  Serial.println("TG1WDT_SYS_RESET"); break;
    case 9:  Serial.println("RTCWDT_SYS_RESET"); break;
    case 12: Serial.println("BROWNOUT_RESET (power!)"); break;
    default: Serial.println("UNKNOWN"); break;
  }
}

void logError(const char* tag, const char* msg) {
  unsigned long t = millis();
  Serial.printf("[ERR  %6lu.%03lu] %s: %s\n", t/1000, t%1000, tag, msg);
}

// ============================================================
// LED控制
// ============================================================
void setLEDs(bool g, bool y, bool r) {
  digitalWrite(LED_GREEN, g ? HIGH : LOW);
  digitalWrite(LED_YELLOW, y ? HIGH : LOW);
  digitalWrite(LED_RED, r ? HIGH : LOW);
}

void updateLEDs(RiskLevel level) {
  switch(level) {
    case RISK_LOW:  setLEDs(true, false, false); break;
    case RISK_MID:  setLEDs(false, true, false); break;
    case RISK_HIGH: setLEDs(false, false, true); break;
    default:        setLEDs(false, false, false); break;
  }
}

RiskLevel evaluateRisk(int bpm, int spo2, bool finger) {
  if(!finger || bpm <= 0) return RISK_NONE;

  int score = 0;
  if(bpm < 45 || bpm > 130) score += 3;
  else if(bpm < 55 || bpm > 110) score += 2;
  else if(bpm < 60 || bpm > 100) score += 1;

  if(spo2 > 0) {
    if(spo2 < 90) score += 3;
    else if(spo2 < 93) score += 2;
    else if(spo2 < 95) score += 1;
  }

  if(score >= 4) return RISK_HIGH;
  if(score >= 2) return RISK_MID;
  return RISK_LOW;
}

// ============================================================
// setup
// ============================================================
void setup() {
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  setLEDs(false, false, false);

  Serial.begin(115200);
  // Serial.setTxTimeoutMs(0) 不可用(旧版arduino-esp32), ESP32 USB-CDC默认已非阻塞
  delay(200);
  Serial.println("\n========================================");
  Serial.println("[BOOT] === MindBand Booting ===");
  logResetReason();
  Serial.printf("[BOOT] Free heap: %u bytes\n", ESP.getFreeHeap());
  Serial.printf("[BOOT] CPU freq: %u MHz\n", ESP.getCpuFreqMHz());
  Serial.println("========================================");

  // 占位 SSID 时彻底关掉 WiFi: 省电 + 去掉后台重连负载, 长时间运行更稳
  if(strcmp(WIFI_SSID, "YOUR_SSID") != 0) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    unsigned long wt0 = millis();
    while(WiFi.status() != WL_CONNECTED && millis() - wt0 < 8000) delay(200);
    if(WiFi.status() == WL_CONNECTED) {
      wifiReady = true;
      udp.begin(UDP_PORT);
      IPAddress ip = WiFi.localIP();
      udpTarget = IPAddress(ip[0], ip[1], ip[2], 255);
      Serial.print("[INFO] WiFi OK IP="); Serial.print(ip);
      Serial.print(" -> "); Serial.print(udpTarget);
      Serial.print(":"); Serial.println(UDP_PORT);
    } else {
      Serial.println("[WARN] WiFi connect failed, UDP disabled");
    }
  } else {
    WiFi.mode(WIFI_OFF);
    Serial.println("[INFO] WiFi off (SSID not configured)");
  }

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

  Serial.println("[INFO] Initializing I2C...");
  Wire.begin(8, 9);
  delay(100);
  Serial.println("[INFO] I2C initialized");

  Serial.println("[INFO] Initializing MAX30102...");
  if(!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERR] MAX30102 init failed — check I2C wiring (SDA=8,SCL=9)");
    // 闪5次红灯告警, 然后继续(串口仍然可用, 方便调试)
    for(int i=0;i<5;i++){digitalWrite(LED_RED,HIGH);delay(200);digitalWrite(LED_RED,LOW);delay(200);}
    // 不卡死, fallback到仅串口输出模式(IR=0)
  } else {
    // Lower LED current and larger ADC range prevent IR saturation at 262143.
    // Parameters: ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange.
    particleSensor.setup(30, 4, 2, 100, 411, 16384);
    Serial.println("[INFO] MAX30102 ready");
  }

  Serial.println("[INFO] Initializing TFT...");
  LCD_Init();
  LCD_Fill(COLOR_BG);
  Serial.println("[INFO] TFT initialized");

  for(int p = 0; p <= 100; p += 10) {
    showStartupScreen(p);
    delay(100);
  }

  Serial.println("[INFO] Calibrating...");
  for(byte i = 0; i < bufferLength; i++) {
    unsigned long t0 = millis();
    while(!particleSensor.available()) {
      particleSensor.check();
      if(millis() - t0 > 100) break;  // 校准阶段100ms超时
    }
    if(particleSensor.available()) {
      redBuffer[i] = particleSensor.getRed();
      irBuffer[i] = particleSensor.getIR();
      particleSensor.nextSample();
    }
  }

  maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer,
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
  delay(2);  // 让出CPU喂IDLE任务: 单核C3上防止Task WDT超时复位(长时间运行关键)
  long irValue = particleSensor.getIR();
  lastIR = irValue;
  bool irSaturated = (irValue >= IR_SATURATION_LEVEL);
  fingerDetected = (irValue >= IR_FINGER_THRESHOLD) && !irSaturated;

  unsigned long rawNow = millis();
  if(rawNow - lastRawPrintTime >= RAW_OUTPUT_INTERVAL) {
    lastRawPrintTime = rawNow;
    pcPrint("[RAW] ir=");
    pcPrint(irValue);
    pcPrint(", finger=");
    pcPrint(fingerDetected ? 1 : 0);
    pcPrint(", sat=");
    pcPrint(irSaturated ? 1 : 0);
    pcPrintln();
  }

  if(checkForBeat(irValue) == true) {
    long delta = millis() - lastBeat;
    lastBeat = millis();
    lastBeatVisual = millis();

    beatsPerMinute = 60 / (delta / 1000.0);
    if(beatsPerMinute < 220 && beatsPerMinute > 30) {
      rates[rateSpot++] = (byte)beatsPerMinute;
      rateSpot %= RATE_SIZE;
      beatAvg = 0;
      for(byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
      beatAvg /= RATE_SIZE;
    }
  }

  for(byte i = 25; i < 100; i++) {
    redBuffer[i - 25] = redBuffer[i];
    irBuffer[i - 25] = irBuffer[i];
  }
  {
    // 总超时保护: 整个 25 点填充循环最多 250ms, 防止 WDT 复位
    unsigned long refillDeadline = millis() + 250;
    for(byte i = 75; i < 100; i++) {
      // 无手指时降低超时: 传感器 FIFO 几乎无数据, 不必久等
      unsigned long perSampleTimeout = fingerDetected ? 40 : 15;
      unsigned long t0 = millis();
      while(!particleSensor.available()) {
        particleSensor.check();
        yield();  // ★ 喂 IDLE 任务, 防止单核 WDT
        if(millis() - t0 > perSampleTimeout) break;
        if(millis() > refillDeadline) break;  // 总超时
      }
      if(millis() > refillDeadline) break;  // 总超时: 放弃本轮剩余采样
      if(particleSensor.available()) {
        redBuffer[i] = particleSensor.getRed();
        irBuffer[i] = particleSensor.getIR();
        particleSensor.nextSample();
      }
    }
  }

  maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer,
      &spo2, &validSPO2, &heartRate, &validHeartRate);

  unsigned long now = millis();

  // ★ 喂狗心跳: 每次 loop 递增, 超过 5 秒无变化说明死锁
  wdtHeartbeat++;

  // ★ 每 10 秒打印一次 loop 诊断 (不影响数据通道)
  if(now - lastLoopLogMs >= 10000) {
    Serial.printf("[DIAG] loop_cnt=%lu max_loop_us=%lu heap=%u\n",
                  loopCount, maxLoopUs, ESP.getFreeHeap());
    lastLoopLogMs = now;
    maxLoopUs = 0;
  }

  if(now - lastDisplayTime < DISPLAY_INTERVAL) return;
  lastDisplayTime = now;

  int prevDisplayBPM = displayBPM;
  int prevDisplaySPO2 = displaySPO2;
  RiskLevel prevDisplayRisk = currentRisk;
  bool prevFinger = lastFingerState;

  displayBPM = (validHeartRate && heartRate > 0 && heartRate < 220) ? heartRate : beatAvg;
  displaySPO2 = (validSPO2 && spo2 > 0 && spo2 <= 100) ? spo2 : 0;

  RiskLevel newRisk = evaluateRisk(displayBPM, displaySPO2, fingerDetected);
  if(newRisk != currentRisk) {
    if(newRisk != pendingRisk) {
      pendingRisk = newRisk;
      riskPendingStart = now;
    } else if(now - riskPendingStart >= RISK_HYSTERESIS) {
      RiskLevel prev = currentRisk;
      currentRisk = newRisk;
      pendingRisk = newRisk;

      if(fingerDetected && currentRisk != RISK_NONE
          && !(prev == RISK_NONE && currentRisk == RISK_LOW)) {
        currentMode = MODE_RISK_CARD;
        riskCardStart = now;
        if(currentRisk == RISK_MID || currentRisk == RISK_HIGH) alertCount++;
      }
    }
  } else {
    pendingRisk = newRisk;
    riskPendingStart = now;
  }

  updateLEDs(currentRisk);

  // 串口协议（兼容 PPG_ESP32_OLED.ino 边缘格式：含 avg 与 NoFinger 标志）
  pcPrint("[DATA] ir=");
  pcPrint(irValue);
  pcPrint(", bpm=");
  pcPrint(displayBPM);
  pcPrint(", avg=");
  pcPrint(beatAvg);
  pcPrint(", spo2=");
  pcPrint(displaySPO2);
  pcPrint(", finger=");
  pcPrint(fingerDetected ? 1 : 0);
  pcPrint(", sat=");
  pcPrint(irSaturated ? 1 : 0);
  pcPrint(", risk=");
  pcPrint((int)currentRisk);
  pcPrint(", alerts=");
  pcPrint(alertCount);
  pcPrint(", hb=");
  pcPrint(wdtHeartbeat);
  if(!fingerDetected) pcPrint(" NoFinger");
  pcPrintln();
  // Serial.flush() is intentionally avoided here. On ESP32-C3 USB CDC it may
  // block when the host is busy and make the PC-side UI appear frozen.

  // UDP 广播相同 [DATA] 包到当前网段
  if(wifiReady && WiFi.status() == WL_CONNECTED) {
    char pkt[192];
    snprintf(pkt, sizeof(pkt),
      "[DATA] ir=%ld, bpm=%d, avg=%d, spo2=%d, finger=%d, sat=%d, risk=%d, alerts=%u, hb=%lu%s",
      irValue, displayBPM, beatAvg, displaySPO2,
      fingerDetected ? 1 : 0, irSaturated ? 1 : 0, (int)currentRisk, alertCount,
      wdtHeartbeat,
      fingerDetected ? "" : " NoFinger");
    udp.beginPacket(udpTarget, UDP_PORT);
    udp.print(pkt);
    udp.endPacket();
  }

  // ===== 12s 显示序列：8s 心率图(MONITOR) + 4s 弹窗(RISK_CARD) =====
  // ★ 关键优化: 只在数据变化时才重绘, 避免软件 SPI 长时间阻塞导致 WDT
  static bool waitingDrawn = false;
  static bool seqActive = false;
  static unsigned long seqStart = 0;
  const unsigned long SEQ_TOTAL    = 12000;
  const unsigned long SEQ_MONITOR  = 8000;

  if(!fingerDetected) {
    seqActive = false;
    if(!waitingDrawn) {
      showWaitingScreen();
      waitingDrawn = true;
      lastFingerState = false;
    }
    yield();  // 让出 CPU
    return;
  }

  // 首次检测到手指 → 启动 12s 序列
  if(!seqActive) {
    seqActive = true;
    seqStart = now;
    waitingDrawn = false;
    cardDrawn = false;  // ★ 新序列重置卡片状态
  }

  bool displayChanged = (displayBPM != prevDisplayBPM)
                     || (displaySPO2 != prevDisplaySPO2)
                     || (currentRisk != prevDisplayRisk)
                     || (fingerDetected != prevFinger);

  unsigned long elapsed = now - seqStart;
  if(elapsed < SEQ_MONITOR) {
    if(displayChanged) {
      showMonitorScreen(displayBPM, displaySPO2, currentRisk);
    }
  } else if(elapsed < SEQ_TOTAL) {
    // 风险卡片只在首次进入时绘制, 不重复刷新
    if(!cardDrawn || displayChanged) {
      showRiskCard(currentRisk, displayBPM, displaySPO2);
      cardDrawn = true;
    }
  } else {
    // 12s 完成 → 回到初始界面（保持静态）
    seqActive = false;
    cardDrawn = false;
    showWaitingScreen();
    waitingDrawn = true;
  }
  lastFingerState = fingerDetected;

  yield();  // ★ 每次 loop 结束前让出 CPU, 确保 IDLE 任务运行
}
