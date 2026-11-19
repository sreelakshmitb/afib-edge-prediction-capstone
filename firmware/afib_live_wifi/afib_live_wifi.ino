// ESP32-WROOM-32 / MAX30003, continuous battery-powered Wi-Fi ECG stream.
// Flash this sketch for live mode; the existing SD replay sketch remains separate.
#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>

// Edit these three values before flashing. The Pi and ESP32 use the same Wi-Fi.
const char *WIFI_SSID = "YOUR_HOTSPOT_NAME";
const char *WIFI_PASSWORD = "YOUR_HOTSPOT_PASSWORD";
IPAddress PI_IP(10, 113, 208, 214); // Pi's current Wi-Fi address
constexpr uint16_t PI_PORT = 8767;

constexpr uint8_t SCK_PIN = 18, MISO_PIN = 19, MOSI_PIN = 23;
constexpr uint8_t SD_CS = 13, ECG_CS = 26;
constexpr uint8_t SW_RST = 0x08, SYNCH = 0x09, FIFO_RST = 0x0A;
constexpr uint8_t INFO = 0x0F, CNFG_GEN = 0x10, CNFG_CAL = 0x12;
constexpr uint8_t CNFG_EMUX = 0x14, CNFG_ECG = 0x15, ECG_FIFO = 0x21;
constexpr uint16_t BLOCK_SAMPLES = 50;
constexpr uint32_t FS = 250;
const SPISettings ECG_SETTINGS(1000000, MSBFIRST, SPI_MODE0);
WiFiClient client;
uint8_t packet[8 + BLOCK_SAMPLES * 4];
uint16_t buffered = 0;
uint32_t sequence = 0, totalSamples = 0;
uint32_t firstSampleMs = 0, lastReportMs = 0;

void writeReg(uint8_t reg, uint32_t value) {
  digitalWrite(SD_CS, HIGH);
  SPI.beginTransaction(ECG_SETTINGS);
  digitalWrite(ECG_CS, LOW);
  SPI.transfer(reg << 1);
  SPI.transfer((value >> 16) & 0xff);
  SPI.transfer((value >> 8) & 0xff);
  SPI.transfer(value & 0xff);
  digitalWrite(ECG_CS, HIGH);
  SPI.endTransaction();
}

uint32_t readReg(uint8_t reg) {
  digitalWrite(SD_CS, HIGH);
  SPI.beginTransaction(ECG_SETTINGS);
  digitalWrite(ECG_CS, LOW);
  SPI.transfer((reg << 1) | 1);
  uint32_t value = uint32_t(SPI.transfer(0)) << 16;
  value |= uint32_t(SPI.transfer(0)) << 8;
  value |= SPI.transfer(0);
  digitalWrite(ECG_CS, HIGH);
  SPI.endTransaction();
  return value;
}

void put16(uint8_t *dst, uint16_t x) {
  dst[0] = uint8_t(x); dst[1] = uint8_t(x >> 8);
}
void put32(uint8_t *dst, uint32_t x) {
  for (int i = 0; i < 4; ++i) dst[i] = uint8_t(x >> (8 * i));
}

void fail(const char *message) {
  Serial.print("FAIL: "); Serial.println(message);
  client.stop();
  delay(1000);
  ESP.restart(); // never continue a partial ten-minute window
}

void sendPacket() {
  if (!buffered) return;
  put32(packet, sequence);
  put16(packet + 4, buffered);
  put16(packet + 6, 0);
  const size_t length = 8 + size_t(buffered) * 4;
  size_t sent = 0;
  const uint32_t start = millis();
  while (sent < length && client.connected()) {
    size_t n = client.write(packet + sent, length - sent);
    sent += n;
    if (millis() - start > 1000) fail("Wi-Fi stream stalled");
    if (!n) delay(1);
  }
  if (sent != length) fail("Wi-Fi disconnected");
  ++sequence;
  buffered = 0;
}

void addSample(int32_t value) {
  if (!totalSamples) firstSampleMs = millis();
  put32(packet + 8 + size_t(buffered) * 4, uint32_t(value));
  ++buffered;
  ++totalSamples;
  if (buffered == BLOCK_SAMPLES) sendPacket();
}

void startECG() {
  writeReg(SW_RST, 0);
  delay(100);
  (void)readReg(INFO);
  uint32_t info = readReg(INFO);
  Serial.printf("MAX30003 INFO: 0x%06lX\n", (unsigned long)info);
  if ((info & 0xF00000UL) != 0x500000UL) fail("MAX30003 INFO invalid");
  writeReg(CNFG_CAL, 0);       // internal calibration OFF
  writeReg(CNFG_EMUX, 0);      // electrodes connected to ECG inputs
  writeReg(CNFG_ECG, 0x405000); // 250 sps, gain 20, 0.5/40 Hz
  writeReg(CNFG_GEN, 0x100000);
  writeReg(SYNCH, 0);
  writeReg(FIFO_RST, 0);
  writeReg(CNFG_GEN, 0x180017);
  delay(1200);                 // discard analog settling
  writeReg(FIFO_RST, 0);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("AFIB live MAX30003 -> Pi Wi-Fi");
  pinMode(SD_CS, OUTPUT); pinMode(ECG_CS, OUTPUT);
  digitalWrite(SD_CS, HIGH); digitalWrite(ECG_CS, HIGH);
  SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  const uint32_t wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - wifiStart > 20000) fail("Wi-Fi connect timeout");
    delay(250);
  }
  Serial.print("ESP32 IP: "); Serial.println(WiFi.localIP());
  if (!client.connect(PI_IP, PI_PORT, 5000)) fail("Pi TCP connect failed");
  client.setNoDelay(true);
  if (client.print("LIVE 250 20\n") != 12) fail("Pi greeting failed");
  startECG();
  lastReportMs = millis();
  Serial.println("Streaming electrode ECG; first forecast after ten minutes.");
}

void loop() {
  if (!client.connected() || WiFi.status() != WL_CONNECTED) fail("Connection lost");
  for (unsigned i = 0; i < 32; ++i) {
    uint32_t word = readReg(ECG_FIFO);
    uint8_t tag = (word >> 3) & 7;
    if (tag == 0 || tag == 2) {
      int32_t value = int32_t((word >> 6) & 0x3ffff);
      if (value & 0x20000) value -= 0x40000;
      addSample(value);
      if (tag == 2) break;
    } else if (tag == 6) {
      break; // FIFO empty
    } else if (tag == 7) {
      fail("MAX30003 FIFO overflow");
    } else {
      fail("Unexpected MAX30003 FIFO tag");
    }
  }
  if (millis() - lastReportMs >= 5000) {
    Serial.printf("Live samples: %lu | elapsed: %lu s\n",
                  (unsigned long)totalSamples,
                  (unsigned long)((millis() - firstSampleMs) / 1000));
    lastReportMs = millis();
  }
  delay(1);
}
