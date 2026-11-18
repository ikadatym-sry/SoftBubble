/*
 * ==============================================================================
 * Bubble 3D Tactile Sensor - ESP8266 MPRLS Pressure Reader
 * ==============================================================================
 * Pinout:
 *   MPRLS Pin 1 (VIN) -> ESP8266 3V3 (or 5V/VIN)
 *   MPRLS Pin 2 (3Vo) -> (Leave Empty / Float)
 *   MPRLS Pin 3 (GND) -> ESP8266 GND (G)
 *   MPRLS Pin 4 (SCL) -> ESP8266 Pin D1 (GPIO 5)
 *   MPRLS Pin 5 (SDA) -> ESP8266 Pin D2 (GPIO 4)
 *   MPRLS Pin 6 (EOC) -> (Leave Empty / Float)
 *   MPRLS Pin 7 (RST) -> (Leave Empty / Float)
 *
 * Serial Protocol: 115200 baud
 *   Streams: "P:<pressure_hpa>\n" at ~30 Hz
 */

#include <Wire.h>

#define SDA_PIN D2
#define SCL_PIN D1
#define MPRLS_ADDR 0x18

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(100);

  Serial.println("\n--- [ESP8266] MPRLS Pressure Sensor Scanner & Reader ---");

  // I2C Scan verification
  Wire.beginTransmission(MPRLS_ADDR);
  byte err = Wire.endTransmission();
  if (err == 0) {
    Serial.println("  ✅ Found MPRLS sensor at address 0x18!");
  } else {
    Serial.print("  ⚠️ Could not reach 0x18 (Wire error = ");
    Serial.print(err);
    Serial.println("). Check D1(SCL) and D2(SDA) wiring!");
  }
}

float readPressureHPa() {
  Wire.beginTransmission(MPRLS_ADDR);
  Wire.write(0xAA);
  Wire.write(0x00);
  Wire.write(0x00);
  if (Wire.endTransmission() != 0) {
    return -1.0;
  }

  delay(12); // MPRLS conversion delay

  if (Wire.requestFrom((uint8_t)MPRLS_ADDR, (uint8_t)4) >= 4) {
    uint8_t stat = Wire.read();
    uint32_t b1 = Wire.read();
    uint32_t b2 = Wire.read();
    uint32_t b3 = Wire.read();

    if ((stat & 0x20) == 0 && (stat & 0x04) == 0) {
      uint32_t raw = (b1 << 16) | (b2 << 8) | b3;
      float psi = (float)(raw - 0x19999A) * 25.0 / (float)(0xE66666 - 0x19999A);
      float hpa = psi * 68.947572932;
      return hpa;
    }
  }
  return -1.0;
}

void loop() {
  float p = readPressureHPa();
  if (p > 300.0 && p < 2000.0) {
    Serial.print("P:");
    Serial.println(p, 2);
  }
  delay(33); // ~30 Hz telemetry rate
}
