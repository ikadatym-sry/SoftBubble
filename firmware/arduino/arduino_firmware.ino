/*
 * ==============================================================================
 * Bubble 3D Tactile Sensor - High-Speed WS2812B LED Controller
 * ==============================================================================
 * Hardware:
 *   - Arduino Uno R3 connected via USB (/dev/ttyACM0)
 *   - 4x WS2812B RGB NeoPixels on Pin D6
 *
 * Protocol:
 *   - "B<val>\n" : Sets brightness (0 to 100%). Example: "B25\n"
 *   - "?\n"      : Queries current brightness
 *
 * High-speed serial execution (115200 baud), zero I2C blocking.
 * ==============================================================================
 */

#include <Adafruit_NeoPixel.h>

#define LED_PIN 6
#define NUM_LEDS 4

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

int current_brightness = 25;

void applyBrightness(int b) {
  current_brightness = constrain(b, 0, 100);
  // Linear to perceptual brightness map
  int val = map(current_brightness, 0, 100, 0, 255);
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(val, val, val));
  }
  strip.show();
}

void setup() {
  Serial.begin(115200);
  strip.begin();
  applyBrightness(current_brightness);

  Serial.println("[ARDUINO_LED] Ready. WS2812B on D6. Brightness: 25");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.peek();
    if (c == 'B' || c == 'b') {
      Serial.read(); // consume 'B'
      int val = Serial.parseInt();
      applyBrightness(val);
      Serial.print("[LED] Brightness: ");
      Serial.println(current_brightness);

      // Discard trailing characters up to newline
      while (Serial.available() > 0 && Serial.peek() != 'B' && Serial.peek() != 'b') {
        char ch = Serial.read();
        if (ch == '\n' || ch == '\r') break;
      }
    } else if (c == '?') {
      Serial.read();
      Serial.print("[LED] Status: Brightness=");
      Serial.println(current_brightness);
    } else {
      Serial.read(); // discard unrecognized byte
    }
  }
}
