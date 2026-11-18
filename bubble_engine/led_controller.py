"""
Hardware SPI Controller for WS2812B NeoPixel LEDs on Raspberry Pi 5.
Directly drives LEDs on GPIO 10 (Physical Pin 19 - SPI0 MOSI) without an external Arduino.
"""

import threading
import time
from typing import List, Tuple

try:
    import spidev
    HAS_SPI = True
except ImportError:
    spidev = None
    HAS_SPI = False


class BubbleLEDController:
    def __init__(
        self,
        num_leds: int = 4,
        bus: int = 0,
        device: int = 0,
        speed_hz: int = 6400000,
        initial_brightness: int = 25,
    ):
        self.num_leds = num_leds
        self.bus = bus
        self.device = device
        self.speed_hz = speed_hz
        self.current_brightness = max(0, min(100, int(initial_brightness)))
        self._lock = threading.Lock()
        self.spi = None

        # Precompute 8-bit WS2812B to SPI byte expansion table (6.4 MHz)
        # Bit 0 -> 0xC0 (0b11000000) ~312ns high, 938ns low
        # Bit 1 -> 0xF8 (0b11111000) ~781ns high, 469ns low
        self._lookup = []
        for b in range(256):
            pat = []
            for bit in range(7, -1, -1):
                pat.append(0xF8 if ((b >> bit) & 1) else 0xC0)
            self._lookup.append(pat)

        self._init_spi()
        self.set_brightness(self.current_brightness)

    def _init_spi(self) -> bool:
        if not HAS_SPI:
            print("[BubbleLED] spidev module not available.")
            return False
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(self.bus, self.device)
            self.spi.max_speed_hz = self.speed_hz
            self.spi.mode = 0
            print(f"[BubbleLED] Connected on SPI{self.bus}.{self.device} (GPIO 10 / Pin 19) @ {self.speed_hz/1e6:.1f} MHz.")
            return True
        except Exception as e:
            print(f"[BubbleLED] SPI open error: {e}")
            self.spi = None
            return False

    def set_brightness(self, percent: int):
        with self._lock:
            self.current_brightness = max(0, min(100, int(percent)))
            # Map 0-100% to 0-255 PWM level
            pwm_val = int(round(self.current_brightness * 255.0 / 100.0))
            self._send_raw([(pwm_val, pwm_val, pwm_val)] * self.num_leds)

    def set_color(self, r: int, g: int, b: int):
        with self._lock:
            self._send_raw([(r, g, b)] * self.num_leds)

    def _send_raw(self, rgb_list: List[Tuple[int, int, int]]):
        if self.spi is None:
            return
        # WS2812B expects GRB order
        tx = []
        for r, g, b in rgb_list:
            r_c = max(0, min(255, int(r)))
            g_c = max(0, min(255, int(g)))
            b_c = max(0, min(255, int(b)))
            tx.extend(self._lookup[g_c])
            tx.extend(self._lookup[r_c])
            tx.extend(self._lookup[b_c])
        # Reset latch: > 80us low
        tx.extend([0x00] * 80)
        try:
            self.spi.writebytes2(tx)
        except Exception as e:
            print(f"[BubbleLED] Write error: {e}")

    def close(self):
        with self._lock:
            if self.spi is not None:
                try:
                    self._send_raw([(0, 0, 0)] * self.num_leds)
                    self.spi.close()
                except Exception:
                    pass
                self.spi = None
