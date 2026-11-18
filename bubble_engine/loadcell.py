"""
Load Cell Driver (HX711) for Raspberry Pi 5.
Reads stage normal force (F_loadcell) in Newtons (N) via HX711 and lgpio.
Uses calibrated counts_per_gram from loadcell_calibration.json.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

GRAVITY_MPS2 = 9.80665  # Standard gravity (m/s^2) for N = kg * g

# Hardware GPIO backends
try:
    import lgpio
except Exception as exc:
    lgpio = None
    _LGPIO_ERROR = exc
else:
    _LGPIO_ERROR = None

try:
    import RPi.GPIO as GPIO
except Exception as exc:
    GPIO = None
    _GPIO_ERROR = exc
else:
    _GPIO_ERROR = None


class _LGPIOBackend:
    name = "lgpio"

    def __init__(self, dout_pin: int, pd_sck_pin: int):
        if lgpio is None:
            raise RuntimeError(f"lgpio module required on Pi 5: {_LGPIO_ERROR}")
        self.dout_pin = int(dout_pin)
        self.pd_sck_pin = int(pd_sck_pin)
        chip_text = os.environ.get("PI_BUBBLE_GPIO_CHIP", "gpiochip0")
        chip_number = int(chip_text.replace("/dev/", "").replace("gpiochip", ""))
        self._handle = lgpio.gpiochip_open(chip_number)
        self._closed = False

        try:
            lgpio.gpio_claim_output(self._handle, self.pd_sck_pin, 0)
        except TypeError:
            lgpio.gpio_claim_output(self._handle, 0, self.pd_sck_pin, 0)
        try:
            lgpio.gpio_claim_input(self._handle, self.dout_pin)
        except TypeError:
            lgpio.gpio_claim_input(self._handle, 0, self.dout_pin)

    def output(self, pin: int, level: int):
        lgpio.gpio_write(self._handle, int(pin), 1 if level else 0)

    def input(self, pin: int) -> int:
        return int(lgpio.gpio_read(self._handle, int(pin)))

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            lgpio.gpiochip_close(self._handle)
        except Exception:
            pass


class _RPiGPIOBackend:
    name = "RPi.GPIO"

    def __init__(self, dout_pin: int, pd_sck_pin: int):
        if GPIO is None:
            raise RuntimeError(f"RPi.GPIO module required: {_GPIO_ERROR}")
        self.dout_pin = int(dout_pin)
        self.pd_sck_pin = int(pd_sck_pin)
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pd_sck_pin, GPIO.OUT, initial=GPIO.LOW)
        try:
            GPIO.setup(self.dout_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        except (AttributeError, TypeError):
            GPIO.setup(self.dout_pin, GPIO.IN)

    def output(self, pin: int, level: int):
        self.GPIO.output(int(pin), self.GPIO.HIGH if level else self.GPIO.LOW)

    def input(self, pin: int) -> int:
        return int(self.GPIO.input(int(pin)))

    def close(self):
        pass


class BubbleLoadCell:
    """
    Thread-safe Load Cell manager for real-time F_loadcell telemetry.
    Reads mass in grams and converts directly to Newtons (N).
    """

    DEFAULT_CALIB_PATHS = [
        Path("/home/bubble/Desktop/Bubble/loadcell_calibration.json"),
        Path(__file__).resolve().parent.parent / "loadcell_calibration.json",
        Path("./loadcell_calibration.json"),
    ]

    def __init__(
        self,
        dt_pin: Optional[int] = None,
        sck_pin: Optional[int] = None,
        counts_per_gram: Optional[float] = None,
    ):
        self.dt_pin = dt_pin
        self.sck_pin = sck_pin
        self.counts_per_gram = counts_per_gram
        self.offset = 0.0
        self.is_connected = False
        self.backend = None

        self._load_calibration()

        # Fallback defaults if calibration file wasn't found
        if self.dt_pin is None:
            self.dt_pin = 5
        if self.sck_pin is None:
            self.sck_pin = 6
        if self.counts_per_gram is None or abs(self.counts_per_gram) < 1e-6:
            self.counts_per_gram = -127.1415

        self.latest_raw = 0
        self.latest_weight_g = 0.0
        self.latest_force_n = 0.0
        self.last_update_time = 0.0
        self._raw_window = deque(maxlen=5)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._init_hardware()

    def _load_calibration(self):
        for p in self.DEFAULT_CALIB_PATHS:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if self.counts_per_gram is None:
                        self.counts_per_gram = float(data.get("counts_per_gram", -127.1415))
                    if self.dt_pin is None:
                        self.dt_pin = int(data.get("dt_pin", 5))
                    if self.sck_pin is None:
                        self.sck_pin = int(data.get("sck_pin", 6))
                    print(f"[LoadCell] Loaded calibration from {p}: CPG={self.counts_per_gram:.2f}, DT={self.dt_pin}, SCK={self.sck_pin}")
                    return
                except Exception as e:
                    print(f"[LoadCell] Note reading {p}: {e}")

    def _init_hardware(self) -> bool:
        backends = [_LGPIOBackend, _RPiGPIOBackend]
        for backend_cls in backends:
            try:
                self.backend = backend_cls(self.dt_pin, self.sck_pin)
                self.is_connected = True
                print(f"[LoadCell] Initialized hardware backend: {self.backend.name} on DT={self.dt_pin}, SCK={self.sck_pin}")
                self.reset()
                return True
            except Exception as e:
                pass

        print(f"[LoadCell] Notice: No physical GPIO backend available. Running in simulated/offline mode.")
        self.is_connected = False
        return False

    def reset(self):
        if not self.is_connected or not self.backend:
            return
        self.backend.output(self.sck_pin, 0)
        time.sleep(0.001)
        self.power_down()
        self.power_up()

    def power_down(self):
        if self.backend:
            self.backend.output(self.sck_pin, 0)
            self.backend.output(self.sck_pin, 1)
            time.sleep(0.0001)

    def power_up(self):
        if self.backend:
            self.backend.output(self.sck_pin, 0)
            time.sleep(0.001)

    def _read_raw(self, timeout: float = 1.0) -> Optional[int]:
        if not self.is_connected or not self.backend:
            return None

        # Wait until DOUT goes LOW (ready)
        deadline = time.monotonic() + timeout
        while self.backend.input(self.dt_pin) != 0:
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.001)

        value = 0
        # Read 24 bits MSB first
        for _ in range(24):
            self.backend.output(self.sck_pin, 1)
            value = (value << 1) | int(self.backend.input(self.dt_pin))
            self.backend.output(self.sck_pin, 0)

        # 25th pulse for Channel A, Gain 128
        self.backend.output(self.sck_pin, 1)
        self.backend.output(self.sck_pin, 0)

        # 24-bit 2's complement conversion
        if value & 0x800000:
            value -= 0x1000000
        return value

    def tare(self, samples: int = 12, timeout: float = 1.5) -> float:
        """Zeroes the load cell baseline with robust median of multiple samples."""
        if not self.is_connected:
            return 0.0

        print(f"[LoadCell] Taring stage load cell ({samples} samples)...")
        readings = []
        for _ in range(samples * 2):
            raw = self._read_raw(timeout=timeout)
            if raw is not None and -8000000 < raw < 8000000:
                readings.append(raw)
                if len(readings) >= samples:
                    break
            time.sleep(0.04)

        if len(readings) >= 3:
            self.offset = float(np.median(readings))
            self._raw_window.clear()
            for _ in range(5):
                self._raw_window.append(self.offset)
            print(f"[LoadCell] Tare complete. Offset = {self.offset:.1f}")
        else:
            print(f"[LoadCell] Warning: Tare failed to capture sufficient readings.")
        return self.offset

    def read_sync(self) -> Tuple[float, float]:
        """Reads mass (grams) and force (Newtons) synchronously with median filtering."""
        raw = self._read_raw(timeout=0.35)
        if raw is not None and -8000000 < raw < 8000000:
            self._raw_window.append(raw)
            clean_raw = float(np.median(self._raw_window))

            weight_g = (clean_raw - self.offset) / self.counts_per_gram
            # Prevent tiny noise drift around zero
            if abs(weight_g) < 0.35:
                weight_g = 0.0
            force_n = (weight_g / 1000.0) * GRAVITY_MPS2

            with self._lock:
                self.latest_raw = int(clean_raw)
                self.latest_weight_g = weight_g
                self.latest_force_n = force_n
                self.last_update_time = time.time()
            return weight_g, force_n

        with self._lock:
            return self.latest_weight_g, self.latest_force_n

    def start_polling(self, hz: float = 12.0):
        if self._running or not self.is_connected:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, args=(hz,), daemon=True)
        self._thread.start()
        print(f"[LoadCell] Background telemetry thread started at ~{hz} Hz.")

    def _worker(self, hz: float):
        interval = 1.0 / max(1.0, hz)
        while self._running:
            try:
                self.read_sync()
            except Exception:
                if not self._running:
                    break
            time.sleep(interval)

    def get_force_n(self) -> float:
        with self._lock:
            return self.latest_force_n

    def get_weight_grams(self) -> float:
        with self._lock:
            return self.latest_weight_g

    def get_telemetry(self) -> Dict:
        with self._lock:
            return {
                "force_loadcell_n": round(self.latest_force_n, 4),
                "weight_grams": round(self.latest_weight_g, 2),
                "raw": self.latest_raw,
                "is_connected": self.is_connected,
                "timestamp": self.last_update_time,
            }

    def close(self):
        self._running = False
        if self.backend:
            try:
                self.backend.close()
            except Exception:
                pass
        self.is_connected = False
