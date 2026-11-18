"""
MPRLS Pressure Sensor Native Linux I2C Driver for Raspberry Pi 5.
Directly accesses Linux /dev/i2c-1 via smbus2 or ioctl.
Runs non-blocking background polling thread so vision loop never stutters.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from typing import Optional


class MPRLSDriver:
    I2C_SLAVE = 0x0703
    I2C_ADDR = 0x18

    def __init__(self, bus_num: int = 1):
        self.bus_num = bus_num
        self.dev_path = f"/dev/i2c-{bus_num}"
        self.fd: Optional[int] = None
        self.smbus_obj = None
        self.mode = None
        self.ada_sensor = None
        self.is_connected = False
        self.latest_pressure = 0.0
        self.latest_raw_count = 0
        self.filtered_pressure = 1013.25
        self.ema_alpha = 0.25  # Smooths out 24-bit MEMS ADC jitter while maintaining fast response (~80ms)
        self.p_baseline = 1013.25
        self._pending_spike: Optional[float] = None
        self._err_count = 0
        self._busy_streak = 0
        self._last_reconnect_attempt = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.init_sensor()

    def init_sensor(self) -> bool:
        # Safely clean up any existing handles
        if self.smbus_obj is not None:
            try:
                self.smbus_obj.close()
            except Exception:
                pass
            self.smbus_obj = None

        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None

        # 1. Try smbus2 with i2c_rdwr (Best on Raspberry Pi 5 RP1 chip)
        try:
            import smbus2
            bus = smbus2.SMBus(self.bus_num)
            w_msg = smbus2.i2c_msg.write(self.I2C_ADDR, [0xAA, 0x00, 0x00])
            bus.i2c_rdwr(w_msg)
            time.sleep(0.006)
            for _ in range(12):
                r_st = smbus2.i2c_msg.read(self.I2C_ADDR, 1)
                bus.i2c_rdwr(r_st)
                if not (list(r_st)[0] & 0x20):
                    break
                time.sleep(0.001)
            r_msg = smbus2.i2c_msg.read(self.I2C_ADDR, 4)
            bus.i2c_rdwr(r_msg)
            raw = list(r_msg)
            if len(raw) == 4 and not (raw[0] & 0x25):
                p = self._calc_hpa(raw)
                if 300.0 <= p <= 2000.0:
                    self.smbus_obj = bus
                    self.mode = "smbus2"
                    self.latest_pressure = p
                    self.latest_raw_count = (raw[1] << 16) | (raw[2] << 8) | raw[3]
                    self.p_baseline = p
                    self._err_count = 0
                    self._busy_streak = 0
                    self.is_connected = True
                    print(f"[MPRLS] Connected via smbus2 (i2c_rdwr)! Pressure: {p:.2f} hPa")
                    return True
        except Exception:
            self.smbus_obj = None

        # 2. Try native Linux /dev/i2c-1
        try:
            if os.path.exists(self.dev_path):
                if self.fd is not None:
                    try:
                        os.close(self.fd)
                    except Exception:
                        pass
                    self.fd = None

                fd = os.open(self.dev_path, os.O_RDWR)
                fcntl.ioctl(fd, self.I2C_SLAVE, self.I2C_ADDR)

                # Send test measurement command: 0xAA, 0x00, 0x00
                os.write(fd, bytes([0xAA, 0x00, 0x00]))
                time.sleep(0.02)

                raw = os.read(fd, 4)
                if len(raw) == 4:
                    status = raw[0]
                    if not (status & 0x04):
                        p = self._calc_hpa(raw)
                        if 300.0 <= p <= 2000.0:
                            self.fd = fd
                            self.mode = "native"
                            self.latest_pressure = p
                            self.p_baseline = p
                            self.is_connected = True
                            print(f"[MPRLS] Connected via Native Linux I2C (/dev/i2c-{self.bus_num})! Ambient: {p:.2f} hPa")
                            return True
                os.close(fd)
        except Exception:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
                self.fd = None

        # 3. Fallback: Adafruit Blinka
        try:
            import board
            import busio
            import adafruit_mprls
            i2c = board.I2C()
            ada = adafruit_mprls.MPRLS(i2c, psi_min=0, psi_max=25)
            p = float(ada.pressure)
            if 300.0 <= p <= 2000.0:
                self.ada_sensor = ada
                self.mode = "adafruit"
                self.latest_pressure = p
                self.p_baseline = p
                self.is_connected = True
                print(f"[MPRLS] Connected via Adafruit Blinka! Ambient: {p:.2f} hPa")
                return True
        except Exception:
            pass

        self.is_connected = False
        return False

    @staticmethod
    def _calc_hpa(raw_bytes) -> float:
        raw_count = (raw_bytes[1] << 16) | (raw_bytes[2] << 8) | raw_bytes[3]
        psi = (raw_count - 0x19999A) * 25.0 / (0xE66666 - 0x19999A)
        return float(psi * 68.947572932)

    def _update_filtered_pressure(self, raw_p: float) -> float:
        with self._lock:
            if self.filtered_pressure <= 300.0 or self.filtered_pressure > 2000.0 or self.filtered_pressure == 1013.25:
                self.filtered_pressure = raw_p
                self.latest_pressure = round(self.filtered_pressure, 2)
                self._pending_spike = None
                return self.latest_pressure

            # 2-Sample Outlier Rejection (Jump Guard):
            # In a pneumatic bubble, real pressure cannot jump > 3.0 hPa in 25ms unless it's an electrical bit-flip.
            # An isolated 1-frame glitch is rejected. If sustained for 2+ consecutive frames, it's accepted as real rapid touch.
            delta = abs(raw_p - self.filtered_pressure)
            if delta > 3.0:
                if self._pending_spike is None or abs(raw_p - self._pending_spike) > 2.0:
                    # First anomalous frame: store pending and hold current filtered pressure
                    self._pending_spike = raw_p
                    return self.latest_pressure
                else:
                    # Second consecutive confirming frame: it's a real sharp pressure jump, accept it!
                    self._pending_spike = None
            else:
                self._pending_spike = None

            self.filtered_pressure = self.ema_alpha * raw_p + (1.0 - self.ema_alpha) * self.filtered_pressure
            self.latest_pressure = round(self.filtered_pressure, 2)
            return self.latest_pressure

    def read_pressure_sync(self) -> float:
        if not self.is_connected:
            # Auto-reconnection: Retry init every 1.5 seconds so hardware reconnect recovers seamlessly
            now = time.time()
            if now - self._last_reconnect_attempt >= 1.5:
                self._last_reconnect_attempt = now
                if self.init_sensor():
                    print(f"[MPRLS] Auto-recovered! Pressure: {self.latest_pressure:.2f} hPa")
            return self.latest_pressure

        if self.mode == "smbus2" and self.smbus_obj is not None:
            import smbus2
            try:
                # 1. Trigger measurement command
                w_msg = smbus2.i2c_msg.write(self.I2C_ADDR, [0xAA, 0x00, 0x00])
                self.smbus_obj.i2c_rdwr(w_msg)
                time.sleep(0.005)

                # 2. Active Polling: Check status byte until Busy (Bit 5 / 0x20) is cleared
                ready = False
                for _ in range(12):
                    r_st = smbus2.i2c_msg.read(self.I2C_ADDR, 1)
                    self.smbus_obj.i2c_rdwr(r_st)
                    st = list(r_st)[0]
                    if not (st & 0x20):
                        ready = True
                        self._busy_streak = 0
                        break
                    time.sleep(0.001)

                if ready:
                    # 3. Read 4 bytes [status, b1, b2, b3]
                    r_msg = smbus2.i2c_msg.read(self.I2C_ADDR, 4)
                    self.smbus_obj.i2c_rdwr(r_msg)
                    raw = list(r_msg)
                    # Verify status byte: 0x20 (Busy), 0x04 (Integrity), 0x01 (Saturation) must all be 0
                    if len(raw) == 4 and not (raw[0] & 0x25):
                        with self._lock:
                            self.latest_raw_count = (raw[1] << 16) | (raw[2] << 8) | raw[3]
                        p = self._calc_hpa(raw)
                        if 300.0 <= p <= 2000.0:
                            self._err_count = 0
                            return self._update_filtered_pressure(p)
                else:
                    self._busy_streak += 1
                    if self._busy_streak > 25:
                        # Sensor is stuck in busy latch-up; mark disconnected to trigger auto-reinit
                        self.is_connected = False
            except Exception:
                self._err_count += 1
                if self._err_count > 10:
                    self.is_connected = False
                    if self.smbus_obj is not None:
                        try:
                            self.smbus_obj.close()
                        except Exception:
                            pass
                        self.smbus_obj = None
            return self.latest_pressure

        if self.mode == "adafruit" and self.ada_sensor is not None:
            try:
                p = float(self.ada_sensor.pressure)
                if 300.0 <= p <= 2000.0:
                    self._err_count = 0
                    return self._update_filtered_pressure(p)
            except Exception:
                self._err_count += 1
                if self._err_count > 15:
                    self.is_connected = False
            return self.latest_pressure

        if self.fd is None:
            return self.latest_pressure

        try:
            os.write(self.fd, bytes([0xAA, 0x00, 0x00]))
            time.sleep(0.005)
            for _ in range(10):
                st_byte = os.read(self.fd, 1)
                if st_byte and not (st_byte[0] & 0x20):
                    break
                time.sleep(0.001)
            raw = os.read(self.fd, 4)
            if len(raw) == 4 and not (raw[0] & 0x25):
                with self._lock:
                    self.latest_raw_count = (raw[1] << 16) | (raw[2] << 8) | raw[3]
                p = self._calc_hpa(raw)
                if 300.0 <= p <= 2000.0:
                    self._err_count = 0
                    return self._update_filtered_pressure(p)
        except Exception:
            self._err_count += 1
            if self._err_count > 15:
                self.is_connected = False
                try:
                    if self.fd is not None:
                        os.close(self.fd)
                except Exception:
                    pass
                self.fd = None
        return self.latest_pressure

    def start_background_polling(self, hz: float = 40.0):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, args=(1.0 / hz,), daemon=True)
        self._thread.start()

    def _poll_loop(self, interval: float):
        while self._running:
            self.read_pressure_sync()
            time.sleep(interval)

    def get_pressure(self) -> float:
        with self._lock:
            return self.latest_pressure

    def get_raw_count(self) -> int:
        with self._lock:
            return self.latest_raw_count

    def zero_baseline(self) -> float:
        with self._lock:
            if self.latest_pressure > 500.0:
                self.p_baseline = self.latest_pressure
                print(f"[MPRLS] Baseline pressure zeroed to: {self.p_baseline:.2f} hPa")
                return self.p_baseline
        return self.p_baseline


# Shared instance
mpr_sensor = MPRLSDriver(bus_num=1)
HAS_MPRLS = mpr_sensor.is_connected
