"""
Z-Axis Stepper Motor Controller for Raspberry Pi 5.
Controls the NEMA stepper via the native libgpiod runner.
PUL: GPIO 24 | DIR: GPIO 23 | ENA: GPIO 26

Safety Limit Switches (User Confirmed):
  - Limit 1 = TOP Limit    : GPIO 17 (Physical Pin 11)
  - Limit 2 = BOTTOM Limit : GPIO 27 (Physical Pin 13)

Safety Behavior:
  If ANY limit switch is triggered:
  1. Instantly aborts/stops all motion (< 15ms).
  2. Automatically returns the stage back to Starting Point (0.00 mm / Home).
  3. Re-enables safe ready state.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

STEPPER_RUNNER = "/home/bubble/Desktop/Bubble/native/build/stepper_runner"
GPIO_CHIP = os.environ.get("PI_BUBBLE_GPIO_CHIP", "gpiochip0")

PUL_PIN = 24
DIR_PIN = 23
ENA_PIN = 26

# Safety Limit Switches:
LIMIT_TOP_PIN = 17      # Limit 1: Upper limit (Physical Pin 11)
LIMIT_BOTTOM_PIN = 27   # Limit 2: Lower limit (Physical Pin 13)

PULSES_PER_REV = 1600.0
MM_PER_REV = 8.0  # Lead 8mm
MM_PER_PULSE = MM_PER_REV / PULSES_PER_REV  # 0.005 mm/pulse
FREQUENCY_HZ = 1600.0  # 1 rev/sec = 8 mm/sec


class BubbleStepper:
    def __init__(self):
        self.runner_bin = STEPPER_RUNNER
        self.available = Path(self.runner_bin).exists()
        self.current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._recovery_lock = threading.Lock()

        self.position_mm = 0.0
        self.start_position_mm = 0.0
        self.is_moving = False
        self._is_recovering = False
        self._thread: Optional[threading.Thread] = None

        self._gpio_handle = None
        self._has_limits = False
        self._init_limits()

    def _init_limits(self):
        try:
            import lgpio
            chip_num = int(GPIO_CHIP.replace("gpiochip", "").replace("/dev/", ""))
            self._gpio_handle = lgpio.gpiochip_open(chip_num)
            try:
                lgpio.gpio_claim_input(self._gpio_handle, LIMIT_TOP_PIN, lgpio.SET_PULL_UP)
                lgpio.gpio_claim_input(self._gpio_handle, LIMIT_BOTTOM_PIN, lgpio.SET_PULL_UP)
                self._has_limits = True
                print(f"[Stepper] Limit switches initialized: Limit 1 (TOP)=GPIO{LIMIT_TOP_PIN}, Limit 2 (BOTTOM)=GPIO{LIMIT_BOTTOM_PIN}")
            except Exception as e:
                print(f"[Stepper] Note claiming limit pins: {e}")
                self._has_limits = False
        except Exception as e:
            print(f"[Stepper] Note opening gpiochip: {e}")
            self._gpio_handle = None
            self._has_limits = False

    def is_top_triggered(self) -> bool:
        """Limit 1: Top limit switch (returns True if pressed/grounded)."""
        if self._has_limits and self._gpio_handle is not None:
            try:
                import lgpio
                return lgpio.gpio_read(self._gpio_handle, LIMIT_TOP_PIN) == 0
            except Exception:
                pass
        return False

    def is_bottom_triggered(self) -> bool:
        """Limit 2: Bottom limit switch (returns True if pressed/grounded)."""
        if self._has_limits and self._gpio_handle is not None:
            try:
                import lgpio
                return lgpio.gpio_read(self._gpio_handle, LIMIT_BOTTOM_PIN) == 0
            except Exception:
                pass
        return False

    def get_limits_state(self) -> Dict[str, bool]:
        return {
            "top_limit1_triggered": self.is_top_triggered(),
            "bottom_limit2_triggered": self.is_bottom_triggered(),
        }

    def jog(self, direction: str, distance_mm: float = 1.0, speed_mm_s: float = 4.0, blocking: bool = False) -> bool:
        """
        Jog Z-stage by distance_mm in 'up' or 'down' direction.
        Can run non-blocking (default) or blocking with hard safety limit interlock.
        """
        dir_clean = str(direction).lower()
        if dir_clean not in ("up", "down"):
            return False

        if self._is_recovering:
            print("⚠️ [Stepper] Safety recovery in progress. Please wait...")
            return False

        # Pre-movement boundary interlock
        if dir_clean == "up" and self.is_top_triggered():
            print("⚠️ [Stepper Safety] Move UP blocked: Limit 1 (TOP) is already TRIGGERED!")
            self._trigger_recovery("TOP")
            return False
        if dir_clean == "down" and self.is_bottom_triggered():
            print("⚠️ [Stepper Safety] Move DOWN blocked: Limit 2 (BOTTOM) is already TRIGGERED!")
            self._trigger_recovery("BOTTOM")
            return False

        if not self.available:
            print(f"[Stepper] Runner {self.runner_bin} not found. Running in simulation.")
            self._update_sim(direction, distance_mm)
            return True

        with self._lock:
            if self.is_moving:
                print("[Stepper] Motor is currently busy moving.")
                return False
            self.is_moving = True

        self._thread = threading.Thread(
            target=self._run_move,
            args=(direction, distance_mm, speed_mm_s),
            daemon=True,
        )
        self._thread.start()
        if blocking:
            self._thread.join()
        return True

    def _run_move(self, direction: str, distance_mm: float, speed_mm_s: float):
        try:
            dir_state = 1 if direction.lower() == "down" else 0
            pulses = int(round(distance_mm / MM_PER_PULSE))
            frequency = min(3200.0, max(200.0, speed_mm_s / MM_PER_PULSE))
            duration_s = pulses / frequency

            cmd = [
                self.runner_bin,
                "--chip", GPIO_CHIP,
                "--pul", str(PUL_PIN),
                "--dir", str(DIR_PIN),
                "--ena", str(ENA_PIN),
                "--dir-state", str(dir_state),
                "--ena-active", "0",
                "--ena-inactive", "0",
                "--frequency", f"{frequency:.1f}",
                "--seconds", f"{duration_s:.6f}",
            ]

            with self._lock:
                self.current_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

            # Active Limit Switch Watchdog during physical travel
            stop_watchdog = threading.Event()
            limit_hit_name = None

            def _watchdog_worker():
                nonlocal limit_hit_name
                while not stop_watchdog.is_set():
                    top_hit = self.is_top_triggered()
                    bottom_hit = self.is_bottom_triggered()

                    if top_hit or bottom_hit:
                        limit_hit_name = "TOP" if top_hit else "BOTTOM"
                        print(f"\n🚨 [LIMIT TRIPPED] Limit {limit_hit_name} HIT while moving {direction.upper()}! Aborting all motions immediately!")
                        self.stop()
                        break
                    time.sleep(0.012)

            watchdog_thread = threading.Thread(target=_watchdog_worker, daemon=True)
            watchdog_thread.start()

            _stdout, stderr = self.current_process.communicate()
            stop_watchdog.set()

            if limit_hit_name:
                # Limit was hit: Trigger automatic return to starting point
                self._trigger_recovery(limit_hit_name)
                return

            if self.current_process and self.current_process.returncode == 0:
                delta = distance_mm if direction.lower() == "down" else -distance_mm
                self.position_mm += delta
                print(f"[Stepper] Moved {direction} {distance_mm:.2f} mm. Position: {self.position_mm:.2f} mm")
            else:
                if stderr:
                    print(f"[Stepper] Move stderr: {stderr}")

        except Exception as e:
            print(f"[Stepper] Execution exception: {e}")
        finally:
            with self._lock:
                self.is_moving = False
                self.current_process = None

    def _trigger_recovery(self, hit_limit: str):
        """Starts background recovery thread to return to starting point."""
        threading.Thread(target=self._auto_recover_to_start, args=(hit_limit,), daemon=True).start()

    def _auto_recover_to_start(self, hit_limit: str):
        """
        Safety Interlock Routine:
        1. Immediately stops everything.
        2. Automatically returns stage back to Starting Point (0.00 mm / Home).
        3. Clears limit and re-enables ready state.
        """
        with self._recovery_lock:
            if self._is_recovering:
                return
            self._is_recovering = True

        try:
            print(f"\n🛑 [SAFETY INTERLOCK ACTIVE] Limit {hit_limit} triggered!")
            print("   -> Step 1: Stopping all active actions.")
            self.stop()
            time.sleep(0.35)

            with self._lock:
                current_z = self.position_mm
                target_home = self.start_position_mm

            print(f"   -> Step 2: Auto-reversing back to starting position ({target_home:.2f} mm)...")

            if current_z > target_home:
                # Carriage is below starting point (e.g. searching/pressing bubble)
                # Any emergency stop (TOP button pressed by user or BOTTOM hit) MUST retract UP
                dist_up = current_z - target_home
                print(f"   -> Returning UP by {dist_up:.2f} mm to starting point...")
                self._run_raw_move("up", dist_up, speed_mm_s=2.5)
            elif current_z < target_home:
                dist_down = target_home - current_z
                print(f"   -> Moving DOWN by {dist_down:.2f} mm to starting point...")
                self._run_raw_move("down", dist_down, speed_mm_s=2.5)
            else:
                # Already at home
                if hit_limit == "TOP":
                    print("   -> At home; backing off DOWN 1.5 mm to clear top limit switch...")
                    self._run_raw_move("down", 1.5, speed_mm_s=2.0)

            with self._lock:
                self.position_mm = self.start_position_mm

            time.sleep(0.2)
            print("✅ [SAFETY INTERLOCK COMPLETE] Returned safely to Starting Point (0.00 mm). Ready.\n")

        finally:
            with self._recovery_lock:
                self._is_recovering = False

    def _run_raw_move(self, direction: str, distance_mm: float, speed_mm_s: float = 5.0):
        """Low-level stepper move for recovery without recursion."""
        if not self.available:
            self._update_sim(direction, distance_mm)
            return

        dir_state = 1 if direction.lower() == "down" else 0
        pulses = int(round(distance_mm / MM_PER_PULSE))
        frequency = min(3200.0, max(200.0, speed_mm_s / MM_PER_PULSE))
        duration_s = pulses / frequency

        cmd = [
            self.runner_bin,
            "--chip", GPIO_CHIP,
            "--pul", str(PUL_PIN),
            "--dir", str(DIR_PIN),
            "--ena", str(ENA_PIN),
            "--dir-state", str(dir_state),
            "--ena-active", "0",
            "--ena-inactive", "0",
            "--frequency", f"{frequency:.1f}",
            "--seconds", f"{duration_s:.6f}",
        ]

        try:
            with self._lock:
                self.current_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                self.is_moving = True

            self.current_process.communicate()
        except Exception as e:
            print(f"[Stepper-Raw] Move exception: {e}")
        finally:
            with self._lock:
                self.is_moving = False
                self.current_process = None

    def _update_sim(self, direction: str, distance_mm: float):
        delta = distance_mm if direction.lower() == "down" else -distance_mm
        self.position_mm += delta
        print(f"[Stepper-SIM] Position: {self.position_mm:.2f} mm")

    def stop(self):
        with self._lock:
            if self.current_process is not None:
                try:
                    self.current_process.terminate()
                    self.current_process.kill()
                except Exception:
                    pass
                self.current_process = None
            self.is_moving = False
        print("[Stepper] Emergency stop executed.")

    def reset_position(self):
        with self._lock:
            self.position_mm = 0.0
            self.start_position_mm = 0.0
        print("[Stepper] Position zeroed to 0.00 mm (Starting Point).")

    def set_start_position(self, val_mm: float = 0.0):
        with self._lock:
            self.start_position_mm = float(val_mm)
        print(f"[Stepper] Starting point set to: {self.start_position_mm:.2f} mm")

    def wait_idle(self, timeout_s: float = 30.0) -> bool:
        """Blocks until any active movement or recovery is finished."""
        start_t = time.time()
        while self.is_moving or self._is_recovering:
            if time.time() - start_t > timeout_s:
                return False
            time.sleep(0.05)
        return True

    def go_home(self, speed_mm_s: float = 2.5, blocking: bool = False) -> bool:
        """Returns stepper back to Starting Point (0.00 mm)."""
        with self._lock:
            if self.is_moving or self._is_recovering:
                print("[Stepper] Motor is currently busy moving.")
                return False
            target_disp = self.position_mm - self.start_position_mm

        if abs(target_disp) < 0.01:
            print(f"[Stepper] Already at starting point ({self.start_position_mm:.2f} mm).")
            return True

        direction = "up" if target_disp > 0 else "down"
        dist = abs(target_disp)
        print(f"[Stepper] Returning home to starting point (Moving {direction} {dist:.2f} mm)...")
        return self.jog(direction, dist, speed_mm_s=speed_mm_s, blocking=blocking)
