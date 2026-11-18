"""
Tornado Asynchronous Web & WebSocket Server for Bubble 3D Tactile Streaming.
Runs with Optical Flow Dot Tracker (63 dots), Solid Shaded 3D Dome Engine,
Real-time Engineering Telemetry, and Z-Stage Stepper Rig Controls.
"""

import os
import sys
import json
import base64
import time
import threading
import fcntl
import csv
from datetime import datetime
from pathlib import Path
from typing import Set, Tuple

import cv2
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bubble_engine.camera import BubbleCamera
from test_optical_flow_live import OpticalFlowDotEngine
from bubble_engine.dome_3d import Tactile3DDomeEngine
from bubble_engine.shadow_engine import ContactShadowEngine
from bubble_engine.stepper import BubbleStepper
from bubble_engine.loadcell import BubbleLoadCell
from bubble_engine.mprls import MPRLSDriver, mpr_sensor, HAS_MPRLS
from bubble_engine.led_controller import BubbleLEDController

def read_server_pressure():
    return mpr_sensor.get_pressure()


class TactileServerState:
    def __init__(self):
        self.camera = BubbleCamera(width=640, height=480, lens_position=12.0, fps=30)
        # Responsive threshold 2.8px (immune to subpixel sensor noise and ambient micro-vibrations)
        self.tracker = OpticalFlowDotEngine(min_distance_px=38.0, displacement_thresh_px=2.8)
        self.shadow_engine = ContactShadowEngine(roi_center=(320, 240), roi_radius_px=145, mm_per_px=0.171, shadow_thresh_delta=10.0)
        self.dome_engine = Tactile3DDomeEngine(
            dome_radius_mm=45.0,
            dome_height_mm=24.0,
            indent_scale_mm_per_px=0.65,
            rbf_sigma_mm=5.0,
            disp_thresh_px=2.8,
        )
        self.stepper = BubbleStepper()
        self.loadcell = BubbleLoadCell()

        # Native Hardware SPI LED Controller (WS2812B NeoPixel on GPIO 10 / Pin 19)
        self.led_ctrl = BubbleLEDController(num_leds=4, initial_brightness=25)
        self.current_brightness = self.led_ctrl.current_brightness
        self.auto_brightness_enabled = True
        self.blanking_counter = 0

        # Optional Serial Controller fallback (if Arduino is still plugged in)
        self.serial_ports = []
        self._init_serial_ports()

        self.ws_clients: Set[tornado.websocket.WebSocketHandler] = set()
        self.running = False
        self._calibrate_requested = True  # Auto-calibrate at startup
        self.p_baseline = 1013.25

        self.latest_payload = None
        self._lock = threading.Lock()
        
        # TestVersion1 Data Recording Engine (results/ folder)
        self.results_dir = Path(__file__).resolve().parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.record_start_time = 0.0
        self.record_file = None
        self.record_writer = None
        self.record_filename = ""
        self.record_sample_count = 0
        self._record_lock = threading.Lock()

        # Physical force calibration parameters (α, β, γ)
        self.alpha_pneumatic = 1.0
        self.beta_elastic = 0.0
        self.gamma_offset = 0.0
        self._load_force_calibration()

    def start_recording(self, shape_name: str = "", trial_idx: int = 0) -> str:
        with self._record_lock:
            if self.is_recording:
                return self.record_filename
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            if shape_name:
                shape_clean = shape_name.strip().lower()
                save_dir = self.results_dir / shape_clean
                save_dir.mkdir(parents=True, exist_ok=True)
                if trial_idx > 0:
                    self.record_filename = f"{shape_clean}_trial_{trial_idx}_{now_str}.csv"
                else:
                    self.record_filename = f"{shape_clean}_{now_str}.csv"
                filepath = save_dir / self.record_filename
            else:
                self.record_filename = f"rec_{now_str}.csv"
                filepath = self.results_dir / self.record_filename

            self.record_file = open(filepath, "w", newline="", encoding="utf-8")
            self.record_writer = csv.writer(self.record_file)
            # Exactly 11 columns in EXACT requested order:
            self.record_writer.writerow([
                "Timestamp",
                "Pressed?",
                "Pressure_Raw",
                "Pressure_hPa",
                "LoadCell_Raw",
                "LoadCell_N",
                "Dimension_px",
                "Area_px2",
                "Dimension_mm",
                "Area_mm2",
                "Displacement_px"
            ])
            self.record_file.flush()
            self.record_start_time = time.time()
            self.record_sample_count = 0
            self.is_recording = True
            print(f"[REC] Started recording to: {filepath}")
            return self.record_filename

    def stop_recording(self) -> Tuple[str, int]:
        with self._record_lock:
            if not self.is_recording:
                return "", 0
            fname = self.record_filename
            count = self.record_sample_count
            self.is_recording = False
            if self.record_file is not None:
                try:
                    self.record_file.flush()
                    self.record_file.close()
                except Exception:
                    pass
                self.record_file = None
                self.record_writer = None
            print(f"[REC] Stopped recording: {fname} ({count} samples saved)")
            return fname, count

    def _load_force_calibration(self):
        calib_file = Path(__file__).resolve().parent / "force_calibration.json"
        if calib_file.exists():
            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                self.alpha_pneumatic = float(cal.get("alpha_pneumatic", 1.0))
                self.beta_elastic = float(cal.get("beta_elastic_n_per_mm", 0.0))
                self.gamma_offset = float(cal.get("gamma_offset_n", 0.0))
                print(f"[ForceCalibration] Loaded: α={self.alpha_pneumatic:.3f}, β={self.beta_elastic:.4f}, γ={self.gamma_offset:+.4f}")
            except Exception as e:
                print(f"[ForceCalibration] Note: {e}")

    def _init_serial_ports(self):
        try:
            import serial
            import glob
            candidate_ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
            for port in candidate_ports:
                try:
                    s = serial.Serial(port, 115200, timeout=0.05)
                    time.sleep(0.5)
                    s.write(f'B{self.current_brightness}\n'.encode())
                    self.serial_ports.append(s)
                    print(f"[SERIAL] Connected to MCU on {port} at 115200 baud.")
                except Exception as e:
                    print(f"[SERIAL] Note ({port}): {e}")
        except Exception as e:
            print(f"[SERIAL] Driver error: {e}")

    def set_brightness(self, val: int):
        val = max(0, min(100, int(val)))
        if val != self.current_brightness:
            self.current_brightness = val
            self.blanking_counter = 4
            self.led_ctrl.set_brightness(val)
            for s in list(self.serial_ports):
                try:
                    s.write(f'B{val}\n'.encode())
                except Exception:
                    pass

    def start_engine(self):
        self.camera.start()
        self.running = True
        if self.loadcell.is_connected:
            self.loadcell.tare(samples=8)
            self.loadcell.start_polling(hz=12)
        if mpr_sensor.is_connected:
            mpr_sensor.start_background_polling(hz=40)
        if self.serial_ports:
            threading.Thread(target=self._serial_telemetry_loop, daemon=True).start()
        thread = threading.Thread(target=self._processing_loop, daemon=True)
        thread.start()

    def _serial_telemetry_loop(self):
        print(f"[SERIAL] Listening on {len(self.serial_ports)} port(s) for pressure telemetry...")
        while self.running:
            for s in list(self.serial_ports):
                try:
                    line = s.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("P:"):
                        p_val = float(line[2:])
                        if 300.0 <= p_val <= 2000.0:
                            mpr_sensor.latest_pressure = p_val
                            mpr_sensor.is_connected = True
                except Exception:
                    pass
            time.sleep(0.01)

    def _processing_loop(self):
        print("[ServerState] Vision processing loop active with Optical Flow & 3D Dome.")
        calib_frames = []
        last_auto_light = time.time()
        last_mprls_retry = time.time()
        is_contact_prev = False

        while self.running:
            ok, frame_rgb, _frame_id = self.camera.read()
            if not ok or frame_rgb is None:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Auto-Brightness Feedback Loop (every 1.0s, only adjusts when membrane is at REST)
            if self.auto_brightness_enabled and not is_contact_prev and (time.time() - last_auto_light >= 1.0):
                last_auto_light = time.time()
                gray_lum = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                mean_lum = float(np.mean(gray_lum[self.tracker.roi_mask]))
                TARGET_LUM = 95.0
                if mean_lum < (TARGET_LUM - 12.0) and self.current_brightness < 60:
                    self.set_brightness(self.current_brightness + 2)
                elif mean_lum > (TARGET_LUM + 12.0) and self.current_brightness > 5:
                    self.set_brightness(self.current_brightness - 2)

            # Check if calibration requested
            if self._calibrate_requested:
                calib_frames.append(frame_bgr)
                if len(calib_frames) >= 5:
                    avg_base = np.mean(calib_frames, axis=0).astype(np.uint8)
                    num_dots = self.tracker.calibrate(avg_base)
                    if num_dots >= 10 and self.tracker.baseline_pts is not None:
                        self.dome_engine.set_baseline_dots(self.tracker.baseline_pts)
                        self.shadow_engine.calibrate(avg_base)
                        # Zero pressure baseline to the actual ambient room pressure:
                        p_now = read_server_pressure()
                        if p_now > 500.0:
                            self.p_baseline = p_now
                            mpr_sensor.zero_baseline()
                            print(f"[ServerState] Atmospheric pressure zeroed to: {self.p_baseline:.2f} hPa")
                        if self.loadcell.is_connected:
                            self.loadcell.tare(samples=10)
                        calib_frames.clear()
                        self._calibrate_requested = False
                        print(f"[ServerState] Calibrated {num_dots} optical flow dots and shadow baseline.")
                    else:
                        print(f"[ServerState] Calibration retry... ({num_dots} dots found)")
                        calib_frames.clear()
                        time.sleep(0.3)
                time.sleep(0.02)
                continue

            # Blanking Guard: Skip optical flow during LED brightness transition (prevents phantom motion)
            if self.blanking_counter > 0:
                self.blanking_counter -= 1
                if self.tracker.baseline_pts is not None:
                    self.tracker.current_pts = self.tracker.baseline_pts.copy()
                    raw_g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    self.tracker.baseline_gray = self.tracker.clahe.apply(raw_g)
                    self.tracker.prev_gray = self.tracker.baseline_gray.copy()
                time.sleep(0.02)
                continue

            # Auto-reconnect MPRLS if disconnected
            if not mpr_sensor.is_connected and (time.time() - last_mprls_retry >= 2.0):
                last_mprls_retry = time.time()
                if mpr_sensor.init_sensor():
                    mpr_sensor.start_background_polling(hz=40)
                    self.p_baseline = mpr_sensor.latest_pressure

            # 1. Real-time MPRLS Pressure Telemetry (Pressure-First Physical Gate)
            # In a closed pneumatic bubble, physical contact ALWAYS produces internal pressure rise (ΔP > 0).
            # If pressure does NOT rise, suppress all phantom optical flow / ambient lighting noise!
            p_curr = read_server_pressure()
            dp = max(0.0, p_curr - self.p_baseline) if (mpr_sensor.is_connected and p_curr > 0) else 0.0
            is_pressurized = (dp >= 0.38) if mpr_sensor.is_connected else True

            # Dynamic Tare Tracking & Anti-Deadlock Recovery:
            # - If baseline was set higher than current ambient pressure, rapidly pull down.
            # - When untouched, gently drift baseline with room air pressure and thermal changes.
            # - Anti-latchup guard: if contact was flagged but ΔP is marginal (< 0.50 hPa), release baseline smoothly!
            if mpr_sensor.is_connected and p_curr > 500.0:
                if p_curr < self.p_baseline:
                    self.p_baseline = 0.85 * self.p_baseline + 0.15 * p_curr
                elif not is_contact_prev:
                    self.p_baseline = 0.995 * self.p_baseline + 0.005 * p_curr
                elif dp < 0.50:
                    self.p_baseline = 0.990 * self.p_baseline + 0.010 * p_curr

            # 2. Track dots & optical flow displacements (Stream 1: Marker Motion)
            res = self.tracker.process_frame(frame_bgr)
            if res is None:
                time.sleep(0.01)
                continue

            # Strict Gating: If not pressurized, suppress all active dot triggers from optical noise
            if not is_pressurized:
                active_ids = []
            else:
                active_ids = res["active_ids"]
            displacements = res["displacements"]

            # 3. Deform 3D dome surface & calculate telemetry
            dome_res = self.dome_engine.deform(
                self.tracker.current_pts,
                displacements,
                active_ids
            )
            telemetry = dome_res["telemetry"]
            telemetry["pressure_hpa"] = round(p_curr, 2)
            telemetry["delta_p_hpa"] = round(dp, 2)

            # Robust Multi-Modal Contact Gate:
            # - Optical marker cluster confirmed (contact_active == True) AND chamber is pressurized (ΔP >= 0.38 hPa)
            # - OR high normal force on flat indenter apex (ΔP >= 0.70 hPa)
            # This completely eliminates phantom false-positive contact triggers!
            optical_contact = bool(telemetry.get("contact_active", False))
            strong_pressure = (dp >= 0.70)
            physical_contact = bool(is_pressurized and (optical_contact or strong_pressure))
            telemetry["contact_active"] = physical_contact
            is_contact_prev = physical_contact

            # Shear Force & Slip Kinematics
            shear_mag = 0.0
            shear_angle = 0.0
            shear_state = "REST"
            if physical_contact and len(active_ids) >= 3:
                active_dx = [displacements[i][0] for i in active_ids]
                active_dy = [displacements[i][1] for i in active_ids]
                net_dx = float(np.mean(active_dx))
                net_dy = float(np.mean(active_dy))
                shear_mag = float(np.hypot(net_dx, net_dy))
                shear_angle = float(np.degrees(np.arctan2(net_dy, net_dx)))
                shear_state = "[ SHEAR / SLIP ]" if shear_mag >= 1.8 else "[ PURE NORMAL ]"

            telemetry["shear_mag_px"] = round(shear_mag, 2)
            telemetry["shear_angle_deg"] = round(shear_angle, 1)
            telemetry["shear_state"] = shear_state
            telemetry["shear_desc"] = f"{shear_state} ({shear_mag:.1f}px)" if shear_mag >= 1.2 else "Pure Normal"

            # Calculate Contact Tilt Angle theta from spherical dome geometry (R = 45.0 mm)
            contact_r_mm = float(telemetry.get("center_r_mm", 0.0))
            dome_r = 45.0
            sin_theta = float(np.clip(contact_r_mm / dome_r, 0.0, 0.95))
            contact_tilt_rad = float(np.arcsin(sin_theta))
            contact_tilt_deg = float(np.degrees(contact_tilt_rad))
            cos_theta = float(np.cos(contact_tilt_rad))
            telemetry["contact_tilt_deg"] = round(contact_tilt_deg, 1)

            # 4. Contact Shadow & Shading Analysis (Stream 2: Contact Shadow & Shape)
            marker_center_px = None
            if physical_contact:
                cx_px = 320.0 + (telemetry["center_x_mm"] / 0.237)
                cy_px = 240.0 - (telemetry["center_y_mm"] / 0.237)
                marker_center_px = (cx_px, cy_px)

            shadow_res = self.shadow_engine.process_frame(
                frame_bgr,
                marker_center_px=marker_center_px,
                is_contact_from_markers=physical_contact,
                contact_tilt_deg=contact_tilt_deg,
                cos_theta=cos_theta,
            )

            # Fuse shadow telemetry into main telemetry
            telemetry["shadow_detected"] = shadow_res["shadow_detected"]
            telemetry["shape_name"] = shadow_res["shape_name"] if physical_contact else "None"
            telemetry["shadow_l_mm"] = shadow_res["length_mm"] if physical_contact else 0.0
            telemetry["shadow_w_mm"] = shadow_res["width_mm"] if physical_contact else 0.0
            telemetry["corrected_l_mm"] = shadow_res.get("corrected_l_mm", 0.0) if physical_contact else 0.0
            telemetry["corrected_w_mm"] = shadow_res.get("corrected_w_mm", 0.0) if physical_contact else 0.0
            telemetry["aspect_ratio"] = shadow_res.get("effective_aspect", shadow_res["aspect_ratio"]) if physical_contact else 0.0

            # 3.5 Normal Force Model Validation: F_sensor = ΔP × Area vs. F_loadcell
            # Units: ΔP (hPa) -> × 100 Pa/hPa. Area (mm²) -> × 1e-6 m²/mm².
            # F_sensor (N) = (ΔP_hPa × Area_mm²) / 10,000.0
            l_mm = telemetry.get("corrected_l_mm") or telemetry.get("footprint_l_mm") or 0.0
            w_mm = telemetry.get("corrected_w_mm") or telemetry.get("footprint_w_mm") or 0.0
            contact_area_mm2 = 0.0
            if physical_contact:
                # Direct contour area (no shape guessing needed)
                if shadow_res.get("area_mm2", 0.0) > 0.0:
                    contact_area_mm2 = float(shadow_res["area_mm2"])
                elif l_mm > 0.0 and w_mm > 0.0:
                    contact_area_mm2 = float((np.pi / 4.0) * l_mm * w_mm)

            f_raw_n = (dp * contact_area_mm2) / 10000.0 if (physical_contact and contact_area_mm2 > 0) else 0.0
            depth_mm = float(telemetry.get("max_depth_mm", 0.0))
            f_sensor_n = max(0.0, self.alpha_pneumatic * f_raw_n + self.beta_elastic * depth_mm + self.gamma_offset) if physical_contact else 0.0
            f_loadcell_n = self.loadcell.get_force_n() if self.loadcell.is_connected else 0.0

            # Strict 6.5 N Load Cell Safety Guard (Membrane Protection Interlock)
            # Advisor protocol targets up to 5.0 N. E-stop triggers if force reaches >= 6.5 N!
            if self.loadcell.is_connected and f_loadcell_n >= 6.5:
                telemetry["overforce_alert"] = True
                if self.stepper.is_moving:
                    print(f"\n🚨 [SAFETY OVERFORCE] Load cell force reached {f_loadcell_n:.2f} N >= 6.5 N! Emergency stopping stepper!")
                    self.stepper.stop()
            else:
                telemetry["overforce_alert"] = False

            telemetry["contact_area_mm2"] = round(contact_area_mm2, 1)
            telemetry["f_sensor_n"] = round(f_sensor_n, 3)
            telemetry["f_loadcell_n"] = round(f_loadcell_n, 3)
            telemetry["f_error_n"] = round(f_sensor_n - f_loadcell_n, 3)

            # TestVersion1: Real-time Data Recording Engine (results/ folder)
            # Exactly 10 columns strictly ordered:
            # 1. Timestamp (Starts at 0, Example: Time: 00:00)
            # 2. Pressed?: True/False
            # 3. Pressure_Raw (24-bit ADC counts)
            # 4. Pressure_hPa (calibrated pressure in hPa)
            # 5. LoadCell_Raw (raw ADC reading from HX711)
            # 6. LoadCell_N (calibrated force in Newton)
            # 7. Dimension_px (N1 x N2 of detected object in px)
            # 8. Area_px2 (Area in px²)
            # 9. Dimension_mm (N1 x N2 of detected object in mm)
            # 10. Area_mm2 (Area in mm²)
            if self.is_recording and self.record_writer is not None:
                elapsed_sec = time.time() - self.record_start_time
                mins = int(elapsed_sec // 60)
                secs = elapsed_sec % 60
                timestamp_str = f"Time: {mins:02d}:{secs:05.2f}"
                is_pressed = bool(physical_contact)
                pressure_raw = int(mpr_sensor.get_raw_count())
                pressure_hpa = round(float(p_curr), 2)
                loadcell_raw = int(self.loadcell.latest_raw)
                loadcell_n = round(float(f_loadcell_n), 4)

                if physical_contact and shadow_res.get("shadow_detected", False):
                    l_px = float(shadow_res.get("length_px", 0.0))
                    w_px = float(shadow_res.get("width_px", 0.0))
                    dim_px_str = f"{l_px:.1f} x {w_px:.1f}"
                    area_px2_val = round(float(shadow_res.get("area_px2", 0.0)), 1)
                    l_mm_val = float(shadow_res.get("length_mm", 0.0))
                    w_mm_val = float(shadow_res.get("width_mm", 0.0))
                    dim_mm_str = f"{l_mm_val:.2f} x {w_mm_val:.2f}"
                    area_mm2_val = round(float(contact_area_mm2), 2)
                else:
                    dim_px_str = "0.0 x 0.0"
                    area_px2_val = 0.0
                    dim_mm_str = "0.00 x 0.00"
                    area_mm2_val = 0.00

                # Column 11: Real-time Marker Displacement magnitude (pixels)
                primary_ids = telemetry.get("primary_contact_ids", [])
                if physical_contact and primary_ids and displacements:
                    disp_px_val = round(float(max([displacements[i][2] for i in primary_ids if i < len(displacements)])), 2)
                elif physical_contact and displacements:
                    disp_px_val = round(float(max([d[2] for d in displacements])), 2)
                else:
                    disp_px_val = 0.00

                try:
                    self.record_writer.writerow([
                        timestamp_str,
                        is_pressed,
                        pressure_raw,
                        pressure_hpa,
                        loadcell_raw,
                        loadcell_n,
                        dim_px_str,
                        area_px2_val,
                        dim_mm_str,
                        area_mm2_val,
                        disp_px_val
                    ])
                    self.record_file.flush()
                    self.record_sample_count += 1
                except Exception as rec_err:
                    print(f"[REC] Write error: {rec_err}")

            elapsed_now = (time.time() - self.record_start_time) if self.is_recording else 0.0
            mins_now = int(elapsed_now // 60)
            secs_now = elapsed_now % 60
            telemetry["recording"] = {
                "is_recording": self.is_recording,
                "time_str": f"{mins_now:02d}:{secs_now:04.1f}" if self.is_recording else "00:00.0",
                "filename": self.record_filename,
                "samples": self.record_sample_count,
            }

            # 4. Render 3D Perspective Dome View
            vis_3d = self.dome_engine.render_3d_view(
                dome_res["deformed_z"],
                dome_res["dots_3d_current"],
                active_ids,
                displacements,
                dome_res["depression_field"],
                telemetry=telemetry,
                view_w=540,
                view_h=480,
            )

            # 5. 2D camera feed with dots & optical flow vectors + contact shadow overlay
            vis_2d = self.shadow_engine.render_overlay(res["annotated_image"], shadow_res)

            # Prominent Shear Vector Arrow Overlay (Cyan/Gold)
            if len(active_ids) >= 3 and shear_mag >= 1.2:
                arrow_len = int(np.clip(shear_mag * 7.5, 12, 120))
                rad = np.radians(shear_angle)
                cx_contact = int(np.mean([self.tracker.current_pts[i][0] for i in active_ids]))
                cy_contact = int(np.mean([self.tracker.current_pts[i][1] for i in active_ids]))
                end_x = int(cx_contact + arrow_len * np.cos(rad))
                end_y = int(cy_contact + arrow_len * np.sin(rad))
                cv2.circle(vis_2d, (cx_contact, cy_contact), 7, (255, 255, 0), 2)
                cv2.circle(vis_2d, (cx_contact, cy_contact), 3, (255, 255, 0), -1)
                cv2.arrowedLine(vis_2d, (cx_contact, cy_contact), (end_x, end_y), (255, 255, 0), 3, tipLength=0.35)
                cv2.putText(vis_2d, f"SHEAR: {shear_mag:.1f}px {shear_state}", (cx_contact + 10, cy_contact - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # 5. Combined side-by-side (640 + 540 = 1180 x 480)
            combined = np.hstack([vis_2d, vis_3d])

            # 6. JPEG Encodings
            _ret_comb, jpeg_comb = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 68])
            b64_comb = base64.b64encode(jpeg_comb).decode("ascii")

            _ret_c, jpeg_cam = cv2.imencode(".jpg", vis_2d, [cv2.IMWRITE_JPEG_QUALITY, 68])
            b64_cam = base64.b64encode(jpeg_cam).decode("ascii")

            _ret_dome, jpeg_dome = cv2.imencode(".jpg", vis_3d, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64_dome = base64.b64encode(jpeg_dome).decode("ascii")

            with self._lock:
                self.latest_jpeg_comb = jpeg_comb.tobytes()
                self.latest_jpeg_cam = jpeg_cam.tobytes()
                self.latest_jpeg_dome = jpeg_dome.tobytes()
                self.latest_raw_bgr = frame_bgr.copy()

            # 7. Build per-dot list for 3D marker visualization
            dots_3d = []
            if self.tracker.baseline_pts is not None and self.dome_engine.baseline_3d is not None:
                scale_mm = 45.0 / 235.0
                for idx in range(min(len(self.tracker.baseline_pts), self.dome_engine.num_dots)):
                    bx = float(self.dome_engine.baseline_3d[idx, 0])
                    by = float(self.dome_engine.baseline_3d[idx, 1])
                    bz = float(dome_res["dots_3d_current"][idx, 2])
                    dx, dy, disp = displacements[idx]
                    dots_3d.append({
                        "id": idx,
                        "x": round(bx, 2),
                        "y": round(by, 2),
                        "z": round(bz, 2),
                        "dx": round(float(dx * scale_mm), 2),
                        "dy": round(float(dy * scale_mm), 2),
                        "disp_px": round(float(disp), 2),
                        "active": idx in active_ids,
                    })

            # 8. Pack payload
            payload = {
                "combined_base64": b64_comb,
                "cam_base64": b64_cam,
                "dome_base64": b64_dome,
                "dots": dots_3d,
                "active_ids": active_ids,
                "telemetry": telemetry,
                "light": {
                    "brightness": self.current_brightness,
                    "auto": self.auto_brightness_enabled,
                },
                "stepper": {
                    "position_mm": round(self.stepper.position_mm, 3),
                    "is_moving": self.stepper.is_moving,
                    "is_recovering": getattr(self.stepper, "_is_recovering", False),
                    "top_limit": self.stepper.is_top_triggered(),
                    "bottom_limit": self.stepper.is_bottom_triggered(),
                },
                "stiffness_thresh": self.tracker.displacement_thresh_px,
                "debug": {
                    "num_dots": len(self.tracker.baseline_pts) if self.tracker.baseline_pts is not None else 0,
                    "contact": telemetry["contact_active"],
                    "max_disp": round(float(res["max_disp_px"]), 2),
                    "max_depth_mm": round(float(telemetry["max_depth_mm"]), 2),
                },
            }

            with self._lock:
                self.latest_payload = payload

            time.sleep(0.015)  # Cap loop rate to ~30-40 FPS

    def request_calibrate(self):
        self._calibrate_requested = True
        print("[ServerState] Calibration requested.")


state = TactileServerState()


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.set_header("Pragma", "no-cache")
        self.set_header("Expires", "0")
        index_file = SRC_DIR / "web" / "index.html"
        self.write(index_file.read_text(encoding="utf-8"))


class MJPEGHandler(tornado.web.RequestHandler):
    async def get(self):
        self.set_header("Cache-Control", "no-cache, private")
        self.set_header("Pragma", "no-cache")
        self.set_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        try:
            while True:
                with state._lock:
                    if state.latest_payload and "dome_base64" in state.latest_payload:
                        raw_bytes = base64.b64decode(state.latest_payload["dome_base64"])
                    else:
                        raw_bytes = None
                if raw_bytes:
                    self.write(b"--FRAME\r\n")
                    self.write(b"Content-Type: image/jpeg\r\n")
                    self.write(f"Content-Length: {len(raw_bytes)}\r\n\r\n".encode())
                    self.write(raw_bytes)
                    self.write(b"\r\n")
                    await self.flush()
                await tornado.gen.sleep(0.033)
        except Exception:
            pass


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        self._is_busy = False
        state.ws_clients.add(self)
        print(f"[WebSocket] Client connected (Total: {len(state.ws_clients)})")

    def on_message(self, message):
        try:
            data = json.loads(message)
            cmd = data.get("cmd")
            if cmd == "calibrate":
                state.request_calibrate()
            elif cmd == "set_brightness":
                val = int(data.get("val", 25))
                state.auto_brightness_enabled = False
                state.set_brightness(val)
            elif cmd == "set_auto_light":
                state.auto_brightness_enabled = bool(data.get("val", True))
            elif cmd == "stepper_jog":
                direction = str(data.get("direction", "down")).lower()
                dist_mm = float(data.get("mm", 1.0))
                speed = float(data.get("speed", 4.0))
                state.stepper.jog(direction, dist_mm, speed)
            elif cmd == "stepper_home":
                speed = float(data.get("speed", 3.0))
                state.stepper.go_home(speed)
            elif cmd == "stepper_stop":
                state.stepper.stop()
            elif cmd == "stepper_zero":
                state.stepper.reset_position()
            elif cmd in ("tare", "loadcell_tare"):
                if state.loadcell.is_connected:
                    state.loadcell.tare(samples=10)
            elif cmd == "tune_stiffness":
                delta = float(data.get("delta", 0.0))
                new_th = round(float(np.clip(state.tracker.displacement_thresh_px + delta, 1.0, 8.0)), 1)
                state.tracker.displacement_thresh_px = new_th
                state.dome_engine.disp_thresh_px = new_th
                print(f"[ServerState] Stiffness threshold updated to {new_th} px")
            elif cmd == "orbit_3d":
                dx = float(data.get("dx", 0.0))
                dy = float(data.get("dy", 0.0))
                state.dome_engine.azimuth_deg = (state.dome_engine.azimuth_deg - dx * 0.7) % 360.0
                state.dome_engine.elevation_deg = float(np.clip(state.dome_engine.elevation_deg + dy * 0.5, 5.0, 88.0))
                state.dome_engine._update_rotation_matrix()
            elif cmd == "reset_view":
                mode = str(data.get("mode", "iso"))
                state.dome_engine.reset_view(mode)
            elif cmd == "toggle_dot_circles":
                state.tracker.show_circles = bool(data.get("enabled", True))
                print(f"[ServerState] Marker dot circles visibility: {state.tracker.show_circles}")
            elif cmd == "toggle_record":
                if state.is_recording:
                    state.stop_recording()
                else:
                    shape = str(data.get("shape", ""))
                    trial = int(data.get("trial", 0))
                    state.start_recording(shape_name=shape, trial_idx=trial)
            elif cmd == "start_record":
                shape = str(data.get("shape", ""))
                trial = int(data.get("trial", 0))
                state.start_recording(shape_name=shape, trial_idx=trial)
            elif cmd == "stop_record":
                state.stop_recording()
            elif cmd == "capture_snapshot":
                custom_raw_path = str(data.get("raw_path", "")).strip()
                with state._lock:
                    comb = getattr(state, "latest_jpeg_comb", None)
                    cam = getattr(state, "latest_jpeg_cam", None)
                    dome = getattr(state, "latest_jpeg_dome", None)
                    raw_bgr = getattr(state, "latest_raw_bgr", None)
                    telem = (state.latest_payload.get("telemetry", {}) if state.latest_payload else {})
                    flc = telem.get("f_loadcell_n", 0.0)
                    dp = telem.get("delta_p_hpa", 0.0)

                snap_dir = state.results_dir / "snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")

                if raw_bgr is not None:
                    raw_file = snap_dir / f"snapshot_raw_{ts}.jpg"
                    cv2.imwrite(str(raw_file), raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if custom_raw_path:
                        cv2.imwrite(custom_raw_path, raw_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        print(f"📸 [Snapshot] Saved custom RAW to {custom_raw_path}")
                if comb is not None:
                    with open(snap_dir / f"snapshot_combined_{ts}.jpg", "wb") as f:
                        f.write(comb)
                if cam is not None:
                    with open(snap_dir / f"snapshot_cam2d_{ts}.jpg", "wb") as f:
                        f.write(cam)
                if dome is not None:
                    with open(snap_dir / f"snapshot_dome3d_{ts}.jpg", "wb") as f:
                        f.write(dome)

                print(f"📸 [Snapshot] Saved snapshots to {snap_dir} (ts: {ts}, F_lc: {flc:.2f}N, dP: {dp:.2f}hPa)")
        except Exception as e:
            print(f"[WebSocket] Error parsing message: {e}")

    def on_close(self):
        state.ws_clients.discard(self)
        print(f"[WebSocket] Client disconnected (Remaining: {len(state.ws_clients)})")


async def _safe_send_frame(client, data_str):
    try:
        if client.ws_connection and not client.ws_connection.is_closing():
            await client.write_message(data_str)
    except Exception:
        state.ws_clients.discard(client)
    finally:
        client._is_busy = False


def broadcast_loop():
    if not state.ws_clients or state.latest_payload is None:
        return

    with state._lock:
        payload = state.latest_payload

    data_str = json.dumps(payload)
    loop = tornado.ioloop.IOLoop.current()

    for client in list(state.ws_clients):
        if not client.ws_connection or client.ws_connection.is_closing():
            state.ws_clients.discard(client)
            continue
        if getattr(client, "_is_busy", False):
            continue
        client._is_busy = True
        loop.add_callback(_safe_send_frame, client, data_str)


def make_app():
    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/stream.mjpg", MJPEGHandler),
            (r"/ws", WebSocketHandler),
        ],
        websocket_ping_interval=10,
        websocket_ping_timeout=30,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    state.start_engine()

    app = make_app()
    app.listen(port, address="0.0.0.0")
    print(f"=====================================================")
    print(f"  BUBBLE 3D TACTILE ENGINE ACTIVE")
    print(f"  Web 3D Console: http://192.168.4.1:{port}")
    print(f"=====================================================")

    # Schedule WebSocket broadcast every 33ms (~30 FPS)
    tornado.ioloop.PeriodicCallback(broadcast_loop, 33).start()
    tornado.ioloop.IOLoop.current().start()
