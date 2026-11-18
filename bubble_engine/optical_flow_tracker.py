"""
Optical Flow Dot Tracker with Persistent IDs
Captures camera frames, detects all dots on the membrane, assigns unique IDs,
and tracks displacement vectors using Lucas-Kanade Optical Flow (cv2.calcOpticalFlowPyrLK).
"""

import time
import cv2
import numpy as np
from pathlib import Path

# Camera import
try:
    from picamera2 import Picamera2
    from libcamera import controls as libcamera_controls
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False


class OpticalFlowDotEngine:
    def __init__(
        self,
        img_w: int = 640,
        img_h: int = 480,
        roi_center=(320, 240),
        roi_radius=239,
        min_distance_px: float = 38.0,
        displacement_thresh_px: float = 3.0,
    ):
        self.W = img_w
        self.H = img_h
        self.roi_center = roi_center
        self.roi_radius = roi_radius
        self.min_distance_px = min_distance_px
        self.displacement_thresh_px = displacement_thresh_px

        # Precompute ROI mask
        yy, xx = np.ogrid[: self.H, : self.W]
        dist2 = (xx - roi_center[0]) ** 2 + (yy - roi_center[1]) ** 2
        self.roi_mask = dist2 <= (roi_radius**2)

        # CLAHE for contrast equalization across center and outer boundary
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Lucas-Kanade Optical Flow Parameters
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Persistent Dot State
        self.baseline_gray = None
        self.baseline_pts = None  # (N, 2) float32
        self.dot_ids = []         # [0, 1, 2, ..., N-1]
        self.prev_gray = None
        self.current_pts = None   # (N, 2) float32
        self.show_circles = True  # Trypophobia toggle: True = encircle black dots, False = simple dots

    def detect_all_dots(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        High-precision dot detector capturing all printed black dots across center and boundary.
        Uses adaptive circularity for foreshortened edge dots and illumination-adaptive gating.
        Rejects translucent silicone gap artifacts.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        blur = cv2.medianBlur(enhanced, 5)

        th = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
        )
        th[~self.roi_mask] = 0
        cleaned = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.morph_kernel)

        cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_dots = []

        for c in cnts:
            area = cv2.contourArea(c)
            if 35 <= area <= 3500:
                peri = cv2.arcLength(c, True)
                circ = (4.0 * np.pi * area / (peri * peri)) if peri > 0 else 0
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])

                    # Software Defect Filter: Ignore fabrication artifact at bottom-left (formerly Dot 38 at x~135, y~310)
                    if (115 <= cx <= 160) and (290 <= cy <= 335):
                        continue
                    r_center = np.hypot(cx - self.roi_center[0], cy - self.roi_center[1])

                    # Outer edge dots appear elliptical due to steep curvature
                    is_edge = (r_center >= 160.0)
                    min_c = 0.10 if is_edge else 0.18
                    if circ < min_c:
                        continue

                    ix, iy = int(round(cx)), int(round(cy))

                    core = gray[max(0, iy - 2) : min(self.H, iy + 3), max(0, ix - 2) : min(self.W, ix + 3)]
                    core_mean = float(np.mean(core)) if core.size > 0 else 255.0

                    y_min, y_max = max(0, iy - 16), min(self.H, iy + 17)
                    x_min, x_max = max(0, ix - 16), min(self.W, ix + 17)
                    sub_y, sub_x = np.ogrid[y_min:y_max, x_min:x_max]
                    d2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2
                    ring_mask = (d2 >= 10**2) & (d2 <= 16**2)
                    ring_pixels = gray[y_min:y_max, x_min:x_max][ring_mask]
                    bg_mean = float(np.mean(ring_pixels)) if len(ring_pixels) > 0 else core_mean
                    contrast = bg_mean - core_mean
                    ratio = core_mean / bg_mean if bg_mean > 0 else 1.0

                    # Relative contrast gate: Real printed black dots are distinctly darker than local silicone
                    # Ambient-invariant: ratio <= 0.85 & contrast >= 12.0 works reliably across both bright daytime and dark environments
                    if contrast >= 12.0 and ratio <= 0.85:
                        raw_dots.append((cx, cy, area))

        # Non-Maximum Suppression (NMS): spacing between dots on physical grid is ~38 px
        clean_pts = []
        for x, y, a in sorted(raw_dots, key=lambda d: d[2], reverse=True):
            if not any(np.hypot(x - cx, y - cy) < 32.0 for cx, cy in clean_pts):
                clean_pts.append((x, y))

        # Sort dots row-by-row (top-to-bottom, left-to-right) for logical, stable numbering
        clean_pts.sort(key=lambda p: (round(p[1] / 30.0), p[0]))

        return np.array(clean_pts, dtype=np.float32)

    def calibrate(self, bgr_frame: np.ndarray) -> int:
        """
        Calibrates baseline: detects all dots and assigns permanent IDs.
        """
        pts = self.detect_all_dots(bgr_frame)
        if len(pts) < 10:
            print(f"[ERROR] Only {len(pts)} dots detected. Calibration failed.")
            return 0

        self.baseline_pts = pts.copy()
        self.current_pts = pts.copy()
        self.dot_ids = list(range(len(pts)))
        raw_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        self.baseline_gray = self.clahe.apply(raw_gray)
        self.prev_gray = self.baseline_gray.copy()

        print(f"[SUCCESS] Calibrated {len(self.dot_ids)} dots with persistent IDs (0 to {len(self.dot_ids)-1}).")
        return len(self.dot_ids)

    def process_frame(self, bgr_frame: np.ndarray):
        """
        Tracks all dot IDs using Lucas-Kanade Optical Flow.
        Returns:
            tracked_data: dict containing per-dot displacement, active IDs, and annotated image.
        """
        if self.baseline_pts is None:
            return None

        raw_gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = self.clahe.apply(raw_gray)
        N = len(self.dot_ids)

        # Reshape for calcOpticalFlowPyrLK: (N, 1, 2)
        p0 = self.current_pts.reshape(-1, 1, 2)

        # Track from prev_gray to cur_gray
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, cur_gray, p0, None, **self.lk_params
        )

        # Also track from baseline_gray to cur_gray to eliminate cumulative drift
        p_base, status_base, _ = cv2.calcOpticalFlowPyrLK(
            self.baseline_gray, cur_gray, self.baseline_pts.reshape(-1, 1, 2), None, **self.lk_params
        )

        p1 = p1.reshape(-1, 2)
        p_base = p_base.reshape(-1, 2)
        status = status.flatten()
        status_base = status_base.flatten()

        vis = bgr_frame.copy()
        active_ids = []
        displacements = []
        max_disp = 0.0

        for i in range(N):
            base_x, base_y = self.baseline_pts[i]
            cur_x, cur_y = None, None

            # 1. Prefer baseline-referenced tracking if valid and displacement <= 22 px
            # (inter-dot spacing is 38 px; anything > 22 px is hopping into neighbor dot under shadow)
            if status_base[i] == 1:
                cand_x, cand_y = p_base[i]
                if np.hypot(cand_x - base_x, cand_y - base_y) <= 22.0:
                    cur_x, cur_y = cand_x, cand_y

            # 2. Fallback to frame-to-frame tracking
            if cur_x is None and status[i] == 1:
                cand_x, cand_y = p1[i]
                if np.hypot(cand_x - base_x, cand_y - base_y) <= 22.0:
                    cur_x, cur_y = cand_x, cand_y

            # 3. If lost or jumping unrealistically (> 22 px), snap back to baseline
            if cur_x is None:
                cur_x, cur_y = self.baseline_pts[i]

            dx = cur_x - base_x
            dy = cur_y - base_y
            disp = float(np.hypot(dx, dy))
            displacements.append((dx, dy, disp))

            if disp > max_disp:
                max_disp = disp

            # Check if this specific dot ID is indented / active
            is_active = (disp >= self.displacement_thresh_px)
            if is_active:
                active_ids.append(i)

            # Update current tracked position
            self.current_pts[i] = [cur_x, cur_y]

            # Visualization: Encircle physical black dot + center marker
            ix, iy = int(round(cur_x)), int(round(cur_y))
            bx, by = int(round(base_x)), int(round(base_y))

            if self.show_circles:
                # Fit contour/ring directly around the physical dark dot
                r_box = 12
                y1, y2 = max(0, iy - r_box), min(self.H, iy + r_box + 1)
                x1, x2 = max(0, ix - r_box), min(self.W, ix + r_box + 1)
                patch = cur_gray[y1:y2, x1:x2]
                ring_drawn = False

                ring_col = (0, 165, 255) if is_active else (0, 255, 180)  # Amber/Red if active, Emerald/Cyan if rest
                thick = 2 if is_active else 1

                if patch.shape[0] >= 8 and patch.shape[1] >= 8:
                    p_mean = float(np.mean(patch))
                    _, p_bin = cv2.threshold(patch, max(15, int(p_mean - 12)), 255, cv2.THRESH_BINARY_INV)
                    cnts, _ = cv2.findContours(p_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        # Find contour closest to dot center (r_box, r_box) rather than arbitrary noise
                        best_c = None
                        best_dist = 999.0
                        for c in cnts:
                            if cv2.contourArea(c) >= 10:
                                M = cv2.moments(c)
                                if M["m00"] > 0:
                                    mcx = M["m10"] / M["m00"]
                                    mcy = M["m01"] / M["m00"]
                                    d = float(np.hypot(mcx - r_box, mcy - r_box))
                                    if d < best_dist and d <= 6.0:
                                        best_dist = d
                                        best_c = c

                        if best_c is not None:
                            if len(best_c) >= 5 and cv2.contourArea(best_c) >= 15:
                                ellipse = cv2.fitEllipse(best_c)
                                (ecx, ecy), (ew, eh), eang = ellipse
                                ew_clamped = float(np.clip(ew, 6.0, 20.0))
                                eh_clamped = float(np.clip(eh, 6.0, 20.0))
                                full_ellipse = ((float(ecx + x1), float(ecy + y1)), (ew_clamped, eh_clamped), float(eang))
                                cv2.ellipse(vis, full_ellipse, ring_col, thick, lineType=cv2.LINE_AA)
                                ring_drawn = True
                            else:
                                (cx_c, cy_c), rad_c = cv2.minEnclosingCircle(best_c)
                                rad_clamped = int(np.clip(rad_c, 5, 12))
                                cv2.circle(vis, (int(cx_c + x1), int(cy_c + y1)), rad_clamped, ring_col, thick, lineType=cv2.LINE_AA)
                                ring_drawn = True

                if not ring_drawn:
                    # Fallback circular reticle ring directly centered at (ix, iy)
                    cv2.circle(vis, (ix, iy), 8, ring_col, thick, lineType=cv2.LINE_AA)

            if is_active:
                # Active / pressed dot: Center dot + Vector Arrow
                cv2.circle(vis, (ix, iy), 4 if not self.show_circles else 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(vis, (bx, by), (ix, iy), (0, 165, 255), 2, tipLength=0.35)
                # Text: ID (bright yellow)
                cv2.putText(vis, str(i), (ix + 7, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                # Rest dot: Subtle center core dot + ID
                cv2.circle(vis, (ix, iy), 3 if not self.show_circles else 2, (34, 197, 94), -1, lineType=cv2.LINE_AA)
                cv2.putText(vis, str(i), (ix + 5, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 255, 180), 1, lineType=cv2.LINE_AA)

        self.prev_gray = cur_gray.copy()

        # Overlay summary statistics on top-left of image
        cv2.putText(
            vis,
            f"Total Dots: {N} | Active Pressed: {len(active_ids)} | Max Disp: {max_disp:.2f}px",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        if active_ids:
            cv2.putText(
                vis,
                f"Indent IDs: {active_ids[:12]}" + ("..." if len(active_ids) > 12 else ""),
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 165, 255),
                2,
            )

        return {
            "num_dots": N,
            "active_ids": active_ids,
            "max_disp_px": max_disp,
            "displacements": displacements,
            "annotated_image": vis,
        }


def run_live_test(num_frames: int = 40):
    print("=" * 65)
    print("OPTICAL FLOW DOT TRACKER WITH PERSISTENT IDs (PROTOTYPE)")
    print("=" * 65)

    engine = OpticalFlowDotEngine()

    if not HAS_PICAMERA:
        sample_path = Path(__file__).resolve().parent / "unblocked_baseline.jpg"
        print(f"[INFO] Picamera2 not found locally. Running test on sample image '{sample_path}'...")
        img = cv2.imread(str(sample_path))
        if img is None:
            print("[ERROR] Sample image not found.")
            return

        num_dots = engine.calibrate(img)
        print(f"[TEST] Successfully calibrated {num_dots} dots with permanent IDs.")

        test_frame = img.copy()
        res = engine.process_frame(test_frame)

        out_dir = Path(__file__).resolve().parent.parent / "tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "optical_flow_static_test.jpg"
        cv2.imwrite(str(out_path), res["annotated_image"])
        print(f"[TEST] Saved test image to {out_path}")
        return

    # On Pi 5: Initialize Picamera2
    print("[Pi 5] Initializing Picamera2...")
    picam = Picamera2()
    cfg = picam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam.configure(cfg)
    picam.start()

    controls = {
        "AfMode": libcamera_controls.AfModeEnum.Manual,
        "LensPosition": 12.0,
    }
    picam.set_controls(controls)
    time.sleep(1.2)

    # Capture baseline
    print("[Pi 5] Capturing baseline frame...")
    frame_rgb = picam.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    num_dots = engine.calibrate(frame_bgr)
    if num_dots == 0:
        picam.stop()
        return

    print(f"\n[Pi 5] Running Optical Flow on {num_frames} live frames...")
    out_dir = Path("test/optical_flow_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    for f_idx in range(num_frames):
        t0 = time.perf_counter()
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        res = engine.process_frame(frame_bgr)
        dt = (time.perf_counter() - t0) * 1000.0

        if f_idx % 10 == 0 or len(res["active_ids"]) > 0:
            print(
                f"Frame {f_idx:02d} | Tracked: {res['num_dots']} dots | "
                f"Active Indent IDs ({len(res['active_ids'])}): {res['active_ids']} | "
                f"Max Disp: {res['max_disp_px']:.2f}px | Latency: {dt:.1f}ms"
            )
            cv2.imwrite(str(out_dir / f"flow_frame_{f_idx:03d}.jpg"), res["annotated_image"])

    total_time = time.perf_counter() - t_start
    fps = num_frames / total_time
    print(f"\n[BENCHMARK] Optical Flow Average Speed: {fps:.1f} FPS (Latency: {1000.0/fps:.1f} ms/frame)")

    picam.stop()
    print("=" * 65)


if __name__ == "__main__":
    run_live_test()
