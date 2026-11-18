"""
High-Speed Marker Dot Tracker with Subpixel Centroiding and Baseline Drift Compensation.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


class DotTracker:
    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        roi_center: Tuple[int, int] = (320, 240),
        roi_radius: int = 218,
        min_dot_area: float = 20.0,
        max_dot_area: float = 3000.0,
        min_circularity: float = 0.35,
        max_match_dist_px: float = 25.0,
        contact_threshold_px: float = 0.85,
    ):
        self.W = img_width
        self.H = img_height
        self.roi_center = roi_center
        self.roi_radius = roi_radius
        self.min_dot_area = min_dot_area
        self.max_dot_area = max_dot_area
        self.min_circularity = min_circularity
        self.max_match_dist_px = max_match_dist_px
        self.contact_threshold_px = contact_threshold_px

        # Precompute ROI mask
        yy, xx = np.ogrid[: self.H, : self.W]
        dist2 = (xx - roi_center[0]) ** 2 + (yy - roi_center[1]) ** 2
        self.roi_mask = dist2 <= (roi_radius**2)

        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Baseline reference dots: (N, 2)
        self.reference_dots: Optional[np.ndarray] = None
        self.is_calibrated = False

    def detect_dots(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Detects dot centroids with subpixel moments.
        Returns array of shape (N, 2) with (x, y) float coordinates.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        # Adaptive thresholding for dark dots on bright illuminated background
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            5,
        )
        thresh[~self.roi_mask] = 0

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.morph_kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        centroids = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 65.0 <= area <= self.max_dot_area:
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circ = 4.0 * np.pi * (area / (peri * peri))
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        r_center = np.hypot(cx - self.roi_center[0], cy - self.roi_center[1])
                        is_edge = (r_center >= 170.0)
                        min_c = 0.20 if is_edge else self.min_circularity
                        if circ < min_c:
                            continue

                        ix, iy = int(round(cx)), int(round(cy))

                        core = gray[max(0, iy - 2) : min(self.H, iy + 3), max(0, ix - 2) : min(self.W, ix + 3)]
                        core_mean = float(np.mean(core)) if core.size > 0 else 255.0

                        y_min, y_max = max(0, iy - 18), min(self.H, iy + 19)
                        x_min, x_max = max(0, ix - 18), min(self.W, ix + 19)
                        sub_y, sub_x = np.ogrid[y_min:y_max, x_min:x_max]
                        d2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2
                        ring_mask = (d2 >= 12**2) & (d2 <= 18**2)
                        ring_pixels = gray[y_min:y_max, x_min:x_max][ring_mask]
                        bg_mean = float(np.mean(ring_pixels)) if len(ring_pixels) > 0 else core_mean
                        contrast = bg_mean - core_mean

                        is_valid_dot = (contrast >= 25.0 and core_mean <= 68.0) or (is_edge and contrast >= 38.0)
                        if is_valid_dot:
                            centroids.append((cx, cy, area))

        if not centroids:
            return np.empty((0, 2), dtype=np.float32)

        # NMS suppression to reject artifacts closer than 38px
        clean_pts = []
        for x, y, a in sorted(centroids, key=lambda d: d[2], reverse=True):
            if not any(np.hypot(x - cx, y - cy) < 38.0 for cx, cy in clean_pts):
                clean_pts.append((x, y))

        return np.array(clean_pts, dtype=np.float32)

    def calibrate_baseline(self, frames: List[np.ndarray]) -> bool:
        """
        Averages dot positions over several uncontacted frames to build a clean baseline.
        """
        if not frames:
            return False

        all_detections = [self.detect_dots(f) for f in frames if f is not None]
        valid_dets = [d for d in all_detections if len(d) > 20]
        if not valid_dets:
            return False

        # Pick detection with most dots as seed
        seed_idx = int(np.argmax([len(d) for d in valid_dets]))
        seed = valid_dets[seed_idx]

        accum = np.zeros_like(seed)
        counts = np.zeros(len(seed), dtype=np.int32)

        for d in valid_dets:
            deltas = d[:, None, :] - seed[None, :, :]
            d2 = np.sum(deltas * deltas, axis=2)
            closest_seed = np.argmin(d2, axis=1)
            min_d2 = np.min(d2, axis=1)

            for cur_idx, s_idx in enumerate(closest_seed):
                if min_d2[cur_idx] <= (self.max_match_dist_px**2):
                    accum[s_idx] += d[cur_idx]
                    counts[s_idx] += 1

        valid_mask = counts >= max(1, len(valid_dets) // 2)
        final_ref = accum[valid_mask] / counts[valid_mask, None]

        self.reference_dots = final_ref.astype(np.float32)
        self.is_calibrated = len(self.reference_dots) > 10
        print(f"[DotTracker] Calibrated baseline with {len(self.reference_dots)} reference dots.")
        return self.is_calibrated

    def track(
        self, bgr_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Tracks current frame against calibrated baseline.
        Returns:
            matched_ref: (M, 2) reference positions
            matched_cur: (M, 2) current positions
            displacements: (M, 2) displacement vectors (cur - ref) corrected for global shift
            contact_detected: bool
        """
        if not self.is_calibrated or self.reference_dots is None:
            dots = self.detect_dots(bgr_image)
            return np.empty((0, 2)), dots, np.empty((0, 2)), False

        current_dots = self.detect_dots(bgr_image)
        if len(current_dots) == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), False

        # Match nearest reference dot
        ref = self.reference_dots
        deltas = current_dots[:, None, :] - ref[None, :, :]
        d2 = np.sum(deltas * deltas, axis=2)

        # Greedy assignment
        cur_idx, ref_idx = np.nonzero(d2 <= (self.max_match_dist_px**2))
        if cur_idx.size == 0:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), False

        dist_vals = d2[cur_idx, ref_idx]
        sort_order = np.argsort(dist_vals)

        matched_c = set()
        matched_r = set()
        pairs = []

        for idx in sort_order:
            c = int(cur_idx[idx])
            r = int(ref_idx[idx])
            if c not in matched_c and r not in matched_r:
                matched_c.add(c)
                matched_r.add(r)
                pairs.append((r, c))

        if not pairs:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), False

        r_indices = [p[0] for p in pairs]
        c_indices = [p[1] for p in pairs]

        m_ref = ref[r_indices]
        m_cur = current_dots[c_indices]
        raw_disps = m_cur - m_ref

        # Subtract median global drift (temperature/vibration compensation)
        global_drift = np.median(raw_disps, axis=0)
        corrected_disps = raw_disps - global_drift

        # Contact detection check
        magnitudes = np.hypot(corrected_disps[:, 0], corrected_disps[:, 1])
        active_dots = np.count_nonzero(magnitudes >= self.contact_threshold_px)
        contact_detected = active_dots >= 3

        return m_ref, m_cur, corrected_disps, contact_detected
