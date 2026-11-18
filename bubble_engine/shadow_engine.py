#!/usr/bin/env python3
"""
Contact Shadow & Shading Analysis Engine (Stream 2 of Two-Stream Tactile Sensing).
Analyzes photometric shading, contact shadows, and micro-geometry on the translucent
silicone membrane independent of marker dot motion.
Fuses with Marker Tracking to enable High-Resolution Contour & Shape Classification.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ContactShadowEngine:
    def __init__(
        self,
        roi_center: Tuple[int, int] = (320, 240),
        roi_radius_px: int = 145,
        mm_per_px: float = 0.171,
        shadow_thresh_delta: float = 10.0,
    ):
        self.roi_center = roi_center
        self.roi_radius_px = roi_radius_px
        self.mm_per_px = mm_per_px
        self.shadow_thresh_delta = shadow_thresh_delta

        self.baseline_lab_l: Optional[np.ndarray] = None
        self.baseline_gray: Optional[np.ndarray] = None
        self.baseline_green: Optional[np.ndarray] = None
        self._from_clean_file = False

        # Precompute circular dome mask
        self.dome_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(self.dome_mask, self.roi_center, self.roi_radius_px, 255, -1)

        # Morphological kernels for bridging dot holes in shadow patch
        # Kernel 21x21 bridges intra-contact dot holes without bleeding into outer deformation slope (A_contact vs A_deformation)
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Check for saved clean baseline (unblocked membrane)
        from pathlib import Path
        base_file = Path(__file__).resolve().parent / "unblocked_baseline.jpg"
        if base_file.exists():
            clean_bgr = cv2.imread(str(base_file))
            if clean_bgr is not None:
                self.calibrate(clean_bgr, force=True)
                self._from_clean_file = True

    def calibrate(self, bgr_img: np.ndarray, force: bool = False):
        """
        Stores baseline illumination under resting (uncontacted) state.
        """
        if not force and self._from_clean_file:
            return
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        self.baseline_lab_l = lab[:, :, 0].astype(np.float32)
        self.baseline_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.baseline_green = bgr_img[:, :, 1].astype(np.float32)

    def process_frame(
        self,
        bgr_img: np.ndarray,
        marker_center_px: Optional[Tuple[float, float]] = None,
        is_contact_from_markers: bool = False,
        contact_tilt_deg: float = 0.0,
        cos_theta: float = 1.0,
    ) -> Dict:
        """
        Analyzes contact shadow & shading on the membrane.
        Returns contour, bounding box, dimensions, aspect ratio, and shape classification.
        """
        h, w = bgr_img.shape[:2]
        if self.baseline_green is None:
            self.calibrate(bgr_img)

        # Contact Validation Gate:
        # If no physical contact is confirmed via optical flow markers or chamber pressure,
        # reject ambient occlusions (e.g. hands/fingers waving in the air casting external shadows).
        if not is_contact_from_markers:
            return {
                "shadow_detected": False,
                "shape_name": "None",
                "aspect_ratio": 0.0,
                "effective_aspect": 0.0,
                "circularity": 0.0,
                "length_px": 0.0,
                "width_px": 0.0,
                "area_px2": 0.0,
                "length_mm": 0.0,
                "width_mm": 0.0,
                "area_mm2": 0.0,
                "corrected_l_mm": 0.0,
                "corrected_w_mm": 0.0,
                "contact_tilt_deg": 0.0,
                "center_px": (0.0, 0.0),
                "box_points": None,
                "contour": None,
                "shadow_mask": np.zeros_like(self.dome_mask, dtype=np.uint8),
            }

        # Photometric shadow on green-channel backscattering
        curr_green = bgr_img[:, :, 1].astype(np.float32)
        delta_l = np.maximum(0.0, self.baseline_green - curr_green).astype(np.uint8)

        # Region of interest: entire dome area inside self.dome_mask
        search_mask = self.dome_mask
        delta_roi = cv2.bitwise_and(delta_l, delta_l, mask=search_mask)

        # Threshold shadow darkening
        _, th = cv2.threshold(delta_roi, int(self.shadow_thresh_delta), 255, cv2.THRESH_BINARY)

        # Morphological operations to bridge the black marker dots inside contact patch
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.kernel_close)
        clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel_open)

        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_c = None
        best_score = -1.0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 150:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx, cy = 320.0, 240.0

            if marker_center_px is not None:
                dist = np.hypot(cx - marker_center_px[0], cy - marker_center_px[1])
                score = a / (1.0 + 0.02 * dist)
            else:
                dist_to_center = np.hypot(cx - 320.0, cy - 240.0)
                score = a / (1.0 + 0.01 * dist_to_center)

            if score > best_score:
                best_score = score
                best_c = c

        if best_c is None:
            return {
                "shadow_detected": False,
                "shape_name": "None",
                "aspect_ratio": 0.0,
                "circularity": 0.0,
                "length_px": 0.0,
                "width_px": 0.0,
                "area_px2": 0.0,
                "length_mm": 0.0,
                "width_mm": 0.0,
                "area_mm2": 0.0,
                "center_px": (0.0, 0.0),
                "box_points": None,
                "contour": None,
                "shadow_mask": clean,
            }

        # Compute convex hull of the contact contour to bridge internal marker dots
        hull = cv2.convexHull(best_c)
        best_area = float(cv2.contourArea(hull))
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect).astype(np.int32)
        rw, rh = rect[1]
        l_mm = float(max(rw, rh) * self.mm_per_px)
        w_mm = float(min(rw, rh) * self.mm_per_px)
        aspect = float(min(rw, rh) / max(rw, rh)) if max(rw, rh) > 0 else 0.0

        # Geometric Unwarping for Tilted Contact (Tangent Plane Projection)
        # When contact is tilted by theta, the projected dimension along the tilt axis shrinks by cos(theta).
        # Unwarp the compressed dimension to recover the true physical contact footprint:
        safe_cos = max(0.40, min(1.0, float(cos_theta)))
        unwarped_w = w_mm / safe_cos
        corrected_l_mm = float(max(l_mm, unwarped_w))
        corrected_w_mm = float(min(l_mm, unwarped_w))
        effective_aspect = min(1.0, float(corrected_w_mm / corrected_l_mm)) if corrected_l_mm > 0 else aspect

        peri = cv2.arcLength(hull, True)
        circ = float(4.0 * np.pi * best_area / (peri * peri)) if peri > 0 else 0.0

        # Geometric invariants for polygon & shape classification
        rect_area = rw * rh
        extent = float(best_area / rect_area) if rect_area > 0 else 0.0

        hull_peri = peri
        approx_hull = cv2.approxPolyDP(hull, 0.032 * hull_peri, True) if hull_peri > 0 else []
        num_vertices = len(approx_hull)

        # Multi-feature Shape Classification Hierarchy (using tilt-compensated effective aspect)
        if effective_aspect < 0.38:
            shape_name = "Cylinder / Bar / Edge"
        elif circ >= 0.84 and num_vertices >= 8:
            shape_name = "Circular / Sphere / Fingertip"
        elif num_vertices == 6 or (5 <= num_vertices <= 7 and 0.65 <= extent <= 0.83 and effective_aspect >= 0.72):
            shape_name = "Hexagon (6-Sided)"
        elif (num_vertices == 4 or extent >= 0.84) and effective_aspect >= 0.76:
            shape_name = "Flat Square / Cube"
        elif effective_aspect >= 0.38:
            shape_name = "Rectangular Box"
        else:
            shape_name = "Unknown / Freeform"

        return {
            "shadow_detected": True,
            "shape_name": shape_name,
            "aspect_ratio": aspect,
            "effective_aspect": effective_aspect,
            "circularity": circ,
            "length_px": float(max(rw, rh)),
            "width_px": float(min(rw, rh)),
            "area_px2": float(best_area),
            "length_mm": l_mm,
            "width_mm": w_mm,
            "area_mm2": float(best_area * (self.mm_per_px ** 2)),
            "corrected_l_mm": corrected_l_mm,
            "corrected_w_mm": corrected_w_mm,
            "contact_tilt_deg": contact_tilt_deg,
            "center_px": rect[0],
            "box_points": box,
            "contour": best_c,
            "shadow_mask": clean,
        }

    def render_overlay(
        self,
        annotated_bgr: np.ndarray,
        shadow_res: Dict,
    ) -> np.ndarray:
        """
        Draws the contact shadow contour and bounding box onto the image.
        """
        if not shadow_res["shadow_detected"]:
            return annotated_bgr

        vis = annotated_bgr.copy()
        box = shadow_res["box_points"]
        cnt = shadow_res["contour"]
        shape = shadow_res["shape_name"]
        l_mm = shadow_res["length_mm"]
        w_mm = shadow_res["width_mm"]
        c_l = shadow_res.get("corrected_l_mm", l_mm)
        c_w = shadow_res.get("corrected_w_mm", w_mm)
        tilt = shadow_res.get("contact_tilt_deg", 0.0)

        if box is not None:
            # Yellow bounding box for shadow footprint
            cv2.drawContours(vis, [box], 0, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        if cnt is not None:
            # Cyan outline for real shadow edge
            cv2.drawContours(vis, [cnt], 0, (255, 255, 0), 2, lineType=cv2.LINE_AA)

        # Label above box
        if box is not None:
            top_pt = min(box, key=lambda p: p[1])
            lx = max(10, int(top_pt[0]) - 30)
            ly = max(20, int(top_pt[1]) - 10)
            dim_str = f"{c_l:.1f}x{c_w:.1f}mm" if tilt < 12.0 else f"{c_l:.1f}x{c_w:.1f}mm @ {tilt:.0f}deg"
            cv2.putText(
                vis,
                f"Contact Patch: {dim_str}",
                (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        return vis
