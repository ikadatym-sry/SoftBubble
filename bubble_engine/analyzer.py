"""
Contact Object Geometry Analyzer with Temporal Debounce, EMA Smoothing, and Shape Persistence.
Measures physical dimensions (Length, Width, Height, Area, Volume) of contacted object.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from collections import deque, Counter
import numpy as np
import cv2


@dataclass
class ObjectMeasurement:
    contact_active: bool
    peak_height_mm: float = 0.0
    mean_height_mm: float = 0.0
    length_mm: float = 0.0
    width_mm: float = 0.0
    contact_area_mm2: float = 0.0
    volume_mm3: float = 0.0
    center_mm: Tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    shape_type: str = "None"
    bbox_corners_grid: Optional[np.ndarray] = None


class ObjectGeometryAnalyzer:
    def __init__(
        self,
        grid_size: int = 64,
        bubble_diameter_mm: float = 90.0,
        height_threshold_mm: float = 0.40,
    ):
        self.grid_size = grid_size
        self.bubble_diameter_mm = bubble_diameter_mm
        self.height_threshold_mm = height_threshold_mm

        # Cell size in mm (G-1 steps across bubble diameter)
        self.cell_size_mm = bubble_diameter_mm / float(grid_size - 1)
        self.cell_area_mm2 = self.cell_size_mm * self.cell_size_mm

        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Temporal smoothing & debounce buffers to eliminate jitter and flickering
        self.shape_history = deque(maxlen=9)
        self.smooth_length: Optional[float] = None
        self.smooth_width: Optional[float] = None
        self.smooth_peak_h: Optional[float] = None
        self.smooth_cx: Optional[float] = None
        self.smooth_cy: Optional[float] = None
        self.smooth_angle: Optional[float] = None
        self.last_bbox_corners: Optional[np.ndarray] = None
        self.no_contact_frames = 10

    def _reset_filters(self):
        self.shape_history.clear()
        self.smooth_length = None
        self.smooth_width = None
        self.smooth_peak_h = None
        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_angle = None
        self.last_bbox_corners = None

    def analyze(self, height_map_mm: np.ndarray) -> ObjectMeasurement:
        """
        Segments the contact footprint and computes object geometry with temporal debounce and stabilization.
        """
        if height_map_mm is None or height_map_mm.size == 0:
            return self._handle_no_contact()

        raw_contact = height_map_mm[height_map_mm >= self.height_threshold_mm]
        if raw_contact.size == 0:
            return self._handle_no_contact()

        # Robust plateau peak: 95th percentile avoids rim spikes while isolating true top contact face
        robust_peak = float(np.percentile(raw_contact, 95))
        adaptive_thresh = max(self.height_threshold_mm, 0.85 * robust_peak)
        mask = (height_map_mm >= adaptive_thresh).astype(np.uint8) * 255
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return self._handle_no_contact()

        # Largest contact contour
        largest_cnt = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(largest_cnt)

        if area_pixels < 3:
            return self._handle_no_contact()

        # Contact is firmly active
        self.no_contact_frames = 0

        # Fit Rotated Minimum Area Bounding Box
        rect = cv2.minAreaRect(largest_cnt)
        (center_gx, center_gy), (w_px, h_px), angle = rect
        box_points = cv2.boxPoints(rect)  # (4, 2)

        # Ensure length is the longer dimension
        if w_px < h_px:
            length_px, width_px = h_px, w_px
            angle_corrected = (angle + 90.0) % 180.0
        else:
            length_px, width_px = w_px, h_px
            angle_corrected = angle % 180.0

        # Membrane curvature compensation factor (1.08x) accounts for silicone corner compliance
        contact_scale = 1.08
        raw_length_mm = float(length_px * self.cell_size_mm) * contact_scale
        raw_width_mm = float(width_px * self.cell_size_mm) * contact_scale

        # Center in mm relative to bubble apex (0, 0)
        raw_center_x = float((center_gx - ((self.grid_size - 1) / 2.0)) * self.cell_size_mm)
        raw_center_y = float((center_gy - ((self.grid_size - 1) / 2.0)) * self.cell_size_mm)

        # Contact region statistics
        contact_indices = cleaned_mask > 0
        contact_heights = height_map_mm[contact_indices]

        peak_h = float(np.max(contact_heights)) if contact_heights.size else 0.0
        mean_h = float(np.mean(contact_heights)) if contact_heights.size else 0.0
        contact_area_mm2 = float(area_pixels * self.cell_area_mm2) * (contact_scale ** 2)
        volume_mm3 = float(np.sum(contact_heights) * self.cell_area_mm2)

        # Multi-feature geometric shape classifier with Contact Maturity Gating:
        # At very shallow indentation (peak_h < 1.8 mm or small area), the silicone dome
        # has only touched a fraction of a millimeter at the apex. The footprint is not yet
        # geometrically conformed to the object boundary, so we gate it as 'Contact Forming...'
        # to prevent premature mid-way misclassifications (e.g. guessing oval/cylinder before seating).
        if peak_h < 1.80 or contact_area_mm2 < 50.0:
            raw_shape_type = "Contact Forming..."
        else:
            perimeter = cv2.arcLength(largest_cnt, True)
            circularity = (
                (4.0 * np.pi * area_pixels) / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            bbox_area = max(w_px * h_px, 1.0)
            extent = float(area_pixels / bbox_area)
            aspect_ratio = raw_length_mm / max(raw_width_mm, 1e-3)

            # Convex Hull & Solidity (distinguishes straight edges/corners from round curvature)
            hull = cv2.convexHull(largest_cnt)
            solidity = float(area_pixels / max(cv2.contourArea(hull), 1.0))

            # Direct Least-Squares Ellipse Fit
            if len(largest_cnt) >= 5:
                (ecx, ecy), (ed1, ed2), eang = cv2.fitEllipse(largest_cnt)
                ellipse_area = np.pi * (ed1 / 2.0) * (ed2 / 2.0)
                ellipse_fit = float(area_pixels / max(ellipse_area, 1.0))
                ellipse_ar = float(max(ed1, ed2) / max(min(ed1, ed2), 1e-3))
            else:
                ellipse_fit = 0.0
                ellipse_ar = 1.0

            # Mathematical classification:
            # 1. Extent >= 0.785 (pi/4): In Euclidean geometry, no circle or ellipse can ever
            #    have an extent > pi/4 (~0.7854). Any shape exceeding 0.785 fills its bounding
            #    box more than an ellipse and MUST be a flat-edged polygon / box.
            if extent >= 0.785:
                raw_shape_type = "Rectangular / Box"
            elif max(aspect_ratio, ellipse_ar) >= 2.0:
                raw_shape_type = "Cylinder / Bar"
            elif solidity >= 0.915 and (ellipse_fit >= 0.70 or circularity >= 0.45):
                # Smooth rounded profile without sharp corner pinches
                if min(aspect_ratio, ellipse_ar) <= 1.35:
                    raw_shape_type = "Circular Disk / Cap"
                else:
                    raw_shape_type = "Oval / Ellipsoid"
            elif extent >= 0.75 or solidity < 0.915:
                # Straight-edged / polygonal profile (corners cause pin-cushion concavity on membrane)
                raw_shape_type = "Rectangular / Box"
            else:
                raw_shape_type = "Circular Disk / Cap" if aspect_ratio <= 1.25 else "Rectangular / Box"

        # Physical Dimension Recovery:
        # Measures true independent Length and Width along principal axes.
        # No artificial squaring or hardcoded dimension equality is applied,
        # allowing any rectangular, oblong, oval, or asymmetric geometry to be faithfully measured.
        if raw_shape_type in ("Circular Disk / Cap", "Disk / Sphere", "Oval / Ellipsoid"):
            # For smooth curved objects, measure directly from ellipse major/minor axes or bounding box
            if len(largest_cnt) >= 5:
                raw_length_mm = float(max(ed1, ed2) * self.cell_size_mm)
                raw_width_mm = float(min(ed1, ed2) * self.cell_size_mm)
            else:
                raw_length_mm = float(length_px * self.cell_size_mm)
                raw_width_mm = float(width_px * self.cell_size_mm)
            contact_area_mm2 = float(area_pixels * self.cell_area_mm2)
        else:
            # Rectangular / Box / Cylinder / Bar / Forming:
            # Measured directly from rotated bounding box with corner compliance scaling
            raw_length_mm = float(length_px * self.cell_size_mm) * contact_scale
            raw_width_mm = float(width_px * self.cell_size_mm) * contact_scale
            contact_area_mm2 = float(area_pixels * self.cell_area_mm2) * (contact_scale ** 2)

        # Apply Temporal EMA Smoothing (0.85 history, 0.15 new for calm steady values)
        if self.smooth_length is None:
            self.smooth_length = raw_length_mm
            self.smooth_width = raw_width_mm
            self.smooth_peak_h = peak_h
            self.smooth_cx = raw_center_x
            self.smooth_cy = raw_center_y
            self.smooth_angle = angle_corrected
        else:
            self.smooth_length = 0.85 * self.smooth_length + 0.15 * raw_length_mm
            self.smooth_width = 0.85 * self.smooth_width + 0.15 * raw_width_mm
            self.smooth_peak_h = 0.85 * self.smooth_peak_h + 0.15 * peak_h
            self.smooth_cx = 0.85 * self.smooth_cx + 0.15 * raw_center_x
            self.smooth_cy = 0.85 * self.smooth_cy + 0.15 * raw_center_y

            # Continuous angular smoothing avoiding 0-180 flip jumps
            diff_a = (angle_corrected - self.smooth_angle + 90.0) % 180.0 - 90.0
            self.smooth_angle = (self.smooth_angle + 0.15 * diff_a) % 180.0

        # Temporal Majority Voting with Immature Contact Gating & Inertia
        if raw_shape_type == "Contact Forming...":
            if not self.shape_history or all(s == "Contact Forming..." for s in self.shape_history):
                self.shape_history.append("Contact Forming...")
                stable_shape = "Contact Forming..."
            else:
                valid_shapes = [s for s in self.shape_history if s != "Contact Forming..."]
                stable_shape = Counter(valid_shapes).most_common(1)[0][0] if valid_shapes else "Contact Forming..."
        else:
            self.shape_history.append(raw_shape_type)
            valid_shapes = [s for s in self.shape_history if s != "Contact Forming..."]
            stable_shape = Counter(valid_shapes).most_common(1)[0][0]

        self.last_bbox_corners = box_points.astype(np.int32)

        final_l = float(self.smooth_length)
        final_w = float(self.smooth_width)

        return ObjectMeasurement(
            contact_active=True,
            peak_height_mm=float(self.smooth_peak_h),
            mean_height_mm=mean_h,
            length_mm=final_l,
            width_mm=final_w,
            contact_area_mm2=contact_area_mm2,
            volume_mm3=volume_mm3,
            center_mm=(float(self.smooth_cx), float(self.smooth_cy)),
            angle_deg=float(self.smooth_angle),
            shape_type=stable_shape,
            bbox_corners_grid=self.last_bbox_corners,
        )

    def _handle_no_contact(self) -> ObjectMeasurement:
        """
        Applies a 5-frame debounce before declaring contact lost to prevent momentary flickering.
        """
        self.no_contact_frames += 1
        if self.no_contact_frames >= 5:
            self._reset_filters()
            return ObjectMeasurement(contact_active=False)

        # Retain last stable measurement during momentary drop
        if self.smooth_length is not None and self.shape_history:
            stable_shape = Counter(self.shape_history).most_common(1)[0][0]
            return ObjectMeasurement(
                contact_active=True,
                peak_height_mm=float(self.smooth_peak_h or 0.0),
                mean_height_mm=float((self.smooth_peak_h or 0.0) * 0.8),
                length_mm=float(self.smooth_length),
                width_mm=float(self.smooth_width),
                contact_area_mm2=float(self.smooth_length * self.smooth_width * 0.8),
                volume_mm3=0.0,
                center_mm=(float(self.smooth_cx or 0.0), float(self.smooth_cy or 0.0)),
                angle_deg=float(self.smooth_angle or 0.0),
                shape_type=stable_shape,
                bbox_corners_grid=self.last_bbox_corners,
            )

        return ObjectMeasurement(contact_active=False)

    def draw_measurement_overlay(
        self,
        heatmap_bgr: np.ndarray,
        meas: ObjectMeasurement,
        display_scale: int = 4,
    ) -> np.ndarray:
        """
        Draws bounding box and measurements on an upscaled heatmap.
        """
        H, W = heatmap_bgr.shape[:2]
        vis = cv2.resize(
            heatmap_bgr, (W * display_scale, H * display_scale), interpolation=cv2.INTER_LINEAR
        )

        if not meas.contact_active or meas.bbox_corners_grid is None:
            cv2.putText(
                vis,
                "READY - NO CONTACT",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            return vis

        # Scaled bounding representation (Circle/Ellipse for rounded objects, Rotated Box for rectangular)
        if meas.shape_type in ("Circular Disk / Cap", "Disk / Sphere"):
            cgx = int(round((meas.center_mm[0] / self.cell_size_mm + (self.grid_size - 1) / 2.0) * display_scale))
            cgy = int(round((meas.center_mm[1] / self.cell_size_mm + (self.grid_size - 1) / 2.0) * display_scale))
            radius = int(round((meas.length_mm / (2.0 * self.cell_size_mm)) * display_scale))
            cv2.circle(vis, (cgx, cgy), radius, (0, 255, 255), 2, cv2.LINE_AA)
        elif meas.shape_type == "Oval / Ellipsoid":
            cgx = int(round((meas.center_mm[0] / self.cell_size_mm + (self.grid_size - 1) / 2.0) * display_scale))
            cgy = int(round((meas.center_mm[1] / self.cell_size_mm + (self.grid_size - 1) / 2.0) * display_scale))
            ax1 = int(round((meas.length_mm / (2.0 * self.cell_size_mm)) * display_scale))
            ax2 = int(round((meas.width_mm / (2.0 * self.cell_size_mm)) * display_scale))
            cv2.ellipse(vis, (cgx, cgy), (max(ax1, ax2), min(ax1, ax2)), int(meas.angle_deg), 0, 360, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            box_scaled = (meas.bbox_corners_grid * display_scale).astype(np.int32)
            cv2.drawContours(vis, [box_scaled], 0, (0, 255, 255), 2, cv2.LINE_AA)

        # Text overlay
        lines = [
            f"SHAPE: {meas.shape_type}",
            f"SIZE: {meas.length_mm:.1f} x {meas.width_mm:.1f} mm",
            f"HEIGHT: {meas.peak_height_mm:.2f} mm",
            f"AREA: {meas.contact_area_mm2:.1f} mm2",
        ]

        y_offset = 24
        for line in lines:
            cv2.putText(
                vis,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 20

        return vis
