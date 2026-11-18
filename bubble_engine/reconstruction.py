"""
High-Performance Tactile 3D Depth Map and Point Cloud Reconstruction Engine.
Reconstructs localized contact deformation profiles from marker displacements.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class TactileReconstructionEngine:
    def __init__(
        self,
        grid_size: int = 64,  # 64x64 grid
        bubble_diameter_mm: float = 90.0,
        bubble_max_depth_mm: float = 25.0,
        pixel_to_mm_scale: float = 0.21,  # ~90mm / 430px diameter
        depth_scale_mm_per_px: float = 1.75,  # physical mm indentation per pixel displacement
        noise_floor_px: float = 0.85,
    ):
        self.grid_size = grid_size
        self.bubble_diameter_mm = bubble_diameter_mm
        self.bubble_max_depth_mm = bubble_max_depth_mm
        self.pixel_to_mm_scale = pixel_to_mm_scale
        self.depth_scale_mm_per_px = depth_scale_mm_per_px
        self.noise_floor_px = noise_floor_px

        # Precompute circular grid mask
        gy, gx = np.ogrid[:self.grid_size, :self.grid_size]
        center = (self.grid_size - 1) / 2.0
        r_grid = np.hypot(gx - center, gy - center)
        self.grid_mask = r_grid <= (center * 0.95)

        # Real-world X, Y coordinates in mm
        half_w = (self.bubble_diameter_mm / 2.0)
        xs_mm = np.linspace(-half_w, half_w, self.grid_size)
        ys_mm = np.linspace(-half_w, half_w, self.grid_size)
        self.grid_x_mm, self.grid_y_mm = np.meshgrid(xs_mm, ys_mm)

        # Exponential Moving Average for temporal stability
        self._height_ema: Optional[np.ndarray] = None
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reconstruct_contact_surface(
        self,
        ref_dots: np.ndarray,
        displacements: np.ndarray,
        contact_detected: bool,
        img_w: int = 640,
        img_h: int = 480,
    ) -> np.ndarray:
        """
        Reconstructs localized 3D height profile Z (in mm) of the contacted object.
        Where there is no contact, Z is strictly 0.0 mm.
        """
        G = self.grid_size
        zero_surface = np.zeros((G, G), dtype=np.float32)

        if not contact_detected or len(ref_dots) == 0 or len(displacements) == 0:
            self._height_ema = None
            return zero_surface

        # Compute displacement magnitudes
        mags = np.hypot(displacements[:, 0], displacements[:, 1])
        active_indices = mags >= self.noise_floor_px

        # Require at least 1 active marker to form contact
        if np.count_nonzero(active_indices) < 1:
            self._height_ema = None
            return zero_surface

        active_pts = ref_dots[active_indices]
        active_mags = mags[active_indices]

        # Isotropic 1:1 mapping from circular camera ROI to square grid (eliminates aspect ratio distortion)
        roi_cx = img_w / 2.0
        roi_cy = img_h / 2.0
        roi_radius = 218.0  # pixels corresponding to bubble_diameter_mm / 2.0
        grid_radius = (G - 1) / 2.0
        scale = grid_radius / roi_radius

        px = grid_radius + (active_pts[:, 0] - roi_cx) * scale
        py = grid_radius + (active_pts[:, 1] - roi_cy) * scale
        px = np.clip(px, 0.0, float(G - 1))
        py = np.clip(py, 0.0, float(G - 1))
        pts_grid = np.column_stack([px, py]).astype(np.float32)

        # For 1 or 2 active markers: use localized Gaussian RBF indentation
        if len(active_pts) < 3:
            gy, gx = np.ogrid[:G, :G]
            contact_surface_mm = np.zeros((G, G), dtype=np.float32)
            sigma = 4.0  # ~6mm physical contact radius
            for pt, mag in zip(pts_grid, active_mags):
                d2 = (gx - pt[0]) ** 2 + (gy - pt[1]) ** 2
                h_peak = min(float(mag) * self.depth_scale_mm_per_px, self.bubble_max_depth_mm)
                contact_surface_mm = np.maximum(
                    contact_surface_mm, h_peak * np.exp(-d2 / (2.0 * sigma * sigma))
                )
            contact_surface_mm = np.where(self.grid_mask, contact_surface_mm, 0.0).astype(np.float32)

            if (
                self._height_ema is None
                or self._height_ema.shape != contact_surface_mm.shape
            ):
                self._height_ema = contact_surface_mm
            else:
                self._height_ema = 0.70 * self._height_ema + 0.30 * contact_surface_mm
            return self._height_ema.copy()

        # For 3+ markers: Convex hull of contact patch
        hull = cv2.convexHull(pts_grid.reshape(-1, 1, 2))
        if hull is None or len(hull) < 3:
            return zero_surface

        hull_mask = np.zeros((G, G), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)
        hull_mask = np.where(self.grid_mask, hull_mask, 0).astype(np.uint8)

        if np.count_nonzero(hull_mask) < 4:
            return zero_surface

        # Distance transform inside the contact hull (creates rounded dome/ridge shape)
        dist = cv2.distanceTransform(hull_mask, cv2.DIST_L2, 5)
        max_dist = float(np.max(dist))
        if max_dist <= 1e-4:
            return zero_surface

        # 2. Local displacement interpolation for intensity
        gy, gx = np.ogrid[:G, :G]
        dx_grid = gx[..., None] - px
        dy_grid = gy[..., None] - py
        dist2 = dx_grid * dx_grid + dy_grid * dy_grid

        sigma = 3.5
        weights = np.exp(-dist2 / (2.0 * sigma * sigma))
        w_sum = np.sum(weights, axis=-1)
        valid = w_sum > 1e-5

        local_mag = np.zeros((G, G), dtype=np.float32)
        local_mag[valid] = np.sum(weights[valid] * active_mags, axis=-1) / w_sum[valid]

        # Combine shape geometry (distance transform falloff) with displacement intensity
        peak_disp_px = float(np.percentile(active_mags, 95.0))
        peak_height_mm = min(
            peak_disp_px * self.depth_scale_mm_per_px, self.bubble_max_depth_mm
        )

        # Distance falloff from hull boundary (smooth Hermite curve over 3.0 cells)
        d_norm = np.clip(dist / 3.0, 0.0, 1.0)
        hull_falloff = d_norm * d_norm * (3.0 - 2.0 * d_norm)

        # Local displacement magnitude normalized to [0, 1]
        norm_disp = np.clip(local_mag / max(peak_disp_px, 1e-3), 0.0, 1.0)

        # Physical height field:
        # Direct contact area has high displacement and full elevation;
        # flank skirt naturally slopes down as marker displacement drops off
        contact_surface_mm = (hull_falloff * (norm_disp ** 0.75) * peak_height_mm).astype(np.float32)
        contact_surface_mm = cv2.GaussianBlur(contact_surface_mm, (0, 0), sigmaX=1.0, sigmaY=1.0)
        contact_surface_mm = np.where(hull_mask > 0, contact_surface_mm, 0.0)

        # Temporal EMA filter
        if (
            self._height_ema is None
            or self._height_ema.shape != contact_surface_mm.shape
        ):
            self._height_ema = contact_surface_mm
        else:
            self._height_ema = 0.70 * self._height_ema + 0.30 * contact_surface_mm

        return self._height_ema.copy()

    def generate_point_cloud(self, height_map_mm: np.ndarray) -> np.ndarray:
        """
        Generates 3D Point Cloud array of shape (N_points, 3) with [X_mm, Y_mm, Z_mm].
        """
        xs = self.grid_x_mm[self.grid_mask]
        ys = self.grid_y_mm[self.grid_mask]
        zs = height_map_mm[self.grid_mask]
        return np.column_stack([xs, ys, zs]).astype(np.float32)

    def generate_depth_heatmap(
        self, height_map_mm: np.ndarray, max_legend_mm: float = 8.0
    ) -> np.ndarray:
        """
        Generates a clean BGR heatmap for UI display with turbo colormap.
        Uncontacted membrane is dark slate; contact regions glow brightly.
        """
        norm = np.clip(height_map_mm / max(max_legend_mm, 1e-3), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)

        # Zero contact areas get dark theme
        zero_mask = height_map_mm < 0.20
        heatmap[zero_mask] = (25, 20, 15)  # Dark slate
        heatmap[~self.grid_mask] = 12

        return heatmap
