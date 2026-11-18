"""
Tactile 3D Dome Deformation Engine with Interactive Orbit Controls.
Renders a solid shaded 3D silicone dome with metallic base collar, Turbo colormap indentation crater,
perspective floor grid, and persistent dot markers with vector flow arrows.
Supports full 360-degree interactive rotation, tilting, zooming, and tuned deformation stiffness.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2


class Tactile3DDomeEngine:
    def __init__(
        self,
        dome_radius_mm: float = 45.0,
        dome_height_mm: float = 24.0,
        roi_center: Tuple[float, float] = (320.0, 240.0),
        roi_radius_px: float = 235.0,
        grid_resolution: int = 25,
        indent_scale_mm_per_px: float = 0.95,
        rbf_sigma_mm: float = 5.0,
        disp_thresh_px: float = 2.0,
    ):
        self.R_mm = dome_radius_mm
        self.H_mm = dome_height_mm
        self.roi_cx, self.roi_cy = roi_center
        self.roi_r_px = roi_radius_px
        self.G = grid_resolution
        self.indent_scale = indent_scale_mm_per_px
        self.sigma_mm = rbf_sigma_mm
        self.disp_thresh_px = disp_thresh_px

        # Interactive View Orientation (Front-facing 3D aligned with camera 1:1)
        self.azimuth_deg = 0.0
        self.elevation_deg = 35.0
        self.zoom_scale = 3.65

        # 3D Dome Surface Grid (X, Y, Z in mm)
        u = np.linspace(-self.R_mm, self.R_mm, self.G, dtype=np.float32)
        v = np.linspace(-self.R_mm, self.R_mm, self.G, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(u, v)
        self.grid_r = np.hypot(self.grid_x, self.grid_y)
        self.dome_mask = self.grid_r <= self.R_mm

        # Resting dome height profile (Hemispherical / Paraboloid blend)
        norm_r = np.clip(self.grid_r / self.R_mm, 0.0, 1.0)
        self.rest_z = np.zeros_like(self.grid_r)
        self.rest_z[self.dome_mask] = self.H_mm * np.sqrt(1.0 - norm_r[self.dome_mask] ** 2)

        # Baseline Dot 3D Coordinates
        self.baseline_2d: Optional[np.ndarray] = None
        self.baseline_3d: Optional[np.ndarray] = None  # (N, 3) [X, Y, Z]
        self.num_dots = 0

        # Precompute Turbo Colormap LUT (256 colors)
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO)
        self.turbo_lut = [tuple(int(c) for c in lut[i, 0]) for i in range(256)]

        # Light direction for diffuse shading (normalized)
        L = np.array([0.42, 0.52, 0.74], dtype=np.float32)
        self.L = L / np.linalg.norm(L)

        # Precompute Quad faces and their baseline midpoints for dynamic depth sorting
        self.face_indices = []
        face_mids = []
        for i in range(self.G - 1):
            for j in range(self.G - 1):
                inside = [
                    self.dome_mask[i, j],
                    self.dome_mask[i + 1, j],
                    self.dome_mask[i + 1, j + 1],
                    self.dome_mask[i, j + 1],
                ]
                if sum(inside) >= 3:
                    mid_x = (self.grid_x[i, j] + self.grid_x[i + 1, j + 1]) * 0.5
                    mid_y = (self.grid_y[i, j] + self.grid_y[i + 1, j + 1]) * 0.5
                    mid_z = (self.rest_z[i, j] + self.rest_z[i + 1, j + 1]) * 0.5
                    self.face_indices.append((i, j))
                    face_mids.append([mid_x, mid_y, mid_z])

        self.face_midpoints = np.array(face_mids, dtype=np.float32)
        self._update_rotation_matrix()

    def _update_rotation_matrix(self):
        el = np.radians(self.elevation_deg)
        az = np.radians(self.azimuth_deg)

        Rz = np.array(
            [[np.cos(az), -np.sin(az), 0.0],
             [np.sin(az),  np.cos(az), 0.0],
             [0.0,         0.0,        1.0]],
            dtype=np.float32,
        )
        Rx = np.array(
            [[1.0, 0.0,         0.0],
             [0.0, np.cos(el), -np.sin(el)],
             [0.0, np.sin(el),  np.cos(el)]],
            dtype=np.float32,
        )
        self.rot_matrix = Rx @ Rz

    def reset_view(self, mode: str = "iso"):
        if mode == "iso":
            self.azimuth_deg = 0.0
            self.elevation_deg = 35.0
            self.zoom_scale = 3.65
        elif mode == "top":
            self.azimuth_deg = 0.0
            self.elevation_deg = 88.0
            self.zoom_scale = 3.65
        elif mode == "side":
            self.azimuth_deg = 0.0
            self.elevation_deg = 12.0
            self.zoom_scale = 3.65
        self._update_rotation_matrix()

    def project_3d_to_screen(
        self,
        pts_3d: np.ndarray,
        view_w: int,
        view_h: int,
        offset_y: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects (..., 3) real-world mm coordinates to (sx, sy) screen pixel coordinates.
        """
        pts_rot = pts_3d @ self.rot_matrix.T
        cx = view_w / 2.0
        cy = view_h / 2.0 + offset_y
        sx = cx + pts_rot[..., 0] * self.zoom_scale
        sy = cy - pts_rot[..., 1] * self.zoom_scale
        return sx, sy

    def set_baseline_dots(self, dots_2d: np.ndarray):
        """
        Calculates resting 3D positions for each detected 2D dot centroid.
        Inverts Y so that image-top (-py) corresponds to +Y (North/Up) in 3D.
        """
        self.baseline_2d = dots_2d.copy()
        self.num_dots = len(dots_2d)

        px = dots_2d[:, 0] - self.roi_cx
        py = dots_2d[:, 1] - self.roi_cy

        scale_mm = self.R_mm / self.roi_r_px
        x_mm = px * scale_mm
        y_mm = -py * scale_mm  # Invert Y to match camera orientation 1:1
        r_mm = np.hypot(x_mm, y_mm)

        norm_r = np.clip(r_mm / self.R_mm, 0.0, 1.0)
        z_mm = self.H_mm * np.sqrt(np.maximum(0.0, 1.0 - norm_r**2))

        self.baseline_3d = np.column_stack([x_mm, y_mm, z_mm]).astype(np.float32)

    def filter_and_cluster_contact(
        self,
        active_ids: List[int],
        displacements: List[Tuple[float, float, float]],
        max_perimeter_radius_mm: float = 38.5,
        proximity_mm: float = 11.5,
    ) -> Dict:
        """
        Spatial clustering and perimeter noise rejection for robust contact footprint estimation.
        - Rejects outer perimeter dots (r > 38.5 mm) affected by silicone ballooning and glare.
        - Groups candidate active dots into contiguous contact clusters (step <= 11.5 mm).
        - Identifies the primary contact cluster with maximum indentation depth.
        - Hysteresis: includes immediate neighboring dots within 9.5 mm with displacement >= 2.4 px.
        - Computes localized contact footprint dimensions (Length x Width mm) and center (X, Y mm).
        """
        if self.baseline_3d is None or not active_ids:
            return {
                "primary_contact_ids": [],
                "is_contact": False,
                "footprint_l_mm": 0.0,
                "footprint_w_mm": 0.0,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "rejected_rim_ids": [],
            }

        # 1. Filter out dots on the clamped outer perimeter (where ballooning/glare occur)
        cand_ids = []
        rejected_rim_ids = []
        for did in active_ids:
            if did < self.num_dots:
                r_mm = np.hypot(self.baseline_3d[did, 0], self.baseline_3d[did, 1])
                if r_mm <= max_perimeter_radius_mm:
                    cand_ids.append(did)
                else:
                    rejected_rim_ids.append(did)

        if not cand_ids:
            return {
                "primary_contact_ids": [],
                "is_contact": False,
                "footprint_l_mm": 0.0,
                "footprint_w_mm": 0.0,
                "center_x_mm": 0.0,
                "center_y_mm": 0.0,
                "rejected_rim_ids": rejected_rim_ids,
            }

        # 2. Spatial proximity clustering (group connected touch regions)
        clusters = []
        visited = set()
        for did in cand_ids:
            if did in visited:
                continue
            cluster = [did]
            visited.add(did)
            queue = [did]
            while queue:
                curr = queue.pop(0)
                curr_xy = self.baseline_3d[curr, :2]
                for other in cand_ids:
                    if other not in visited:
                        other_xy = self.baseline_3d[other, :2]
                        if float(np.hypot(curr_xy[0] - other_xy[0], curr_xy[1] - other_xy[1])) <= proximity_mm:
                            visited.add(other)
                            cluster.append(other)
                            queue.append(other)
            clusters.append(cluster)

        # 3. Select primary contact cluster (highest cumulative indentation weight)
        best_cluster = max(
            clusters,
            key=lambda c: sum(max(0.0, displacements[d][2] - self.disp_thresh_px) for d in c)
        )

        # 4. Hysteresis expansion: include immediate neighboring dots (within 9.5mm of cluster)
        # that have displacement >= 2.4 px
        primary_set = set(best_cluster)
        for d in range(self.num_dots):
            if d in primary_set or d in rejected_rim_ids:
                continue
            r_mm = np.hypot(self.baseline_3d[d, 0], self.baseline_3d[d, 1])
            if r_mm > max_perimeter_radius_mm:
                continue
            d_xy = self.baseline_3d[d, :2]
            min_dist = min(float(np.hypot(d_xy[0] - self.baseline_3d[c, 0], d_xy[1] - self.baseline_3d[c, 1])) for c in best_cluster)
            if min_dist <= 9.5 and displacements[d][2] >= 2.4:
                primary_set.add(d)

        primary_contact_ids = sorted(list(primary_set))

        # 5. Footprint geometry calculation
        pts_mm = self.baseline_3d[primary_contact_ids, :2].astype(np.float32)
        if len(primary_contact_ids) >= 3:
            rect = cv2.minAreaRect(pts_mm)
            rw, rh = rect[1]
            margin = 5.0
            footprint_l = max(rw, rh) + margin
            footprint_w = min(rw, rh) + margin
            cx, cy = rect[0]

            # 4b. Interior contact infill: Only for broad contacts (>= 4 dots, rw/rh >= 12mm)
            # Under a flat indenter, central dots have near-zero 2D displacement.
            # Localized finger touch (< 4 dots) will NOT trigger interior infill.
            contact_r_eff = max(rw, rh) / 2.0
            if len(primary_contact_ids) >= 4 and contact_r_eff >= 6.0:
                for d in range(self.num_dots):
                    if d in primary_set or d in rejected_rim_ids:
                        continue
                    d_dist_to_c = float(np.hypot(self.baseline_3d[d, 0] - cx, self.baseline_3d[d, 1] - cy))
                    if d_dist_to_c <= contact_r_eff:
                        primary_set.add(d)
                primary_contact_ids = sorted(list(primary_set))
        elif len(primary_contact_ids) == 2:
            footprint_l = 10.0
            footprint_w = 6.0
            cx = float((pts_mm[0, 0] + pts_mm[1, 0]) * 0.5)
            cy = float((pts_mm[0, 1] + pts_mm[1, 1]) * 0.5)
        elif len(primary_contact_ids) == 1:
            footprint_l = 8.0
            footprint_w = 8.0
            cx = float(pts_mm[0, 0])
            cy = float(pts_mm[0, 1])
        else:
            footprint_l, footprint_w = 0.0, 0.0
            cx, cy = 0.0, 0.0

        is_cnt = len(primary_contact_ids) >= 1
        return {
            "primary_contact_ids": primary_contact_ids if is_cnt else [],
            "is_contact": is_cnt,
            "footprint_l_mm": float(footprint_l) if is_cnt else 0.0,
            "footprint_w_mm": float(footprint_w) if is_cnt else 0.0,
            "center_x_mm": float(cx) if is_cnt else 0.0,
            "center_y_mm": float(cy) if is_cnt else 0.0,
            "rejected_rim_ids": rejected_rim_ids,
        }

    def deform(
        self,
        current_2d: np.ndarray,
        displacements: List[Tuple[float, float, float]],
        active_ids: List[int],
    ) -> Dict:
        """
        Deforms the 3D dome surface based on optical flow displacements.
        Uses smooth onset (effective_disp = mag - threshold) and tuned stiffness.
        """
        if self.baseline_3d is None or len(current_2d) == 0:
            return {
                "deformed_z": self.rest_z.copy(),
                "depression_field": np.zeros_like(self.rest_z),
                "dots_3d_current": np.empty((0, 3)),
                "contact_active": False,
                "max_depression_mm": 0.0,
            }

        deformed_z = self.rest_z.copy()
        depression_field = np.zeros_like(self.rest_z)
        dots_3d = self.baseline_3d.copy()

        scale_mm = self.R_mm / self.roi_r_px
        max_depression = 0.0

        # Run spatial contact clustering & perimeter filter FIRST
        contact_info = self.filter_and_cluster_contact(active_ids, displacements)
        primary_ids = contact_info["primary_contact_ids"]
        primary_set = set(primary_ids)

        # Find cluster peak depth
        cluster_peak_depth = 0.0
        for did in primary_ids:
            if did < self.num_dots:
                d_indent = max(0.0, displacements[did][2] - self.disp_thresh_px) * self.indent_scale
                if d_indent > cluster_peak_depth:
                    cluster_peak_depth = d_indent

        # Update 3D dot positions
        for i in range(min(len(current_2d), self.num_dots)):
            dx_px, dy_px, mag_px = displacements[i]
            dx_mm = dx_px * scale_mm
            dy_mm = -dy_px * scale_mm  # Invert Y to match 3D coordinate frame

            # Tangential shear shift
            dots_3d[i, 0] = self.baseline_3d[i, 0] + dx_mm
            dots_3d[i, 1] = self.baseline_3d[i, 1] + dy_mm

            # Only depress in Z if this dot belongs to the actual primary contact cluster!
            # Distant dots that stretch tangentially will NOT create bogus vertical depressions.
            if i in primary_set:
                effective_disp = max(0.0, mag_px - self.disp_thresh_px)
                indent_mm = effective_disp * self.indent_scale

                # Infill interior dots (e.g. dome center under flat indenter)
                if contact_info.get("footprint_l_mm", 0.0) > 0.0 and cluster_peak_depth > 0:
                    d_to_c = float(np.hypot(self.baseline_3d[i, 0] - contact_info["center_x_mm"],
                                            self.baseline_3d[i, 1] - contact_info["center_y_mm"]))
                    r_contact = max(contact_info["footprint_l_mm"], contact_info["footprint_w_mm"]) / 2.0
                    if d_to_c <= r_contact and r_contact > 0.0:
                        plateau_indent = cluster_peak_depth * max(0.0, 1.0 - (d_to_c / (r_contact + 1e-3))**2)
                        indent_mm = max(indent_mm, plateau_indent)

                dots_3d[i, 2] = max(0.0, self.baseline_3d[i, 2] - indent_mm)
            else:
                dots_3d[i, 2] = self.baseline_3d[i, 2]

        # Smooth localized surface deformation via RBF on PRIMARY contact dots only!
        # This prevents perimeter stretching and lighting glare from creating bogus craters
        if primary_ids:
            sigma_dot = min(5.0, self.sigma_mm)
            two_sigma2 = 2.0 * (sigma_dot ** 2)
            cutoff2 = (2.5 * sigma_dot) ** 2

            for dot_id in primary_ids:
                if dot_id >= self.num_dots:
                    continue
                bx = self.baseline_3d[dot_id, 0]
                by = self.baseline_3d[dot_id, 1]
                _dx, _dy, mag = displacements[dot_id]
                effective_disp = max(0.0, mag - self.disp_thresh_px)
                peak_depth = effective_disp * self.indent_scale

                if peak_depth > max_depression:
                    max_depression = peak_depth

                dist2 = (self.grid_x - bx) ** 2 + (self.grid_y - by) ** 2
                weight = np.exp(-dist2 / two_sigma2)
                weight[dist2 > cutoff2] = 0.0
                depression_field = np.maximum(depression_field, peak_depth * weight)

            # Central infill: In a continuous planar contact patch (>= 3 dots, e.g. flat indenter),
            # depress the interior region between perimeter dots smoothly with strict localized cutoff.
            cx = contact_info["center_x_mm"]
            cy = contact_info["center_y_mm"]
            r_contact = max(contact_info["footprint_l_mm"], contact_info["footprint_w_mm"]) / 2.0
            if max_depression > 0.0 and r_contact >= 6.0 and len(primary_ids) >= 3:
                dist_c = np.hypot(self.grid_x - cx, self.grid_y - cy)
                r_outer = r_contact + 2.5
                in_mask = dist_c <= r_outer
                infill_field = np.zeros_like(depression_field)
                infill_field[in_mask] = max_depression * (np.cos(np.pi * dist_c[in_mask] / (2.0 * r_outer)) ** 2)
                depression_field = np.maximum(depression_field, infill_field)

            deformed_z = np.maximum(0.0, self.rest_z - depression_field)

        # Compute Engineering Telemetry for Tactile Sensing
        telemetry = {
            "contact_active": contact_info["is_contact"],
            "active_count": len(primary_ids),
            "total_dots": self.num_dots,
            "max_depth_mm": max_depression,
            "center_x_mm": contact_info["center_x_mm"],
            "center_y_mm": contact_info["center_y_mm"],
            "center_r_mm": float(np.hypot(contact_info["center_x_mm"], contact_info["center_y_mm"])),
            "footprint_l_mm": contact_info["footprint_l_mm"],
            "footprint_w_mm": contact_info["footprint_w_mm"],
            "primary_contact_ids": primary_ids,
            "rejected_rim_ids": contact_info["rejected_rim_ids"],
            "shear_mag_mm": 0.0,
            "shear_angle_deg": 0.0,
            "shear_desc": "None",
        }

        if primary_ids:
            weights = []
            xs, ys = [], []
            dxs, dys = [], []
            for dot_id in primary_ids:
                if dot_id < self.num_dots:
                    _dx, _dy, mag = displacements[dot_id]
                    eff = max(0.0, mag - self.disp_thresh_px)
                    w = eff * self.indent_scale
                    weights.append(w)
                    xs.append(self.baseline_3d[dot_id, 0])
                    ys.append(self.baseline_3d[dot_id, 1])
                    dxs.append(_dx * scale_mm)
                    dys.append(-_dy * scale_mm)  # Inverted Y

            sum_w = sum(weights)
            if sum_w > 1e-4:
                mean_dx = float(sum(dx * w for dx, w in zip(dxs, weights)) / sum_w)
                mean_dy = float(sum(dy * w for dy, w in zip(dys, weights)) / sum_w)
                smag = float(np.hypot(mean_dx, mean_dy))
                sang = float(np.degrees(np.arctan2(mean_dy, mean_dx))) % 360.0
                telemetry["shear_mag_mm"] = smag
                telemetry["shear_angle_deg"] = sang

                if smag < 0.35:
                    telemetry["shear_desc"] = "Direct Push (Normal)"
                elif 22.5 <= sang < 67.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Up-Right)"
                elif 67.5 <= sang < 112.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Up)"
                elif 112.5 <= sang < 157.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Up-Left)"
                elif 157.5 <= sang < 202.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Left)"
                elif 202.5 <= sang < 247.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Down-Left)"
                elif 247.5 <= sang < 292.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Down)"
                elif 292.5 <= sang < 337.5:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Down-Right)"
                else:
                    telemetry["shear_desc"] = f"Shear {smag:.1f}mm @ {sang:.0f} deg (Right)"

        return {
            "deformed_z": deformed_z,
            "depression_field": depression_field,
            "dots_3d_current": dots_3d,
            "contact_active": contact_info["is_contact"],
            "max_depression_mm": max_depression,
            "telemetry": telemetry,
        }

    def render_3d_view(
        self,
        deformed_z: np.ndarray,
        dots_3d_current: np.ndarray,
        active_ids: List[int],
        displacements: List[Tuple[float, float, float]],
        depression_field: Optional[np.ndarray] = None,
        telemetry: Optional[Dict] = None,
        view_w: int = 540,
        view_h: int = 480,
    ) -> np.ndarray:
        """
        Renders an interactive solid shaded 3D dome with metallic collar,
        diffuse lighting, Turbo colormap crater, and persistent dot markers.
        """
        canvas = np.full((view_h, view_w, 3), (12, 16, 22), dtype=np.uint8)

        # 1. Perspective Ground Floor Grid
        for g_val in np.linspace(-60.0, 60.0, 9):
            lx = np.array([g_val, g_val])
            ly = np.array([-60.0, 60.0])
            lz = np.array([-4.0, -4.0])
            sx_l, sy_l = self.project_3d_to_screen(np.column_stack([lx, ly, lz]), view_w, view_h)
            cv2.line(
                canvas,
                (int(sx_l[0]), int(sy_l[0])),
                (int(sx_l[1]), int(sy_l[1])),
                (24, 30, 40),
                1,
                lineType=cv2.LINE_AA,
            )
            sx_l2, sy_l2 = self.project_3d_to_screen(np.column_stack([ly, lx, lz]), view_w, view_h)
            cv2.line(
                canvas,
                (int(sx_l2[0]), int(sy_l2[0])),
                (int(sx_l2[1]), int(sy_l2[1])),
                (24, 30, 40),
                1,
                lineType=cv2.LINE_AA,
            )

        # 2. Bezel Base Ring & Metallic Collar
        theta = np.linspace(0, 2 * np.pi, 64)
        bx = self.R_mm * np.cos(theta)
        by = self.R_mm * np.sin(theta)
        bsx0, bsy0 = self.project_3d_to_screen(np.column_stack([bx, by, np.zeros_like(theta)]), view_w, view_h)
        bsxb, bsyb = self.project_3d_to_screen(np.column_stack([bx, by, np.full_like(theta, -4.0)]), view_w, view_h)

        collar_poly = []
        for k in range(len(theta)):
            collar_poly.append([int(bsx0[k]), int(bsy0[k])])
        for k in reversed(range(len(theta))):
            collar_poly.append([int(bsxb[k]), int(bsyb[k])])
        cv2.fillPoly(canvas, [np.array(collar_poly, dtype=np.int32)], (25, 32, 44))
        cv2.polylines(canvas, [np.column_stack([bsx0, bsy0]).astype(np.int32)], True, (55, 75, 98), 2, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [np.column_stack([bsxb, bsyb]).astype(np.int32)], True, (30, 40, 52), 1, lineType=cv2.LINE_AA)

        # 3. Dynamic Depth Sorting for Quads (Back to Front under current rotation)
        depths = self.face_midpoints @ self.rot_matrix[2]
        sorted_face_indices = np.argsort(depths)

        # 4. Surface projection & Diffuse Lighting
        pts_grid_3d = np.stack([self.grid_x, self.grid_y, deformed_z], axis=-1)
        sx, sy = self.project_3d_to_screen(pts_grid_3d, view_w, view_h)
        sx_i = np.round(sx).astype(np.int32)
        sy_i = np.round(sy).astype(np.int32)

        dz_dy, dz_dx = np.gradient(deformed_z, 90.0 / (self.G - 1))
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(nx)
        n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= n_len
        ny /= n_len
        nz /= n_len
        brightness = np.clip(0.38 + 0.62 * (nx * self.L[0] + ny * self.L[1] + nz * self.L[2]), 0.22, 1.0)

        if depression_field is None:
            depression_field = np.maximum(0.0, self.rest_z - deformed_z)

        # 5. Draw Shaded Polygons
        for f_idx in sorted_face_indices:
            i, j = self.face_indices[f_idx]
            poly = np.array(
                [
                    [sx_i[i, j], sy_i[i, j]],
                    [sx_i[i + 1, j], sy_i[i + 1, j]],
                    [sx_i[i + 1, j + 1], sy_i[i + 1, j + 1]],
                    [sx_i[i, j + 1], sy_i[i, j + 1]],
                ],
                dtype=np.int32,
            )
            b = brightness[i, j]
            ind = depression_field[i, j]

            if ind > 0.25:
                # Active contact crater: Turbo colormap glowing brighter with depth
                t = min(1.0, (ind - 0.25) / 4.5)
                base_col = self.turbo_lut[int(t * 255.0)]
                col = tuple(int(np.clip(c * (0.6 + 0.4 * b), 0, 255)) for c in base_col)
            else:
                # Uncontacted silicone skin: Sleek Three.js dark slate blue (RGB: 25, 48, 72)
                col = (int(72 * b), int(48 * b), int(25 * b))

            cv2.fillConvexPoly(canvas, poly, col, lineType=cv2.LINE_AA)
            cv2.polylines(
                canvas,
                [poly],
                True,
                (int(col[0] * 0.75), int(col[1] * 0.75), int(col[2] * 0.75)),
                1,
                lineType=cv2.LINE_AA,
            )

        # 6. Draw 3D Dot Markers on the Dome Surface
        if len(dots_3d_current) > 0 and self.baseline_3d is not None:
            primary_set = set(telemetry.get("primary_contact_ids", active_ids)) if telemetry else set(active_ids)
            all_active_set = set(active_ids)
            csx, csy = self.project_3d_to_screen(dots_3d_current, view_w, view_h)
            scale_mm = self.R_mm / self.roi_r_px

            for i in range(len(dots_3d_current)):
                cur_x, cur_y = int(round(csx[i])), int(round(csy[i]))

                if i in primary_set:
                    # Primary contact dot: Glowing red sphere + specular shine
                    cv2.circle(canvas, (cur_x, cur_y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (cur_x - 1, cur_y - 1), 2, (200, 200, 255), -1, lineType=cv2.LINE_AA)

                    # True Physical 3D Tangential Motion Arrow (Aligned 1:1 with Camera Feed):
                    # Project purely the surface slip vector (dx_mm, dy_mm) at the current deformed Z level.
                    # This completely eliminates vertical Z depression bias, guaranteeing arrow direction matches camera 100%.
                    if displacements is not None and i < len(displacements):
                        dx_px, dy_px, mag_px = displacements[i]
                        if mag_px >= 0.8:
                            t_base = np.array([
                                dots_3d_current[i, 0] - dx_px * scale_mm,
                                dots_3d_current[i, 1] - (-dy_px * scale_mm),
                                dots_3d_current[i, 2]
                            ], dtype=np.float32)
                            t_bsx, t_bsy = self.project_3d_to_screen(t_base.reshape(1, 3), view_w, view_h)
                            arr_dx = csx[i] - t_bsx[0]
                            arr_dy = csy[i] - t_bsy[0]
                            
                            # Clean visual scaling (gain 1.8x) so surface displacement is clearly visible
                            gain = 1.8
                            start_x = int(round(cur_x - arr_dx * gain))
                            start_y = int(round(cur_y - arr_dy * gain))
                            cv2.arrowedLine(canvas, (start_x, start_y), (cur_x, cur_y), (0, 165, 255), 2, tipLength=0.35)

                    cv2.putText(
                        canvas,
                        str(i),
                        (cur_x + 7, cur_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )
                elif i in all_active_set:
                    # Perimeter / stretch artifact: subtle amber outline, no motion arrow/alarm
                    cv2.circle(canvas, (cur_x, cur_y), 4, (30, 120, 220), 1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (cur_x, cur_y), 3, (34, 197, 94), -1, lineType=cv2.LINE_AA)
                else:
                    # Resting dot: Glossy emerald green marker
                    cv2.circle(canvas, (cur_x, cur_y), 4, (34, 197, 94), -1, lineType=cv2.LINE_AA)
                    cv2.circle(canvas, (cur_x - 1, cur_y - 1), 1, (200, 255, 200), -1, lineType=cv2.LINE_AA)
                    cv2.putText(
                        canvas,
                        str(i),
                        (cur_x + 5, cur_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,
                        (140, 210, 160),
                        1,
                        lineType=cv2.LINE_AA,
                    )

            # 6b. Prominent 3D Shear Slip Vector Arrow (Cyan/Gold) at Contact Center
            if telemetry and telemetry.get("contact_active", False) and telemetry.get("shear_mag_px", 0.0) >= 1.2:
                shear_mag = float(telemetry.get("shear_mag_px", 0.0))
                shear_deg = float(telemetry.get("shear_angle_deg", 0.0))
                cx_mm = float(telemetry.get("center_x_mm", 0.0))
                cy_mm = float(telemetry.get("center_y_mm", 0.0))
                cz_mm = float(telemetry.get("max_depth_mm", 0.0))

                center_r2 = cx_mm**2 + cy_mm**2
                center_z = max(0.0, self.H_mm * np.sqrt(max(0.0, 1.0 - center_r2 / (self.R_mm**2))) - cz_mm)
                c_3d = np.array([[cx_mm, cy_mm, center_z]], dtype=np.float32)
                csx_c, csy_c = self.project_3d_to_screen(c_3d, view_w, view_h)
                cc_x, cc_y = int(round(csx_c[0])), int(round(csy_c[0]))

                rad = np.radians(shear_deg)
                s_len_mm = float(np.clip(shear_mag * 1.5, 4.0, 18.0))
                tip_3d = np.array([[
                    cx_mm + s_len_mm * np.cos(rad),
                    cy_mm - s_len_mm * np.sin(rad),
                    center_z
                ]], dtype=np.float32)
                tsx, tsy = self.project_3d_to_screen(tip_3d, view_w, view_h)
                tip_x, tip_y = int(round(tsx[0])), int(round(tsy[0]))

                cv2.circle(canvas, (cc_x, cc_y), 6, (255, 255, 0), -1, lineType=cv2.LINE_AA)
                cv2.arrowedLine(canvas, (cc_x, cc_y), (tip_x, tip_y), (255, 255, 0), 3, tipLength=0.35)
                cv2.putText(canvas, f"SHEAR {shear_mag:.1f}px", (tip_x + 8, tip_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, lineType=cv2.LINE_AA)

        # 7. Live Engineering Telemetry HUD Card
        hud_x, hud_y, hud_w, hud_h = 12, 10, 440, 68
        overlay = canvas.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (14, 20, 28), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        is_contact = bool(telemetry["contact_active"]) if telemetry else (len(active_ids) > 0)
        border_col = (0, 165, 255) if is_contact else (40, 140, 70)
        cv2.rectangle(canvas, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), border_col, 1, lineType=cv2.LINE_AA)

        if telemetry and is_contact:
            max_d = telemetry["max_depth_mm"]
            cx = telemetry["center_x_mm"]
            cy = telemetry["center_y_mm"]
            cr = telemetry["center_r_mm"]
            n_act = telemetry["active_count"]
            n_tot = telemetry["total_dots"]
            s_desc = telemetry["shear_desc"]
            fl = telemetry.get("footprint_l_mm", 0.0)
            fw = telemetry.get("footprint_w_mm", 0.0)

            title_str = (
                f"STATUS: [CONTACT ACTIVE]  |  Depth: {max_d:.2f}mm  |  Footprint: {fl:.1f}x{fw:.1f}mm"
                if fl > 0
                else f"STATUS: [CONTACT ACTIVE]  |  Max Depth: {max_d:.2f} mm"
            )
            cv2.putText(
                canvas,
                title_str,
                (hud_x + 10, hud_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 220, 255),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Contact Center: X={cx:+.1f}, Y={cy:+.1f} mm  (Offset={cr:.1f}mm)",
                (hud_x + 10, hud_y + 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (225, 230, 235),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Active Dots: {n_act:2d}/{n_tot}  |  {s_desc}",
                (hud_x + 10, hud_y + 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 240, 180),
                1,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "STATUS: [READY]  |  Max Depth: 0.00 mm",
                (hud_x + 10, hud_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (34, 197, 94),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Calibrated: {self.num_dots} Dots Locked  |  Zero-Drift Baseline",
                (hud_x + 10, hud_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 200, 160),
                1,
                lineType=cv2.LINE_AA,
            )

        # Hint for mouse control
        cv2.putText(
            canvas,
            "Mouse Drag: Rotate 360 deg | Scroll: Zoom | 'r'=Reset | 't'=Top | 'p'=Side",
            (15, view_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (140, 150, 165),
            1,
            lineType=cv2.LINE_AA,
        )

        return canvas
