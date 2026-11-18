#!/usr/bin/env python3
"""
Pneumatic-Optical Standalone Force Estimator Module.
Implements the calibrated physics-informed model (piecewise runtime equation):
    effective_area_mm2 = segmented_mask_area_px2 * (area_scale_mm_per_px ** 2)
    if delta_pressure_hpa <= 0 or effective_area_mm2 <= 0:
        sensor_force_n = 0.0
    else:
        raw_pressure_area_force_n = delta_pressure_hpa * effective_area_mm2 / 10000.0
        sensor_force_n = max(0.0, alpha_pressure_area * raw_pressure_area_force_n + gamma_offset_n)

Where:
- alpha_pressure_area is the empirical calibration coefficient (2.7863)
- gamma_offset_n is the baseline offset (0.0031 N)
- area_scale_mm_per_px is the optical scale factor (0.171 mm/px)
- effective_area_mm2 is the image-derived effective projected contact area (not exact physical contact area)

Strict Constraints:
- Predictor inputs restricted strictly to: delta_pressure_hpa and contact mask area.
- PROHIBITED as predictors: stage_depth, stage_position, vision_depth,
  marker_displacement, loadcell_force, loadcell_raw, specimen dimensions.
- No silent hard-coded fallback for calibration. Missing file/keys fail loudly.
- Metrics strictly use Prediction R^2 (1 - SS_res / SS_tot) and report Pearson r separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class PneumaticOpticalForceEstimator:
    """
    Central standalone force estimator for tactile bubble sensor.
    Uses calibrated differential chamber pressure and image-derived effective projected contact area.
    """

    # Prohibited predictor keys that violate standalone integrity
    PROHIBITED_PREDICTORS = {
        "stage_depth_mm",
        "stage_position_mm",
        "vision_depth_mm",
        "marker_displacement_px",
        "loadcell_force_n",
        "loadcell_raw_g",
        "known_specimen_dimension",
        "specimen_id",
        "specimen_dimensions",
    }

    REQUIRED_CALIBRATION_KEYS = {
        "alpha_pressure_area",
        "gamma_offset_n",
        "area_scale_mm_per_px",
    }

    def __init__(
        self,
        calibration_path: Optional[Union[str, Path]] = None,
        alpha_pressure_area: Optional[float] = None,
        gamma_offset_n: Optional[float] = None,
        area_scale_mm_per_px: Optional[float] = None,
    ):
        """
        Initializes the force estimator.
        Must provide either calibration_path OR explicit coefficients.
        NO hard-coded fallback allowed. Missing configuration fails loudly.
        """
        if calibration_path is not None:
            calib_file = Path(calibration_path)
            if not calib_file.exists():
                raise FileNotFoundError(
                    f"Calibration file not found at: {calib_file}. "
                    "Fail-loud policy prohibits hard-coded fallbacks."
                )

            try:
                with open(calib_file, "r", encoding="utf-8") as f:
                    calib_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse calibration JSON at {calib_file}: {e}") from e

            # Verify all required keys exist
            missing_keys = self.REQUIRED_CALIBRATION_KEYS - set(calib_data.keys())
            if missing_keys:
                raise KeyError(
                    f"Calibration file {calib_file} is missing required keys: {sorted(list(missing_keys))}. "
                    f"Expected keys: {sorted(list(self.REQUIRED_CALIBRATION_KEYS))}. "
                    "Fail-loud policy strictly prohibits silent fallbacks."
                )

            self.alpha_pressure_area = float(calib_data["alpha_pressure_area"])
            self.gamma_offset_n = float(calib_data["gamma_offset_n"])
            self.area_scale_mm_per_px = float(calib_data["area_scale_mm_per_px"])
            self.calibration_metadata = calib_data

        elif (
            alpha_pressure_area is not None
            and gamma_offset_n is not None
            and area_scale_mm_per_px is not None
        ):
            self.alpha_pressure_area = float(alpha_pressure_area)
            self.gamma_offset_n = float(gamma_offset_n)
            self.area_scale_mm_per_px = float(area_scale_mm_per_px)
            self.calibration_metadata = {"source": "manual_injection"}
        else:
            raise ValueError(
                "PneumaticOpticalForceEstimator requires either a valid calibration_path "
                "or all 3 explicit parameters (alpha_pressure_area, gamma_offset_n, area_scale_mm_per_px). "
                "Silent hard-coded fallback is strictly forbidden."
            )

        if self.area_scale_mm_per_px <= 0.0:
            raise ValueError(f"area_scale_mm_per_px must be positive, got {self.area_scale_mm_per_px}")

    def compute_effective_area_mm2(self, segmented_mask_area_px2: float) -> float:
        """
        Converts raw 2D segmented contact mask pixel area to image-derived effective projected contact area in mm^2.
        Formula: A_mm2 = A_px2 * (s_px ^ 2)
        """
        if math.isnan(segmented_mask_area_px2) or math.isinf(segmented_mask_area_px2):
            raise ValueError(f"segmented_mask_area_px2 cannot be NaN or Inf, got {segmented_mask_area_px2}")
        if segmented_mask_area_px2 < 0.0:
            return 0.0
        return float(segmented_mask_area_px2 * (self.area_scale_mm_per_px ** 2))

    def compute_raw_force_n(self, delta_pressure_hpa: float, effective_area_mm2: float) -> float:
        """
        Computes raw hydrostatic force product:
            F_raw = delta_pressure_hpa * effective_area_mm2 / 10000.0
        Unit conversion:
            1 hPa = 100 N/m^2 = 0.0001 N/mm^2 = (1 / 10,000) N/mm^2.
        Negative pressure policy:
            Clamped to 0.0 (suction inside dome does not exert repulsive normal force on workpiece).
        """
        if math.isnan(delta_pressure_hpa) or math.isinf(delta_pressure_hpa):
            raise ValueError(f"delta_pressure_hpa cannot be NaN or Inf, got {delta_pressure_hpa}")
        if math.isnan(effective_area_mm2) or math.isinf(effective_area_mm2):
            raise ValueError(f"effective_area_mm2 cannot be NaN or Inf, got {effective_area_mm2}")

        safe_dp = max(0.0, float(delta_pressure_hpa))
        safe_area = max(0.0, float(effective_area_mm2))
        return float((safe_dp * safe_area) / 10000.0)

    def predict(
        self,
        delta_pressure_hpa: float,
        segmented_mask_area_px2: Optional[float] = None,
        effective_area_mm2: Optional[float] = None,
        clamp_non_negative: bool = True,
        **forbidden_kwargs: Any,
    ) -> float:
        """
        Predicts normal contact force F_sensor (N).
        Strict academic integrity check: rejects any attempt to pass prohibited predictors.
        """
        # Integrity Gate: Fail loudly if prohibited predictors are passed
        passed_prohibited = set(forbidden_kwargs.keys()) & self.PROHIBITED_PREDICTORS
        if passed_prohibited:
            raise TypeError(
                f"Prohibited predictor(s) passed to standalone force estimator: {sorted(list(passed_prohibited))}. "
                "The force model must operate purely on pneumatic pressure and optical contact area."
            )

        if math.isnan(delta_pressure_hpa) or math.isinf(delta_pressure_hpa):
            raise ValueError(f"delta_pressure_hpa cannot be NaN or Inf, got {delta_pressure_hpa}")

        # Area determination
        if effective_area_mm2 is not None:
            if math.isnan(effective_area_mm2) or math.isinf(effective_area_mm2):
                raise ValueError(f"effective_area_mm2 cannot be NaN or Inf, got {effective_area_mm2}")
            area_mm2 = max(0.0, float(effective_area_mm2))
        elif segmented_mask_area_px2 is not None:
            area_mm2 = self.compute_effective_area_mm2(segmented_mask_area_px2)
        else:
            raise ValueError("Must provide either effective_area_mm2 or segmented_mask_area_px2")

        # Zero contact boundary condition
        safe_dp = max(0.0, float(delta_pressure_hpa))
        if safe_dp <= 0.0 or area_mm2 <= 0.0:
            return 0.0

        f_raw_n = self.compute_raw_force_n(safe_dp, area_mm2)
        f_pred = self.alpha_pressure_area * f_raw_n + self.gamma_offset_n

        if clamp_non_negative:
            return float(max(0.0, f_pred))
        return float(f_pred)

    def predict_detailed(
        self,
        delta_pressure_hpa: float,
        segmented_mask_area_px2: Optional[float] = None,
        effective_area_mm2: Optional[float] = None,
        **forbidden_kwargs: Any,
    ) -> Dict[str, float]:
        """
        Returns full telemetry breakdown of the calculation for logging and verification.
        """
        passed_prohibited = set(forbidden_kwargs.keys()) & self.PROHIBITED_PREDICTORS
        if passed_prohibited:
            raise TypeError(
                f"Prohibited predictor(s) passed to standalone force estimator: {sorted(list(passed_prohibited))}."
            )

        if effective_area_mm2 is not None:
            if math.isnan(effective_area_mm2) or math.isinf(effective_area_mm2):
                raise ValueError(f"effective_area_mm2 cannot be NaN or Inf, got {effective_area_mm2}")
            area_mm2 = max(0.0, float(effective_area_mm2))
            area_px2 = area_mm2 / (self.area_scale_mm_per_px ** 2)
        elif segmented_mask_area_px2 is not None:
            area_px2 = max(0.0, float(segmented_mask_area_px2))
            area_mm2 = self.compute_effective_area_mm2(area_px2)
        else:
            raise ValueError("Must provide either effective_area_mm2 or segmented_mask_area_px2")

        safe_dp = max(0.0, float(delta_pressure_hpa))
        is_contact = (safe_dp > 0.0) and (area_mm2 > 0.0)
        f_raw_n = (safe_dp * area_mm2) / 10000.0 if is_contact else 0.0
        f_sensor_n = max(0.0, self.alpha_pressure_area * f_raw_n + self.gamma_offset_n) if is_contact else 0.0

        return {
            "delta_pressure_hpa": float(delta_pressure_hpa),
            "effective_area_mm2": float(area_mm2),
            "segmented_mask_area_px2": float(area_px2),
            "raw_pressure_area_force_n": float(f_raw_n),
            "sensor_force_n": float(f_sensor_n),
            "is_contact": bool(is_contact),
        }


# ==============================================================================
# AUDITED STATISTICAL METRIC FUNCTIONS
# ==============================================================================

def compute_prediction_r2(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Computes true Prediction R^2 (Coefficient of Determination):
        R^2 = 1 - (SS_res / SS_tot)
        SS_res = sum((y_true - y_pred)^2)
        SS_tot = sum((y_true - mean(y_true))^2)
    This strictly evaluates 1:1 prediction accuracy and penalizes scale and offset errors.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")
    if len(y_t) < 2:
        return 0.0

    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_pearson_r(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    Computes Pearson correlation coefficient r between y_true and y_pred.
    Evaluates linear co-variation independent of scaling or intercept bias.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_pred)={len(y_p)}")
    if len(y_t) < 2:
        return 0.0

    std_t = np.std(y_t)
    std_p = np.std(y_p)
    if std_t == 0.0 or std_p == 0.0:
        return 0.0

    r_matrix = np.corrcoef(y_t, y_p)
    val = float(r_matrix[0, 1])
    return 0.0 if np.isnan(val) else val


def compute_full_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, float]:
    """
    Computes full audited suite of prediction error metrics:
    - MAE
    - RMSE
    - Prediction R^2 (1 - SS_res / SS_tot)
    - Pearson r (reported separately)
    - Mean Bias
    - Maximum Absolute Error (MaxAE)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    residuals = y_p - y_t
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    pred_r2 = compute_prediction_r2(y_t, y_p)
    pearson_r = compute_pearson_r(y_t, y_p)
    bias = float(np.mean(residuals))
    max_ae = float(np.max(np.abs(residuals)))

    return {
        "mae_n": mae,
        "rmse_n": rmse,
        "prediction_r2": pred_r2,
        "pearson_r": pearson_r,
        "mean_bias_n": bias,
        "max_ae_n": max_ae,
    }
