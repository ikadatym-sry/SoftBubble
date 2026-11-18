#!/usr/bin/env python3
"""
Shared data loading, calibration metric evaluation, and plotting utilities
for Soft Bubble Tactile Sensor offline analysis.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "analysis" / "figures"

# Planar calibration parameters evaluated across continuous time-series benchmarks
ALPHA_PAPER = 4.4305
GAMMA_PAPER = 0.3312

SHAPE_TITLES = {
    "triangle63.5-67": "Isosceles Triangle (63.5 x 63.5 x 67 mm)",
    "hexagon25": "Regular Hexagon (Side 25 mm)",
    "square60.5": "Equilateral Square (60.5 x 60.5 mm)",
    "square34": "Equilateral Square (34 x 34 mm)",
    "square45": "Equilateral Square (45 x 45 mm)",
    "hexagon35": "Regular Hexagon (Side 35 mm)",
    "square40": "Equilateral Square (40 x 40 mm)",
    "hexagon13": "Regular Hexagon (Side 13 mm)",
    "obtuse50-40": "Obtuse Quadrilateral (Base 50 mm, Height 40 mm)",
    "cylinder35": "Axisymmetric Cylinder (Ø35 mm)",
    "cylinder45": "Axisymmetric Cylinder (Ø45 mm)",
    "square30": "Equilateral Square (30 x 30 mm)",
    "square29": "Equilateral Square (29 x 29 mm)",
    "triangle_dome25-10": "Triangular Dome (Square Base 25x25 mm, Height 10 mm)",
    "hollow_cylinder25": "Hollow Cylinder (Ø25 mm)",
    "circular_dome36.5-15": "Circular Dome (Base Ø36.5 mm, Height 15 mm)",
    "square_dome24-12": "Square Dome (Base 24x24 mm, Height 12 mm)",
    "star27x29": "Star Indenter (Width 27 mm, Length 29 mm)",
    "heart28x30.5": "Heart Indenter (Width 28 mm, Length 30.5 mm)",
    "flat": "Flat Plate Baseline (50 x 50 mm)",
    "cylinder": "Cylinder Baseline (Ø15 mm)",
    "square": "Square Baseline (15 x 15 mm)",
    "sphere": "Curved Sphere Baseline (Ø15 mm)",
    "hexagon": "Hexagon Baseline (Side 22 mm)",
}


def resolve_data_root(data_dir: Optional[str] = None) -> Tuple[Path, str]:
    """Resolves CSV dataset root (custom --data-dir, full results/, or built-in sample_trials/)."""
    if data_dir:
        p = Path(data_dir).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Specified data directory not found: {p}")
        return p, "custom"

    for cand in (REPO_ROOT / "data" / "results", REPO_ROOT / "results"):
        if cand.exists() and any(cand.iterdir()):
            return cand, "full"

    sample_dir = REPO_ROOT / "data" / "sample_trials"
    if sample_dir.exists():
        return sample_dir, "sample"

    raise FileNotFoundError("No dataset directory found (checked results/, data/results/, data/sample_trials/).")


def find_shape_csvs(data_root: Path, shape_key: str) -> List[Path]:
    """Finds trial CSV files for a given shape under folder layout or flat sample_trials layout."""
    shape_dir = data_root / shape_key
    if shape_dir.is_dir():
        csvs = sorted(shape_dir.glob("*.csv"))
        if csvs:
            return csvs

    # Flat sample_trials naming: <shape>_trial_<idx>_<timestamp>.csv
    return sorted(data_root.glob(f"{shape_key}_trial_*.csv"))


def discover_shapes(data_root: Path) -> List[str]:
    """Lists all geometry keys available in data_root."""
    shapes = set()
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir() and sub.name not in {"snapshots"} and any(sub.glob("*.csv")):
            shapes.add(sub.name)
    for csv_path in sorted(data_root.glob("*_trial_*.csv")):
        shapes.add(csv_path.name.split("_trial_")[0])
    return sorted(shapes)


def load_shape_trials(
    csv_paths: List[Path],
    alpha: float = ALPHA_PAPER,
    gamma: float = GAMMA_PAPER,
    min_loadcell_n: float = 0.0,
) -> Tuple[List[Dict], List[Dict], float]:
    """
    Loads 11-column trial CSVs, estimates resting chamber baseline pressure P0
    as the median of initial unpressed rows, and returns pressed contact samples,
    per-trial time series, and P0.
    """
    resting_pressures: List[float] = []
    trials_raw: List[List[Dict[str, str]]] = []

    for fpath in csv_paths:
        with open(fpath, mode="r", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        for r in rows[:40]:
            if r["Pressed?"].strip().lower() in ("false", "0", "0.0"):
                resting_pressures.append(float(r["Pressure_hPa"]))
        trials_raw.append(rows)

    base_p = float(np.median(resting_pressures)) if resting_pressures else 1045.0

    pressed_samples: List[Dict] = []
    trials_processed: List[Dict] = []

    for rows in trials_raw:
        t_arr, flc_arr, fpred_arr, dp_arr, ac_arr = [], [], [], [], []
        for idx, r in enumerate(rows):
            flc = float(r["LoadCell_N"])
            p_val = float(r["Pressure_hPa"])
            dp = max(0.0, p_val - base_p)
            ac = float(r["Area_mm2"])
            pressed = r["Pressed?"].strip().lower() in ("true", "1", "1.0")
            fpred = (alpha * (dp * ac) / 10000.0 + gamma) if pressed else 0.0

            ts_str = r["Timestamp"].replace("Time: ", "").strip()
            parts = ts_str.split(":")
            sec = float(parts[0]) * 60.0 + float(parts[1]) if len(parts) == 2 else idx * 0.08

            t_arr.append(sec)
            flc_arr.append(flc)
            fpred_arr.append(fpred)
            dp_arr.append(dp)
            ac_arr.append(ac)

            if pressed and flc >= min_loadcell_n:
                pressed_samples.append({
                    "flc": flc,
                    "fpred": fpred,
                    "dp": dp,
                    "ac": ac,
                    "p_val": p_val,
                    "ts": r["Timestamp"],
                })

        trials_processed.append({
            "times": np.array(t_arr),
            "flc": np.array(flc_arr),
            "fpred": np.array(fpred_arr),
            "dp": np.array(dp_arr),
            "ac": np.array(ac_arr),
        })

    return pressed_samples, trials_processed, base_p


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, Max Abs Error, R^2 (1 - SS_res/SS_tot), and linear regression."""
    err = np.abs(y_pred - y_true)
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    if np.std(y_true) > 1e-5:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
    else:
        slope, intercept = 1.0, 0.0
    return {
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(err)),
        "r2": float(r2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def generate_multi_shape_parity_plot(
    target_shapes: List[Dict],
    title: str,
    header_label: str,
    model_line: str,
    r2_label: str,
    sample_count_label: str,
    output_png: Path,
    output_csv: Optional[Path] = None,
    data_dir: Optional[str] = None,
    samples_per_shape: int = 500,
    seed: int = 42,
    alpha: float = ALPHA_PAPER,
    gamma: float = GAMMA_PAPER,
) -> Dict[str, float]:
    """Samples contact points equally across geometries and renders a parity verification plot."""
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "lines.linewidth": 1.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    data_root, source_kind = resolve_data_root(data_dir)
    rng = np.random.default_rng(seed)

    collected_rows: List[Dict] = []
    all_flc: List[float] = []
    all_fpred: List[float] = []
    plotted_series: List[Dict] = []

    print(f"Loading data from {data_root} ({source_kind} dataset)...")
    for item in target_shapes:
        key = item["key"]
        csvs = find_shape_csvs(data_root, key)
        if not csvs:
            raise FileNotFoundError(f"No CSVs found for '{key}' in {data_root}")

        valid_pts, _, _ = load_shape_trials(csvs, alpha=alpha, gamma=gamma, min_loadcell_n=0.20)
        n_avail = len(valid_pts)
        n_take = min(samples_per_shape, n_avail)
        if n_take < 10:
            raise ValueError(f"Too few pressed points for '{key}' ({n_avail})")

        idx_sampled = rng.choice(n_avail, size=n_take, replace=False)
        sampled = [valid_pts[int(i)] for i in idx_sampled]

        sub_flc = np.array([p["flc"] for p in sampled])
        sub_fpred = np.array([p["fpred"] for p in sampled])
        shape_metrics = compute_metrics(sub_flc, sub_fpred)
        print(f"  • {key:<20}: sampled {n_take}/{n_avail} pts | R²={shape_metrics['r2']*100:.2f}% | MAE={shape_metrics['mae']:.4f} N")

        plotted_series.append({
            **item,
            "sampled_flc": sub_flc,
            "sampled_fpred": sub_fpred,
        })
        all_flc.extend(sub_flc)
        all_fpred.extend(sub_fpred)

        for p in sampled:
            collected_rows.append({
                "Shape": key,
                "Label": item["legend_label"],
                "Timestamp": p["ts"],
                "LoadCell_N": p["flc"],
                "F_sensor_N": p["fpred"],
                "Delta_P_hPa": p["dp"],
                "Area_mm2": p["ac"],
            })

    y_true = np.array(all_flc)
    y_pred = np.array(all_fpred)
    metrics = compute_metrics(y_true, y_pred)
    sign_str = f"+ {metrics['intercept']:.2f}" if metrics["intercept"] >= 0 else f"- {abs(metrics['intercept']):.2f}"

    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    fig.patch.set_facecolor("#ffffff")

    max_force = 5.6
    x_band = np.linspace(0, max_force, 100)
    ax.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5, color="#cbd5e1", alpha=0.45, zorder=1)
    ax.fill_between(x_band, np.maximum(0, x_band - 0.25), x_band + 0.25, color="#94a3b8", alpha=0.35, zorder=1)
    ax.text(5.45, 4.82, "±0.25 N", fontsize=9, color="#475569", ha="right", va="center", rotation=45)
    ax.text(5.45, 4.45, "±0.50 N", fontsize=9, color="#64748b", ha="right", va="center", rotation=45)

    ax.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--", linewidth=1.8, zorder=2)

    for s in plotted_series:
        ax.scatter(
            s["sampled_flc"],
            s["sampled_fpred"],
            color=s["color"],
            marker=s["marker"],
            s=34,
            alpha=0.62,
            edgecolors="none",
            label=s["legend_label"],
            zorder=s.get("zorder", 4),
        )

    x_fit = np.linspace(0.4, 5.2, 50)
    ax.plot(x_fit, metrics["slope"] * x_fit + metrics["intercept"], color="#dc2626", linestyle="-", linewidth=2.2, zorder=5)

    stats_box = (
        f"{header_label}\n"
        f"{model_line}\n"
        f"Linear Fit: $y = {metrics['slope']:.2f}x {sign_str}$\n"
        f"------------------------------------\n"
        f"{r2_label}: {metrics['r2']*100:.2f}%\n"
        f"Mean Abs Error (MAE): {metrics['mae']:.4f} N\n"
        f"Root Mean Sq Error (RMSE): {metrics['rmse']:.4f} N\n"
        f"Max Abs Error: {metrics['max_err']:.4f} N\n"
        f"{sample_count_label}: {len(y_true):,} pts"
    )
    ax.text(
        0.04, 0.96, stats_box, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95, linewidth=1.2),
        fontsize=9.5, family="monospace",
    )

    ax.set_xlim(-0.1, max_force)
    ax.set_ylim(-0.1, max_force)
    ax.set_xlabel("Ground Truth Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
    ax.set_ylabel("Predicted Model Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(
        loc="lower right", framealpha=0.96, edgecolor="#94a3b8", facecolor="#ffffff",
        fontsize=11.5, markerscale=1.4, borderpad=0.65, labelspacing=0.5, handletextpad=0.4,
    )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"✅ Saved parity plot -> {output_png}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=["Shape", "Label", "Timestamp", "LoadCell_N", "F_sensor_N", "Delta_P_hPa", "Area_mm2"]
            )
            writer.writeheader()
            writer.writerows(collected_rows)
        print(f"✅ Saved sampled CSV -> {output_csv}")

    return metrics
