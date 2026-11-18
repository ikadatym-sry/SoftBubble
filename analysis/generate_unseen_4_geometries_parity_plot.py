#!/usr/bin/env python3
"""
Publication-Quality Cross-Geometry Generalization Parity Plot
------------------------------------------------------------
Strictly evaluates 4 Previously Unseen Geometries (Excluding Flat Calibration Baseline):
- Cylinder
- Square
- Sphere
- Triangle
Sampling: 500 points per geometry (N = 2,000 total points, random seed = 42)
"""

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    REPO_ROOT / 'data' / 'results',
    REPO_ROOT / 'results',
    REPO_ROOT.parent / 'TestVersion1' / 'results',
    Path('/Users/aerkjaa0616/Work/Pong/TestVersion1/results')
]
RESULTS_BASE = next((str(c) for c in CANDIDATES if c.exists()), str(REPO_ROOT / 'data' / 'results'))
OUTPUT_DIR = REPO_ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = str(OUTPUT_DIR / "unseen_4_geometries_parity_plot.png")
OUTPUT_CSV = str(REPO_ROOT / "data" / "unseen_4_geometries_sampled_data.csv")
ARTIFACT_DIR = "/Users/aerkjaa0616/.gemini/antigravity/brain/8d8c79aa-f65a-47f0-bf3d-8292121fb34c"

ALPHA = 4.4305
GAMMA = 0.3312

# Strictly 4 Unseen shapes (Excluding Flat baseline)
TARGET_SHAPES = [
    {
        "key": "cylinder",
        "legend_label": "Cylinder",
        "color": "#d97706",  # Amber / Golden Orange
        "marker": "D",
        "zorder": 4
    },
    {
        "key": "square",
        "legend_label": "Square",
        "color": "#059669",  # Emerald Green
        "marker": "s",
        "zorder": 4
    },
    {
        "key": "sphere",
        "legend_label": "Sphere",
        "color": "#7c3aed",  # Purple / Violet
        "marker": "^",
        "zorder": 4
    },
    {
        "key": "triangle_dome25-10",
        "legend_label": "Triangle",
        "color": "#0891b2",  # Cyan / Teal
        "marker": "v",
        "zorder": 4
    }
]

np.random.seed(42)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 300
})

collected_data = []
all_flc = []
all_fpred = []

for item in TARGET_SHAPES:
    key = item["key"]
    folder = os.path.join(RESULTS_BASE, key)
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found for {key} in {folder}")

    resting = []
    rows_all = []
    for c in csvs:
        with open(c, mode="r", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
            rows_all.append(rows)
            for r in rows[:40]:
                if r["Pressed?"].strip().lower() in ["false", "0", "0.0"]:
                    resting.append(float(r["Pressure_hPa"]))
    p0 = float(np.median(resting)) if resting else 1045.0

    valid_pts = []
    for rows in rows_all:
        for r in rows:
            pressed = r["Pressed?"].strip().lower() in ["true", "1", "1.0"]
            flc = float(r["LoadCell_N"])
            p_val = float(r["Pressure_hPa"])
            ac = float(r["Area_mm2"])
            if pressed and flc >= 0.20:
                dp = max(0.0, p_val - p0)
                fpred = (ALPHA * (dp * ac) / 10000.0 + GAMMA)
                valid_pts.append({
                    "flc": flc,
                    "fpred": fpred,
                    "dp": dp,
                    "ac": ac,
                    "p_val": p_val,
                    "ts": r["Timestamp"]
                })

    n_available = len(valid_pts)
    if n_available < 500:
        raise ValueError(f"Not enough points in {key} ({n_available} < 500)")

    idx_sampled = np.random.choice(n_available, size=500, replace=False)
    sampled = [valid_pts[i] for i in idx_sampled]

    sub_flc = np.array([p["flc"] for p in sampled])
    sub_fpred = np.array([p["fpred"] for p in sampled])

    err = np.abs(sub_fpred - sub_flc)
    mae_shape = np.mean(err)
    rmse_shape = np.sqrt(np.mean(err**2))
    ss_tot = np.sum((sub_flc - np.mean(sub_flc))**2)
    ss_res = np.sum((sub_flc - sub_fpred)**2)
    r2_shape = 1.0 - (ss_res / ss_tot)

    item["sampled_flc"] = sub_flc
    item["sampled_fpred"] = sub_fpred
    item["r2"] = r2_shape
    item["mae"] = mae_shape
    item["rmse"] = rmse_shape

    all_flc.extend(sub_flc)
    all_fpred.extend(sub_fpred)

    for p in sampled:
        collected_data.append({
            "Shape": key,
            "Label": item["legend_label"],
            "Timestamp": p["ts"],
            "LoadCell_N": p["flc"],
            "F_sensor_N": p["fpred"],
            "Delta_P_hPa": p["dp"],
            "Area_mm2": p["ac"]
        })

y_true = np.array(all_flc)
y_pred = np.array(all_fpred)
n_total = len(y_true)

overall_err = np.abs(y_pred - y_true)
overall_mae = float(np.mean(overall_err))
overall_rmse = float(np.sqrt(np.mean(overall_err**2)))
overall_max_err = float(np.max(overall_err))
ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
ss_res = float(np.sum((y_true - y_pred)**2))
overall_r2 = float(1.0 - (ss_res / ss_tot))
slope, intercept = np.polyfit(y_true, y_pred, 1)
sign_str = f"+ {intercept:.2f}" if intercept >= 0 else f"- {abs(intercept):.2f}"

# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 7.2))
fig.patch.set_facecolor("#ffffff")

max_force = 5.6

# 1. Shaded error bands
x_band = np.linspace(0, max_force, 100)
ax.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5,
                color="#cbd5e1", alpha=0.45, zorder=1)
ax.fill_between(x_band, np.maximum(0, x_band - 0.25), x_band + 0.25,
                color="#94a3b8", alpha=0.35, zorder=1)

ax.text(5.45, 4.82, "±0.25 N", fontsize=9, color="#475569", ha="right", va="center", rotation=45, fontweight="medium")
ax.text(5.45, 4.45, "±0.50 N", fontsize=9, color="#64748b", ha="right", va="center", rotation=45, fontweight="medium")

# 2. Ideal line (y = x)
ax.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--",
        linewidth=1.8, zorder=2)

# 3. Scatter points for the 4 unseen shapes
for item in TARGET_SHAPES:
    ax.scatter(
        item["sampled_flc"],
        item["sampled_fpred"],
        color=item["color"],
        marker=item["marker"],
        s=34,
        alpha=0.62,
        edgecolors="none",
        label=item["legend_label"],
        zorder=item["zorder"]
    )

# 4. Overall Linear fit line
x_fit = np.linspace(0.4, 5.2, 50)
ax.plot(x_fit, slope * x_fit + intercept, color="#dc2626", linestyle="-",
        linewidth=2.2, zorder=5)

# 5. Annotation box (top-left)
stats_box = (
    f"Evaluation: 4 Unseen Geometries\n"
    f"Model Locked: Flat Baseline\n"
    f"Linear Fit: $y = {slope:.2f}x {sign_str}$\n"
    f"------------------------------------\n"
    f"Generalization ($R^2$): {overall_r2*100:.2f}%\n"
    f"Mean Abs Error (MAE): {overall_mae:.4f} N\n"
    f"Root Mean Sq Error (RMSE): {overall_rmse:.4f} N\n"
    f"Max Abs Error: {overall_max_err:.4f} N\n"
    f"Total Samples ($N_{{test}}$): {n_total:,} pts"
)
ax.text(0.04, 0.96, stats_box, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95, linewidth=1.2),
        fontsize=9.5, family="monospace")

ax.set_xlim(-0.1, max_force)
ax.set_ylim(-0.1, max_force)
ax.set_xlabel("Ground Truth Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
ax.set_ylabel("Predicted Model Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
ax.set_title("Cross-Geometry Generalization: 4 Unseen Indenters", fontweight="bold", pad=12)
ax.grid(True, linestyle=":", alpha=0.6)

# 6. Ultra-clean compact legend at lower right (4 items: Cylinder, Square, Sphere, Triangle)
ax.legend(
    loc="lower right",
    framealpha=0.96,
    edgecolor="#94a3b8",
    facecolor="#ffffff",
    fontsize=11.5,
    markerscale=1.4,
    borderpad=0.65,
    labelspacing=0.5,
    handletextpad=0.4
)

plt.tight_layout()

fig.savefig(OUTPUT_PNG, dpi=300)
print(f"✅ Generated plot saved to: {OUTPUT_PNG}")

artifact_png = os.path.join(ARTIFACT_DIR, "unseen_4_geometries_parity_plot.png")
if os.path.exists(ARTIFACT_DIR): fig.savefig(artifact_png, dpi=300)
print(f"✅ Copied to artifact dir: {artifact_png}")

plt.close(fig)

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=["Shape", "Label", "Timestamp", "LoadCell_N", "F_sensor_N", "Delta_P_hPa", "Area_mm2"])
    writer.writeheader()
    writer.writerows(collected_data)

print(f"✅ Companion CSV saved to: {OUTPUT_CSV}")

