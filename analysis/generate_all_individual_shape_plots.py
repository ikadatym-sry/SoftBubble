#!/usr/bin/env python3
"""
Generate Publication-Quality Individual Benchmark Plots for Every Tested Geometry
---------------------------------------------------------------------------------
For each shape folder in results/, generates:
1. parity_plot.png: Scatter & Linear Regression (F_sensor vs F_loadcell) with stats
2. tracking_plot.png: Multi-trial dynamic time-series step tracking (0.5 to 5.0 N)
3. combined_benchmark_plot.png: 2-panel publication figure (Parity + Tracking)
"""

import os
import glob
import csv
import math
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

ALPHA = 4.4305
GAMMA = 0.3312

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 300
})


def format_shape_title(shape_key: str) -> str:
    titles = {
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
        "hexagon": "Hexagon Baseline (Side 22 mm)"
    }
    return titles.get(shape_key, shape_key.replace("_", " ").title())


def process_shape(shape_name: str):
    folder = os.path.join(RESULTS_BASE, shape_name)
    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csvs:
        return None

    title_name = format_shape_title(shape_name)
    print(f"\n▶️ Processing: {shape_name} ({len(csvs)} CSVs)...")

    # Determine baseline pressure from early resting samples across CSVs
    resting_pressures = []
    trials_raw_data = []

    for fpath in csvs:
        with open(fpath, mode="r") as fp:
            rows = list(csv.DictReader(fp))
        
        # sample resting pressures
        unpressed = [float(r["Pressure_hPa"]) for r in rows[:40] if r["Pressed?"].strip().lower() in ["false", "0", "0.0"]]
        if unpressed:
            resting_pressures.extend(unpressed)
        
        trials_raw_data.append(rows)

    base_p = float(np.median(resting_pressures)) if resting_pressures else 1045.0

    all_flc = []
    all_fpred = []
    all_dp = []
    all_ac = []
    
    trials_processed = []

    for trial_idx, rows in enumerate(trials_raw_data):
        t_arr, flc_arr, fpred_arr, dp_arr, ac_arr = [], [], [], [], []
        for idx, r in enumerate(rows):
            flc = float(r["LoadCell_N"])
            p_val = float(r["Pressure_hPa"])
            dp = max(0.0, p_val - base_p)
            ac = float(r["Area_mm2"])
            pressed = r["Pressed?"].strip().lower() in ["true", "1", "1.0"]
            fpred = (ALPHA * (dp * ac) / 10000.0 + GAMMA) if pressed else 0.0

            ts_str = r["Timestamp"].replace("Time: ", "").strip()
            parts = ts_str.split(":")
            sec = float(parts[0]) * 60.0 + float(parts[1]) if len(parts) == 2 else idx * 0.08

            t_arr.append(sec)
            flc_arr.append(flc)
            fpred_arr.append(fpred)
            dp_arr.append(dp)
            ac_arr.append(ac)

            if pressed:
                all_flc.append(flc)
                all_fpred.append(fpred)
                all_dp.append(dp)
                all_ac.append(ac)

        trials_processed.append({
            "times": np.array(t_arr),
            "flc": np.array(flc_arr),
            "fpred": np.array(fpred_arr),
            "dp": np.array(dp_arr),
            "ac": np.array(ac_arr)
        })

    flc_all = np.array(all_flc)
    fpred_all = np.array(all_fpred)
    n_pts = len(flc_all)

    if n_pts < 10:
        print(f"  ⚠️ Not enough pressed points for {shape_name} (N={n_pts}), skipping.")
        return None

    # Calculate statistics
    mae = float(np.mean(np.abs(fpred_all - flc_all)))
    rmse = float(np.sqrt(np.mean((fpred_all - flc_all)**2)))
    max_err = float(np.max(np.abs(fpred_all - flc_all)))
    ss_tot = float(np.sum((flc_all - np.mean(flc_all))**2))
    ss_res = float(np.sum((flc_all - fpred_all)**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Best-fit regression line
    if np.std(flc_all) > 1e-5:
        slope, intercept = np.polyfit(flc_all, fpred_all, 1)
    else:
        slope, intercept = 1.0, 0.0

    # --------------------------------------------------------------------------
    # 1. PARITY PLOT (parity_plot.png)
    # --------------------------------------------------------------------------
    fig_p, ax_p = plt.subplots(figsize=(7.2, 6.8))
    fig_p.patch.set_facecolor("#ffffff")

    max_force = max(5.5, float(np.percentile(flc_all, 99.8)) * 1.1)

    # Shaded error bands
    x_band = np.linspace(0, max_force, 100)
    ax_p.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5, color="#cbd5e1", alpha=0.45, label="±0.50 N Acceptable Band", zorder=1)
    ax_p.fill_between(x_band, np.maximum(0, x_band - 0.25), x_band + 0.25, color="#94a3b8", alpha=0.35, label="±0.25 N Tight Error Band", zorder=1)

    # Ideal line
    ax_p.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--", linewidth=1.8, label="Ideal Agreement ($y = x$)", zorder=2)

    # Scatter points (subsampled if very large for crisp rendering)
    step_sub = max(1, n_pts // 1200)
    ax_p.scatter(flc_all[::step_sub], fpred_all[::step_sub], color="#2563eb", s=32, alpha=0.60, edgecolors="none", label=f"Tactile Trials Data (5 Trials)", zorder=3)

    # Linear fit line
    x_fit = np.linspace(0.4, 5.2, 50)
    ax_p.plot(x_fit, slope * x_fit + intercept, color="#dc2626", linestyle="-", linewidth=2.0, label=f"Linear Fit ($y = {slope:.2f}x {intercept:+.2f}$)", zorder=4)

    # Annotation box
    stats_box = (
        f"Specimen: {shape_name}\n"
        f"Model: $F_z = \\alpha \\cdot (\\Delta P \\cdot A_c / 10^4) + \\gamma$\n"
        f"------------------------------------\n"
        f"Coefficient ($R^2$): {r2*100:.2f}%\n"
        f"Mean Abs Error (MAE): {mae:.4f} N\n"
        f"Root Mean Sq Error (RMSE): {rmse:.4f} N\n"
        f"Max Abs Error: {max_err:.4f} N\n"
        f"Samples ($N_{{pressed}}$): {n_pts:,} pts"
    )
    ax_p.text(0.05, 0.95, stats_box, transform=ax_p.transAxes, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95, linewidth=1.2),
              fontsize=10, family="monospace")

    ax_p.set_xlim(-0.1, max_force)
    ax_p.set_ylim(-0.1, max_force)
    ax_p.set_xlabel("Ground Truth Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
    ax_p.set_ylabel("Predicted Model Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
    ax_p.set_title(f"Force Parity Verification: {title_name}", fontweight="bold", pad=12)
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()

    parity_out = os.path.join(folder, "parity_plot.png")
    fig_p.savefig(parity_out)
    plt.close(fig_p)

    # --------------------------------------------------------------------------
    # 2. DYNAMIC TRACKING PLOT (tracking_plot.png)
    # --------------------------------------------------------------------------
    # Select representative trial (Trial 1 or 2 with most complete step cycle)
    rep_idx = 0
    max_steps = 0
    for idx, tr in enumerate(trials_processed):
        cnt = np.sum(tr["flc"] > 0.4)
        if cnt > max_steps:
            max_steps = cnt
            rep_idx = idx

    rep = trials_processed[rep_idx]
    fig_t, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
    fig_t.patch.set_facecolor("#ffffff")

    # Top: Force Step Ladder Tracking
    ax1.plot(rep["times"], rep["flc"], color="#0f172a", linestyle="--", linewidth=2.0, label="Reference Load Cell ($F_{\\mathrm{loadcell}}$)", zorder=3)
    ax1.plot(rep["times"], rep["fpred"], color="#ea580c", linestyle="-", linewidth=2.0, label="Pneumatic-Optical Model ($F_{\\mathrm{sensor}}$)", zorder=4)
    ax1.set_ylabel("Normal Force [N]", fontweight="bold")
    ax1.set_title(f"Dynamic Step Force Tracking (0.5 to 5.0 N): {title_name} [Trial {rep_idx+1}]", fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.92)
    ax1.set_ylim(bottom=-0.1, top=max(5.6, float(np.max(rep["flc"])) * 1.15))

    # Annotate target step holds
    target_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    for lvl in target_levels:
        mask_lvl = np.abs(rep["flc"] - lvl) < 0.08
        if np.any(mask_lvl):
            t_mid = float(np.median(rep["times"][mask_lvl]))
            ax1.axhline(lvl, color="#94a3b8", linestyle=":", alpha=0.4)
            ax1.text(t_mid, lvl + 0.15, f"{lvl:.1f} N", color="#475569", fontsize=9, fontweight="bold", ha="center")

    # Bottom: Chamber Pressure dP & Contact Area Ac
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(rep["times"], rep["dp"], color="#16a34a", linewidth=1.8, label="Chamber Pressure $\\Delta P$ [hPa]")
    l2 = ax2_twin.plot(rep["times"], rep["ac"], color="#9333ea", linewidth=1.8, linestyle="-.", label="Contact Footprint Area $A_c$ [$\\mathrm{mm}^2$]")

    ax2.set_xlabel("Time Elapsed [seconds]", fontweight="bold")
    ax2.set_ylabel("$\\Delta P$ [hPa]", color="#16a34a", fontweight="bold")
    ax2_twin.set_ylabel("Area $A_c$ [$\\mathrm{mm}^2$]", color="#9333ea", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Combined legend for bottom panel
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left", framealpha=0.92)

    plt.tight_layout()
    tracking_out = os.path.join(folder, "tracking_plot.png")
    fig_t.savefig(tracking_out)
    plt.close(fig_t)

    # --------------------------------------------------------------------------
    # 3. COMBINED BENCHMARK ANALYSIS (combined_benchmark_plot.png)
    # --------------------------------------------------------------------------
    fig_c, (c_ax1, c_ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    fig_c.patch.set_facecolor("#ffffff")

    # Left: Parity
    c_ax1.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5, color="#cbd5e1", alpha=0.45, label="±0.50 N Band")
    c_ax1.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--", linewidth=1.6, label="Ideal ($y = x$)")
    c_ax1.scatter(flc_all[::step_sub], fpred_all[::step_sub], color="#2563eb", s=28, alpha=0.60, edgecolors="none", label="Data Points")
    c_ax1.plot(x_fit, slope * x_fit + intercept, color="#dc2626", linestyle="-", linewidth=1.8, label=f"Fit ($R^2={r2*100:.1f}\\%$)")
    
    mini_box = f"$R^2 = {r2*100:.2f}\\%$\nMAE = {mae:.3f} N\nRMSE = {rmse:.3f} N\n$N = {n_pts:,}$"
    c_ax1.text(0.05, 0.95, mini_box, transform=c_ax1.transAxes, verticalalignment="top",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95),
               fontsize=10, family="monospace")

    c_ax1.set_xlim(-0.1, max_force)
    c_ax1.set_ylim(-0.1, max_force)
    c_ax1.set_xlabel("Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
    c_ax1.set_ylabel("Predicted Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
    c_ax1.set_title("(a) Force Parity Agreement", fontweight="bold")
    c_ax1.grid(True, linestyle=":", alpha=0.6)
    c_ax1.legend(loc="lower right", fontsize=9)

    # Right: Step Tracking
    c_ax2.plot(rep["times"], rep["flc"], color="#0f172a", linestyle="--", linewidth=1.8, label="$F_{\\mathrm{loadcell}}$ (True)")
    c_ax2.plot(rep["times"], rep["fpred"], color="#ea580c", linestyle="-", linewidth=1.8, label="$F_{\\mathrm{sensor}}$ (Model)")
    c_ax2.set_ylabel("Normal Force [N]", fontweight="bold")
    c_ax2.set_xlabel("Time Elapsed [seconds]", fontweight="bold")
    c_ax2.set_title(f"(b) Dynamic Force Step Response [Trial {rep_idx+1}]", fontweight="bold")
    c_ax2.grid(True, linestyle=":", alpha=0.6)
    c_ax2.legend(loc="upper left", fontsize=9)
    c_ax2.set_ylim(bottom=-0.1, top=max(5.6, float(np.max(rep["flc"])) * 1.15))

    fig_c.suptitle(f"Tactile Benchmark Characterization: {title_name}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    comb_out = os.path.join(folder, "combined_benchmark_plot.png")
    fig_c.savefig(comb_out)
    plt.close(fig_c)

    print(f"  ✅ Saved parity_plot.png")
    print(f"  ✅ Saved tracking_plot.png")
    print(f"  ✅ Saved combined_benchmark_plot.png")

    return {
        "shape": shape_name,
        "title": title_name,
        "n_pts": n_pts,
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "r2": r2
    }


def main():
    folders = [
        "triangle63.5-67",
        "hexagon25",
        "square60.5",
        "square34",
        "square45",
        "hexagon35",
        "square40",
        "hexagon13",
        "obtuse50-40",
        "cylinder35",
        "cylinder45",
        "square30",
        "square29",
        "triangle_dome25-10",
        "hollow_cylinder25",
        "circular_dome36.5-15",
        "square_dome24-12",
        "star27x29",
        "heart28x30.5",
        "flat",
        "cylinder",
        "square",
        "sphere",
        "hexagon"
    ]

    summaries = []
    for f in folders:
        res = process_shape(f)
        if res:
            summaries.append(res)

    print("\n" + "=" * 80)
    print("🎉 ALL INDIVIDUAL GEOMETRY PLOTS GENERATED SUCCESSFULLY!")
    print(f"Processed {len(summaries)} shapes across all benchmark folders.")
    print("=" * 80)


if __name__ == "__main__":
    main()
