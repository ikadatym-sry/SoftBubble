#!/usr/bin/env python3
"""
Generate Individual Benchmark Plots for Available Geometries
------------------------------------------------------------
For each discovered shape, generates under output/<shape>/:
1. parity_plot.png: Scatter & Linear Regression (F_sensor vs F_loadcell) with stats
2. tracking_plot.png: Dynamic time-series step tracking (0.5 to 5.0 N)
3. combined_benchmark_plot.png: 2-panel summary figure (Parity + Tracking)
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from benchmark_utils import (
    ALPHA_PAPER,
    DEFAULT_FIGURES_DIR,
    GAMMA_PAPER,
    SHAPE_TITLES,
    compute_metrics,
    discover_shapes,
    find_shape_csvs,
    load_shape_trials,
    resolve_data_root,
)

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
    "savefig.dpi": 300,
})


def format_shape_title(shape_key: str) -> str:
    return SHAPE_TITLES.get(shape_key, shape_key.replace("_", " ").title())


def process_shape(shape_name: str, data_root: Path, out_base: Path):
    csvs = find_shape_csvs(data_root, shape_name)
    if not csvs:
        return None

    title_name = format_shape_title(shape_name)
    print(f"\n▶️ Processing: {shape_name} ({len(csvs)} CSV(s))...")

    pressed_samples, trials_processed, _ = load_shape_trials(
        csvs, alpha=ALPHA_PAPER, gamma=GAMMA_PAPER, min_loadcell_n=0.0
    )
    n_pts = len(pressed_samples)
    if n_pts < 10:
        print(f"  ⚠️ Not enough pressed points for {shape_name} (N={n_pts}), skipping.")
        return None

    flc_all = np.array([p["flc"] for p in pressed_samples])
    fpred_all = np.array([p["fpred"] for p in pressed_samples])
    metrics = compute_metrics(flc_all, fpred_all)

    out_folder = out_base / shape_name
    out_folder.mkdir(parents=True, exist_ok=True)

    # 1. PARITY PLOT
    fig_p, ax_p = plt.subplots(figsize=(7.2, 6.8))
    fig_p.patch.set_facecolor("#ffffff")
    max_force = max(5.5, float(np.percentile(flc_all, 99.8)) * 1.1)
    x_band = np.linspace(0, max_force, 100)
    ax_p.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5, color="#cbd5e1", alpha=0.45, label="±0.50 N Acceptable Band", zorder=1)
    ax_p.fill_between(x_band, np.maximum(0, x_band - 0.25), x_band + 0.25, color="#94a3b8", alpha=0.35, label="±0.25 N Tight Error Band", zorder=1)
    ax_p.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--", linewidth=1.8, label="Ideal Agreement ($y = x$)", zorder=2)

    step_sub = max(1, n_pts // 1200)
    trial_word = "Trial" if len(csvs) == 1 else "Trials"
    ax_p.scatter(flc_all[::step_sub], fpred_all[::step_sub], color="#2563eb", s=32, alpha=0.60, edgecolors="none", label=f"Tactile Trials Data ({len(csvs)} {trial_word})", zorder=3)

    x_fit = np.linspace(0.4, 5.2, 50)
    ax_p.plot(x_fit, metrics["slope"] * x_fit + metrics["intercept"], color="#dc2626", linestyle="-", linewidth=2.0, label=f"Linear Fit ($y = {metrics['slope']:.2f}x {metrics['intercept']:+.2f}$)", zorder=4)

    stats_box = (
        f"Specimen: {shape_name}\n"
        f"Model: $F_z = \\alpha \\cdot (\\Delta P \\cdot A_c / 10^4) + \\gamma$\n"
        f"------------------------------------\n"
        f"Coefficient ($R^2$): {metrics['r2']*100:.2f}%\n"
        f"Mean Abs Error (MAE): {metrics['mae']:.4f} N\n"
        f"Root Mean Sq Error (RMSE): {metrics['rmse']:.4f} N\n"
        f"Max Abs Error: {metrics['max_err']:.4f} N\n"
        f"Samples ($N_{{pressed}}$): {n_pts:,} pts"
    )
    ax_p.text(
        0.05, 0.95, stats_box, transform=ax_p.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95, linewidth=1.2),
        fontsize=10, family="monospace",
    )
    ax_p.set_xlim(-0.1, max_force)
    ax_p.set_ylim(-0.1, max_force)
    ax_p.set_xlabel("Ground Truth Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
    ax_p.set_ylabel("Predicted Model Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
    ax_p.set_title(f"Force Parity Verification: {title_name}", fontweight="bold", pad=12)
    ax_p.grid(True, linestyle=":", alpha=0.6)
    ax_p.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()
    fig_p.savefig(out_folder / "parity_plot.png")
    plt.close(fig_p)

    # 2. DYNAMIC TRACKING PLOT
    rep_idx = max(range(len(trials_processed)), key=lambda i: int(np.sum(trials_processed[i]["flc"] > 0.4)))
    rep = trials_processed[rep_idx]

    fig_t, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.2]})
    fig_t.patch.set_facecolor("#ffffff")
    ax1.plot(rep["times"], rep["flc"], color="#0f172a", linestyle="--", linewidth=2.0, label="Reference Load Cell ($F_{\\mathrm{loadcell}}$)", zorder=3)
    ax1.plot(rep["times"], rep["fpred"], color="#ea580c", linestyle="-", linewidth=2.0, label="Pneumatic-Optical Model ($F_{\\mathrm{sensor}}$)", zorder=4)
    ax1.set_ylabel("Normal Force [N]", fontweight="bold")
    ax1.set_title(f"Dynamic Step Force Tracking (0.5 to 5.0 N): {title_name} [Trial {rep_idx+1}]", fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", framealpha=0.92)
    ax1.set_ylim(bottom=-0.1, top=max(5.6, float(np.max(rep["flc"])) * 1.15))

    for lvl in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        mask_lvl = np.abs(rep["flc"] - lvl) < 0.08
        if np.any(mask_lvl):
            t_mid = float(np.median(rep["times"][mask_lvl]))
            ax1.axhline(lvl, color="#94a3b8", linestyle=":", alpha=0.4)
            ax1.text(t_mid, lvl + 0.15, f"{lvl:.1f} N", color="#475569", fontsize=9, fontweight="bold", ha="center")

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(rep["times"], rep["dp"], color="#16a34a", linewidth=1.8, label="Chamber Pressure $\\Delta P$ [hPa]")
    l2 = ax2_twin.plot(rep["times"], rep["ac"], color="#9333ea", linewidth=1.8, linestyle="-.", label="Contact Footprint Area $A_c$ [$\\mathrm{mm}^2$]")
    ax2.set_xlabel("Time Elapsed [seconds]", fontweight="bold")
    ax2.set_ylabel("$\\Delta P$ [hPa]", color="#16a34a", fontweight="bold")
    ax2_twin.set_ylabel("Area $A_c$ [$\\mathrm{mm}^2$]", color="#9333ea", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    lines = l1 + l2
    ax2.legend(lines, [line.get_label() for line in lines], loc="upper left", framealpha=0.92)
    plt.tight_layout()
    fig_t.savefig(out_folder / "tracking_plot.png")
    plt.close(fig_t)

    # 3. COMBINED BENCHMARK PLOT
    fig_c, (c_ax1, c_ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2))
    fig_c.patch.set_facecolor("#ffffff")
    c_ax1.fill_between(x_band, np.maximum(0, x_band - 0.5), x_band + 0.5, color="#cbd5e1", alpha=0.45, label="±0.50 N Band")
    c_ax1.plot([0, max_force], [0, max_force], color="#0f172a", linestyle="--", linewidth=1.6, label="Ideal ($y = x$)")
    c_ax1.scatter(flc_all[::step_sub], fpred_all[::step_sub], color="#2563eb", s=28, alpha=0.60, edgecolors="none", label="Data Points")
    c_ax1.plot(x_fit, metrics["slope"] * x_fit + metrics["intercept"], color="#dc2626", linestyle="-", linewidth=1.8, label=f"Fit ($R^2={metrics['r2']*100:.1f}\\%$)")
    mini_box = f"$R^2 = {metrics['r2']*100:.2f}\\%$\nMAE = {metrics['mae']:.3f} N\nRMSE = {metrics['rmse']:.3f} N\n$N = {n_pts:,}$"
    c_ax1.text(0.05, 0.95, mini_box, transform=c_ax1.transAxes, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95), fontsize=10, family="monospace")
    c_ax1.set_xlim(-0.1, max_force)
    c_ax1.set_ylim(-0.1, max_force)
    c_ax1.set_xlabel("Reference Force $F_{\\mathrm{loadcell}}$ [N]", fontweight="bold")
    c_ax1.set_ylabel("Predicted Force $F_{\\mathrm{sensor}}$ [N]", fontweight="bold")
    c_ax1.set_title("(a) Force Parity Agreement", fontweight="bold")
    c_ax1.grid(True, linestyle=":", alpha=0.6)
    c_ax1.legend(loc="lower right", fontsize=9)

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
    fig_c.savefig(out_folder / "combined_benchmark_plot.png")
    plt.close(fig_c)

    print(f"  ✅ Saved 3 plots into {out_folder}")
    return {"shape": shape_name, "n_pts": n_pts, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Generate individual parity & tracking plots")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to results directory (defaults to results/ or sample_trials/)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_FIGURES_DIR / "individual"), help="Output directory")
    args = parser.parse_args()

    data_root, kind = resolve_data_root(args.data_dir)
    out_base = Path(args.out_dir)
    shapes = discover_shapes(data_root)
    print(f"Discovered {len(shapes)} shape(s) in {data_root} ({kind} dataset)")

    summaries = [res for s in shapes if (res := process_shape(s, data_root, out_base))]
    print(f"\n🎉 Generated individual plots for {len(summaries)} geometry folder(s) under {out_base}")


if __name__ == "__main__":
    main()
