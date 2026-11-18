#!/usr/bin/env python3
"""
Multi-Geometry Force Parity Plot (5 Selected Geometries)
--------------------------------------------------------
Compares Flat, Cylinder, Square, Sphere, and Triangular Dome in one figure.
"""

import argparse
from pathlib import Path
import sys

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from benchmark_utils import DEFAULT_FIGURES_DIR, generate_multi_shape_parity_plot

TARGET_SHAPES = [
    {"key": "flat", "legend_label": "Flat", "color": "#2563eb", "marker": "o", "zorder": 4},
    {"key": "cylinder", "legend_label": "Cylinder", "color": "#d97706", "marker": "D", "zorder": 4},
    {"key": "square", "legend_label": "Square", "color": "#059669", "marker": "s", "zorder": 4},
    {"key": "sphere", "legend_label": "Sphere", "color": "#7c3aed", "marker": "^", "zorder": 4},
    {"key": "triangle_dome25-10", "legend_label": "Triangle", "color": "#0891b2", "marker": "v", "zorder": 4},
]


def main():
    parser = argparse.ArgumentParser(description="Generate 5 geometries combined parity plot")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to results directory (defaults to results/ or sample_trials/)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Output figures directory")
    parser.add_argument("--samples", type=int, default=500, help="Samples per geometry (default: 500)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    generate_multi_shape_parity_plot(
        target_shapes=TARGET_SHAPES,
        title="Force Parity Verification: Multi-Geometry Benchmark (5 Indenters)",
        header_label="Specimens: 5 Indenters (up to 500 pts each)",
        model_line="Model: $F_z = \\alpha \\cdot (\\Delta P \\cdot A_c / 10^4) + \\gamma$",
        r2_label="Combined ($R^2$)",
        sample_count_label="Total Samples ($N_{total}$)",
        output_png=out_dir / "combined_5_geometries_parity_plot.png",
        output_csv=out_dir / "combined_5_geometries_sampled_data.csv",
        data_dir=args.data_dir,
        samples_per_shape=args.samples,
    )


if __name__ == "__main__":
    main()
