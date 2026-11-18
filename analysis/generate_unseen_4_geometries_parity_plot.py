#!/usr/bin/env python3
"""
Cross-Geometry Generalization Parity Plot (4 Previously Unseen Geometries)
--------------------------------------------------------------------------
Evaluates Cylinder, Square, Sphere, and Triangular Dome (excluding Flat baseline).
"""

import argparse
from pathlib import Path
import sys

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from benchmark_utils import DEFAULT_FIGURES_DIR, generate_multi_shape_parity_plot

TARGET_SHAPES = [
    {"key": "cylinder", "legend_label": "Cylinder", "color": "#d97706", "marker": "D", "zorder": 4},
    {"key": "square", "legend_label": "Square", "color": "#059669", "marker": "s", "zorder": 4},
    {"key": "sphere", "legend_label": "Sphere", "color": "#7c3aed", "marker": "^", "zorder": 4},
    {"key": "triangle_dome25-10", "legend_label": "Triangle", "color": "#0891b2", "marker": "v", "zorder": 4},
]


def main():
    parser = argparse.ArgumentParser(description="Generate 4 unseen geometries parity plot")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to results directory (defaults to results/ or sample_trials/)")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_FIGURES_DIR), help="Output figures directory")
    parser.add_argument("--samples", type=int, default=500, help="Samples per geometry (default: 500)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    generate_multi_shape_parity_plot(
        target_shapes=TARGET_SHAPES,
        title="Cross-Geometry Generalization: 4 Unseen Indenters",
        header_label="Evaluation: 4 Unseen Geometries",
        model_line="Model Locked: Flat Baseline",
        r2_label="Generalization ($R^2$)",
        sample_count_label="Total Samples ($N_{test}$)",
        output_png=out_dir / "unseen_4_geometries_parity_plot.png",
        output_csv=out_dir / "unseen_4_geometries_sampled_data.csv",
        data_dir=args.data_dir,
        samples_per_shape=args.samples,
    )


if __name__ == "__main__":
    main()
