#!/usr/bin/env python3
"""
Reproduce Benchmark Figures from Available Trial Data
-----------------------------------------------------
Runs all analysis plotting scripts using results/ if present,
or falls back automatically to the bundled data/sample_trials/ dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run all tactile benchmark figure scripts")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional path to full results/ dataset directory")
    args = parser.parse_args()

    scripts = [
        ("1. Cross-Geometry Generalization (4 Unseen Shapes)", "generate_unseen_4_geometries_parity_plot.py"),
        ("2. Multi-Geometry Benchmark (5 Selected Shapes)", "generate_combined_parity_plot_5_shapes.py"),
        ("3. Individual Geometry Plots (Parity + Step Tracking)", "generate_all_individual_shape_plots.py"),
    ]

    print("=" * 72)
    print("  GENERATING TACTILE SENSOR ANALYSIS FIGURES")
    print("=" * 72)

    failures = []
    for title, script_name in scripts:
        script_path = ANALYSIS_DIR / script_name
        cmd = [sys.executable, str(script_path)]
        if args.data_dir:
            cmd.extend(["--data-dir", args.data_dir])

        print(f"\n▶️ Running {title}...")
        res = subprocess.run(cmd, cwd=str(ANALYSIS_DIR))
        if res.returncode == 0:
            print(f"✅ Completed {script_name}")
        else:
            print(f"❌ Failed {script_name} (exit code {res.returncode})")
            failures.append(script_name)

    print("\n" + "=" * 72)
    if failures:
        print(f"❌ Completed with failures: {failures}")
        sys.exit(1)
    print("  ALL FIGURES GENERATED UNDER: analysis/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
