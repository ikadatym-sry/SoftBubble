#!/usr/bin/env python3
"""
================================================================================
REPRODUCE ALL FIGURES FOR IEEE ROBOSOFT 2027 PAPER
================================================================================
Generates all publication figures:
1. Cross-Geometry Generalization Parity Plot (4 Unseen Geometries, N=2,000)
2. Multi-Geometry Benchmark Parity Plot (5 Indenters, N=2,500)
3. Individual Geometry Benchmark Plots (Parity + Dynamic Step Ladder)
================================================================================
"""

import os
import sys
import subprocess
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent

print("=" * 78)
print("  REPRODUCING ALL PUBLICATION FIGURES FOR IEEE ROBOSOFT 2027")
print("=" * 78)

scripts = [
    ("1. Cross-Geometry Generalization (4 Unseen Shapes)", "generate_unseen_4_geometries_parity_plot.py"),
    ("2. Multi-Geometry Benchmark (5 Selected Shapes)", "generate_combined_parity_plot_5_shapes.py"),
    ("3. Individual Geometry Benchmark Figures (All Geometries)", "generate_all_individual_shape_plots.py"),
]

for title, script_name in scripts:
    script_path = ANALYSIS_DIR / script_name
    if not script_path.exists():
        print(f"⚠️ Script not found: {script_name}, skipping.")
        continue
    print(f"\n▶️ Running {title}...")
    res = subprocess.run([sys.executable, str(script_path)], cwd=str(ANALYSIS_DIR))
    if res.returncode == 0:
        print(f"✅ Successfully finished {script_name}")
    else:
        print(f"❌ Error occurred while running {script_name}")

print("\n" + "=" * 78)
print("  ALL FIGURES GENERATED IN: analysis/figures/ or results folders")
print("=" * 78)
