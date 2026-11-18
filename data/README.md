# Tactile Benchmark Data & Sample Recordings

This directory contains sample trial CSVs and a verified summary table for the multi-modal pneumatic-optical soft bubble tactile sensor.

## Contents

- `all_24_geometries_summary.csv`: Summary metrics ($R^2$, MAE, RMSE, Max Error, pressed sample counts) evaluated over all 24 indenter geometries using baseline resting median $P_0$ and planar calibration coefficients ($\alpha = 4.4305$, $\gamma = +0.3312\text{ N}$).
- `sample_trials/`: Representative Trial-1 11-column CSV recordings (8 geometries) bundled so offline analysis scripts run out of the box on a fresh clone.

## 11-Column CSV Data Format

| Column | Type | Description |
| :--- | :---: | :--- |
| `Timestamp` | string | Elapsed experiment time (`MM:SS.ss`) |
| `Pressed?` | bool | Multi-modal contact gate (`True`/`False`) |
| `Pressure_Raw` | int | Honeywell MPRLS 24-bit raw ADC count |
| `Pressure_hPa` | float | Filtered chamber pressure $P$ [hPa] |
| `LoadCell_Raw` | int | HX711 24-bit raw ADC count |
| `LoadCell_N` | float | Reference normal force $F_{\mathrm{loadcell}}$ [N] |
| `Dimension_px` | string | Contact shadow bounding dimensions ($L \times W$) [px] |
| `Area_px2` | float | Contact shadow contour area ($A_c$) [px²] |
| `Dimension_mm` | string | Calibrated contact dimensions ($L \times W$) [mm] |
| `Area_mm2` | float | Calibrated contact area ($A_c$) [mm²] |
| `Displacement_px`| float | Max optical flow marker displacement ($\delta$) [px] |

## Using Your Own or Full Datasets

By default, scripts in `analysis/` automatically use `results/<shape>/*.csv` if present at the repository root (or passed via `--data-dir /path/to/results`), falling back seamlessly to `data/sample_trials/` when running with only the bundled sample files.
