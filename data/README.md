# Tactile Benchmark Dataset

This directory contains experimental telemetry datasets, summary tables, and sampled verification trials for the multi-modal pneumatic-optical soft bubble tactile sensor.

## Files & Directories

- `all_24_geometries_summary.csv`: Complete statistical summary table across all 24 evaluated indenter shapes (120 continuous trials, 88,000+ contact samples).
- `unseen_4_geometries_sampled_data.csv`: Uniformly sampled dataset (2,000 points, 500 per shape) for the 4 previously unseen test geometries (Cylinder, Square, Sphere, Triangle Dome 25-10).
- `combined_5_geometries_sampled_data.csv`: Uniformly sampled dataset (2,500 points, 500 per shape) across 5 representative geometries.
- `sample_trials/`: Raw 11-column CSV recordings for representative shapes, ready for immediate playback and verification without downloading the entire 111 MB dataset.

## 11-Column CSV Data Format

Every recording frame contains 11 synchronized multi-modal variables:

| Column | Type | Description |
| :--- | :---: | :--- |
| `Timestamp` | string | Elapsed experiment time (`MM:SS.ss`) |
| `Pressed?` | bool | Multi-modal physical contact gate (`True`/`False`) |
| `Pressure_Raw` | int | Honeywell MPRLS 24-bit raw ADC reading |
| `Pressure_hPa` | float | Calibrated internal chamber pressure ($P$) [hPa] |
| `LoadCell_Raw` | int | HX711 24-bit raw ADC reading |
| `LoadCell_N` | float | Ground truth reference normal force ($F_{\mathrm{ref}}$) [N] |
| `Dimension_px` | string | Segmented contact footprint bounding box [px] |
| `Area_px2` | float | Convex hull contact footprint area ($A_c$) [px²] |
| `Dimension_mm` | string | Calibrated contact footprint dimensions ($L \times W$) [mm] |
| `Area_mm2` | float | Calibrated contact footprint area ($A_c$) [mm²] |
| `Displacement_px`| float | Membrane center deformation displacement ($\delta_z$) [px] |

## Full Dataset Availability

The full dataset containing all 120 trials across 24 geometries (~111 MB) is consolidated in `results/all.xlsx` and available in the accompanying paper archive.
