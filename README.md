# Multi-Modal Pneumatic-Optical Soft Bubble Tactile Sensor

[![Paper](https://img.shields.io/badge/RoboSoft_2027-Paper_Companion-blue.svg)](https://robosoft2027.org)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official open-source companion codebase and benchmark dataset for:  
**"Multi-Modal Pneumatic-Optical Soft Bubble Tactile Sensor for Zero-Shot Cross-Geometry Force Estimation"**  
*Targeted for IEEE International Conference on Soft Robotics (RoboSoft 2027).*

---

## 📖 Overview

Soft bubble tactile sensors provide compliance and rich tactile feedback for robotic manipulation. However, traditional vision-based tactile sensors often struggle with generalizability across varied indenter geometries and surface profiles.

This repository implements a **multi-modal pneumatic-optical tactile sensor** that pairs internal global-shutter camera vision with high-precision 24-bit MEMS differential barometry. By unifying chamber pressure rise ($\Delta P$) and optical contact patch area ($A_c$), normal contact force ($F_z$) is accurately estimated via the analytical physical formulation:

$$F_z = \alpha \cdot \frac{\Delta P \cdot A_c}{10^4} + \gamma$$

Where:
* $\alpha = 4.4305$: Pneumatic-optical coupling factor (identified on a simple flat indenter)
* $\gamma = +0.3312\text{ N}$: Contact onset baseline offset
* Scaling factor: $10^4\text{ hPa}\cdot\text{mm}^2 \to \text{N}$

### 🎯 Key Highlights & Experimental Results
* **Zero-Shot Cross-Geometry Generalization:** Calibrated solely on a flat indenter and evaluated strictly on **4 previously unseen complex contact geometries** (Cylinder, Square, Sphere, and 3D Triangular Pyramidal Dome) without any recalibration:
  * **$R^2 = 97.38\%$**
  * **$\text{MAE} = 0.1600\text{ N}$** across 2,000 independent test points.
* **Massive Experimental Evaluation:** Tested across **24 distinct geometries** (120 continuous trials, 88,000+ synchronized time-series frames) spanning planar polygons, axisymmetric cylinders, annular cavities, organic contours (heart, star), and 3D domes.
* **Robust Hardware & Signal Processing:** Active status polling driver for Honeywell MPRLS with 2-sample jump guard, electrical bit-flip error masking, pressure-first contact gating ($\Delta P \ge 0.38\text{ hPa}$), and localized 3D membrane crater rendering.

---

## 🗂️ Repository Structure

```
.
├── README.md                   # Main documentation & reproduction guide
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
│
├── bubble_engine/              # Core Python sensing & hardware drivers
│   ├── camera.py               # Libcamera / IMX296 global shutter interface
│   ├── mprls.py                # Honeywell MPRLS 24-bit I2C pressure driver
│   ├── shadow_engine.py        # Vision-based contact shadow & convex hull area (Ac)
│   ├── tracker.py              # Optical flow dot motion tracking
│   ├── dome_3d.py              # 3D membrane reconstruction & crater deformation
│   ├── stepper.py              # Automated vertical carriage stepper driver
│   ├── loadcell.py             # Ground truth HX711 reference load cell reader
│   └── pneumatic_optical_force_estimator.py # Analytical force model
│
├── server/                     # Real-time WebSocket streaming & dashboard
│   ├── server.py               # Main asynchronous telemetry engine & CSV recorder
│   ├── run_desktop.sh          # Desktop launch script with auto-cleanup process trap
│   └── web/
│       └── index.html          # Three.js 3D web dashboard & live telemetry
│
├── experiments/                # Automated data collection & benchmark protocols
│   ├── auto_run_geometry_trials.py # Automated multi-step ladder loading protocol
│   ├── press_and_capture_raw.py    # Clean unannotated camera snapshot capture
│   └── lift_stepper.py             # Z-carriage calibration & homing tool
│
├── analysis/                   # Paper figure generation & statistical validation
│   ├── reproduce_all_paper_figures.py           # 1-Click master figure reproduction script
│   ├── generate_unseen_4_geometries_parity_plot.py # 4 Unseen shapes generalization plot
│   ├── generate_combined_parity_plot_5_shapes.py   # 5 Shapes multi-geometry benchmark plot
│   └── generate_all_individual_shape_plots.py      # Individual parity & step tracking plots
│
├── data/                       # Benchmark telemetry & summary datasets
│   ├── README.md               # Dataset variable definitions
│   ├── all_24_geometries_summary.csv        # Comprehensive 24-geometry metrics table
│   ├── unseen_4_geometries_sampled_data.csv # 2,000 sampled points from unseen shapes
│   └── sample_trials/                       # Sample 11-column CSV recordings
```

---

## ⚡ Quick Start

### 1. Installation

Clone this repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/ikadatym-sry/SoftBubble.git
cd SoftBubble

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce Paper Figures (1-Click)

To reproduce the publication figures from the experimental dataset:

```bash
cd analysis
python3 reproduce_all_paper_figures.py
```

Generated 300 DPI figures will be placed in `analysis/figures/`:
* `unseen_4_geometries_parity_plot.png`: Parity verification across 4 unseen geometries ($R^2 = 97.38\%$).
* `combined_5_geometries_parity_plot.png`: Multi-geometry parity benchmark ($R^2 = 97.82\%$).

### 3. Launch Live Telemetry Server (Hardware Setup)

On a Raspberry Pi 5 connected to the tactile sensor hardware:

```bash
python3 server/server.py
```
Open a browser and navigate to `http://localhost:8080` (or `http://<pi-ip>:8080`) to access the real-time Three.js 3D membrane deformation viewer, optical flow dot vector fields, and live force telemetry graphs.

---

## 🔬 Benchmark Results (24 Geometries)

| Geometry Category | Evaluated Shapes | Total Samples | $R^2$ Score | Mean Abs Error (MAE) |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline Calibration** | Flat Plate (50 × 50 mm) | 2,925 | 98.94% | 0.1081 N |
| **Cross-Geometry Unseen** | Cylinder, Square, Sphere, Triangle Dome | 18,702 | **97.38%** | **0.1600 N** |
| **Extended Benchmark** | Hexagons (13, 25, 35 mm), Squares (29–60 mm), Cylinders (35, 45 mm), Obtuse Polygon, Hollow Cylinder, Domes, Star, Heart | 66,955 | 83.20% | 0.4410 N |
| **GRAND TOTAL** | **24 Geometries (120 Trials)** | **88,582** | **84.29%** | **0.4258 N** |

---

## 🛠️ Hardware Requirements & Wiring

The sensor utilizes a streamlined **single-board embedded architecture** powered entirely by the Raspberry Pi 5 without external microcontrollers:
* **Host Computer:** Raspberry Pi 5 (4GB/8GB)
* **Vision Sensor:** Raspberry Pi Global Shutter Camera (Sony IMX296, 12mm lens) via CSI ribbon cable
* **Differential Barometer:** Honeywell MPRLS Digital Pressure Sensor (I2C address `0x18`, SDA=GPIO 2, SCL=GPIO 3)
* **Reference Rig:** HX711 24-bit ADC + 1 kg / 5 kg Load Cell (DT=GPIO 5, SCK=GPIO 6)
* **Z-Axis Carriage:** NEMA 17 Stepper + TMC2209 driver (STEP=GPIO 20, DIR=GPIO 21, EN=GPIO 16)

---

## 📜 Citation

If you use this codebase, sensor design, or dataset in your research, please cite:

```bibtex
@inproceedings{softbubble2027robosoft,
  title={Multi-Modal Pneumatic-Optical Soft Bubble Tactile Sensor for Zero-Shot Cross-Geometry Force Estimation},
  author={Pong and Advisor},
  booktitle={IEEE International Conference on Soft Robotics (RoboSoft)},
  year={2027}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
