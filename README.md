# SoftBubble: Multi-Modal Pneumatic-Optical Soft Bubble Tactile Sensor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source hardware control, vision/pneumatic telemetry server, 3D deformation visualizer, automated indentation rig scripts, and offline analysis tools for a multi-modal soft bubble tactile sensor.

> **Note:** Paper citation, author list, license confirmation, and external dataset links are tracked in the [Open TODOs](#open-todos--notes) section below.

---

## Overview

The sensor pairs an internal Raspberry Pi camera observing printed membrane markers and contact shadows with a 24-bit MEMS differential barometer measuring sealed chamber pressure rise ($\Delta P$). By fusing chamber pressure rise ($\Delta P$) with segmented optical contact footprint area ($A_c$), normal contact force ($F_z$) is estimated as:

$$F_z = \alpha \cdot \frac{\Delta P \cdot A_c}{10^4} + \beta \cdot \delta_z + \gamma$$

### Key Capabilities
- **Dual-Stream Vision Engine (`bubble_engine/`):**
  - **Optical Flow Dot Tracking (`optical_flow_tracker.py`):** Tracks persistent printed marker IDs using Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) with baseline drift prevention.
  - **Contact Shadow Segmentation (`shadow_engine.py`):** Estimates contact footprint area ($A_c$, mm²) and bounding dimensions from internal membrane shading.
  - **3D Dome Reconstruction (`dome_3d.py`):** Computes and renders a real-time 3D dome depth deformation view using NumPy/OpenCV.
- **Robust Pneumatic Telemetry (`mprls.py`):** Native Linux I2C driver (`smbus2` / `/dev/i2c-1`) for the Honeywell MPRLS 24-bit sensor (`0x18`) with busy-bit active polling, integrity error masking, and a 2-sample outlier jump guard.
- **Asynchronous Web Console (`server/`):** Tornado HTTP + WebSocket/MJPEG server streaming 2D overlays, 3D depth views, force estimates, and 11-column CSV recordings to an HTML5 Canvas web console (`server/web/index.html`).
- **Automated Test Bench Scripts (`experiments/`):** Multi-trial ladder loading, overforce safety limits, Z-carriage jogging, and unannotated raw frame snapshot tools.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
│
├── bubble_engine/                    # Core vision, estimation & hardware drivers
│   ├── camera.py                     # Picamera2 / libcamera capture wrapper
│   ├── optical_flow_tracker.py       # Persistent-ID Lucas-Kanade dot tracker
│   ├── shadow_engine.py              # Contact shadow & footprint area (A_c)
│   ├── dome_3d.py                    # 3D membrane deformation & renderer
│   ├── mprls.py                      # Honeywell MPRLS I2C pressure driver
│   ├── led_controller.py             # WS2812B NeoPixel SPI controller
│   ├── loadcell.py                   # HX711 reference load cell reader (lgpio/RPi.GPIO)
│   ├── stepper.py                    # Z-axis stepper controller & safety limit watchdog
│   ├── pneumatic_optical_force_estimator.py  # Standalone piecewise force estimator
│   ├── analyzer.py                   # Auxiliary 3D geometry analyzer
│   ├── reconstruction.py             # Auxiliary depth-map reconstruction module
│   ├── tracker.py                    # Auxiliary baseline dot tracker
│   └── unblocked_baseline.jpg        # Sample resting membrane frame
│
├── server/                           # Real-time streaming & recording backend
│   ├── server.py                     # Tornado HTTP/WebSocket server & 11-col logger
│   ├── run_desktop.sh                # Chromium desktop launcher with cleanup trap
│   └── web/
│       └── index.html                # HTML5 Canvas telemetry & control UI
│
├── experiments/                      # Rig automation & calibration configs
│   ├── auto_run_geometry_trials.py   # Multi-trial step force benchmark (0.5–5.0 N)
│   ├── press_and_capture_raw.py      # Press to target force & save raw frame
│   ├── run_safe_stepper_test.py      # Press-and-hold safety validation test
│   ├── lift_stepper.py               # Lift Z-carriage and zero home reference
│   ├── force_calibration.json        # Live server force model coefficients
│   └── loadcell_calibration.json     # HX711 scale calibration & GPIO pinout
│
├── analysis/                         # Offline plotting & summary scripts
│   ├── benchmark_utils.py            # Shared CSV loader & parity plotting helpers
│   ├── reproduce_all_paper_figures.py
│   ├── generate_unseen_4_geometries_parity_plot.py
│   ├── generate_combined_parity_plot_5_shapes.py
│   └── generate_all_individual_shape_plots.py
│
└── data/                             # Sample trials & summary metrics
    ├── README.md
    ├── all_24_geometries_summary.csv
    └── sample_trials/                # Bundled Trial-1 sample CSVs (8 geometries)
```

---

## Quick Start

### 1. Installation

On a workstation (for offline analysis / development):

```bash
git clone https://github.com/ikadatym-sry/SoftBubble.git
cd SoftBubble

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On **Raspberry Pi 5** (for live sensor acquisition), install system hardware packages as well:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-lgpio python3-smbus2 python3-spidev
```

---

### 2. Offline Analysis (Works Out-of-the-Box)

Run all analysis plots using the bundled `data/sample_trials/` dataset (or pass `--data-dir /path/to/full/results`):

```bash
python3 analysis/reproduce_all_paper_figures.py
```

Generated 300 DPI PNG plots and sampled CSV outputs are saved under `analysis/figures/`.

To inspect offline dot detection on the included sample membrane image (`bubble_engine/unblocked_baseline.jpg`):

```bash
python3 bubble_engine/optical_flow_tracker.py
```

---

### 3. Live Server & Automated Experiments (Raspberry Pi 5)

Start the live telemetry and recording server:

```bash
python3 server/server.py
```

Then open `http://localhost:8080` in a browser (or run `./server/run_desktop.sh` on Raspberry Pi OS desktop for a managed Chromium app window).

Run an automated 5-trial force ladder benchmark while `server/server.py` is active:

```bash
python3 experiments/auto_run_geometry_trials.py --shape flat --trials 5
```

---

## Hardware Architecture & Pinout

This codebase runs on a **single Raspberry Pi 5** without an external Arduino/ESP8266 microcontroller:

| Subsystem | Device / Sensor | Interface & Pins (BCM) | Module |
| :--- | :--- | :--- | :--- |
| **Host Computer** | Raspberry Pi 5 | Linux / I2C / SPI / CSI / GPIO | — |
| **Internal Camera** | Pi Camera via `picamera2` *(tested with Camera Module 3 Wide / `imx708_wide`, `LensPosition=12.0`)* | CSI-2 ribbon | `bubble_engine/camera.py` |
| **Pressure Sensor** | Honeywell MPRLS (24-bit, 0–25 psi) | I2C Bus 1 (`0x18`), SDA=`GPIO 2`, SCL=`GPIO 3` | `bubble_engine/mprls.py` |
| **Reference Force** | Load Cell + HX711 24-bit ADC | `DT=GPIO 5`, `SCK=GPIO 6` *(configurable via JSON)* | `bubble_engine/loadcell.py` |
| **LED Illumination** | 4× WS2812B NeoPixels | SPI0 MOSI (`GPIO 10`, Physical Pin 19) | `bubble_engine/led_controller.py` |
| **Z-Axis Stepper** | NEMA Stepper + Step/Dir Driver | `PUL=GPIO 24`, `DIR=GPIO 23`, `ENA=GPIO 26`<br>`LIMIT_TOP=GPIO 17`, `LIMIT_BOTTOM=GPIO 27` | `bubble_engine/stepper.py` |

> **Important Note on Stepper Motion:** Real motor stepping delegates pulse generation to an external native `libgpiod` CLI binary configured via `PI_BUBBLE_STEPPER_RUNNER` (default `./native/build/stepper_runner`, **not bundled in this repository**). When the binary is absent, `BubbleStepper` automatically falls back to software simulation (`[Stepper-SIM]`) so hardware UI/tests remain safe to explore.

---

## Calibration Notes

Because pneumatic-optical coupling depends on each silicone membrane's thickness, chamber volume, camera distance, and lens scale, calibrate parameters for your own hardware build:

1. **Force Estimation Calibration (`experiments/force_calibration.json`):**
   - `server/server.py` searches `BUBBLE_FORCE_CALIBRATION`, `./force_calibration.json`, and `experiments/force_calibration.json`.
   - Note that `experiments/force_calibration.json` currently stores an example live runtime tuning (`alpha_pneumatic=5.85`, `gamma_offset_n=0.1`), whereas the offline `analysis/` scripts use the planar-fitted benchmark coefficients (`alpha=4.4305`, `gamma=0.3312` with `beta=0`) over a wider dataset (`data/all_24_geometries_summary.csv`). Update `force_calibration.json` to match your target calibration curve.
2. **Baseline Pressure Reference ($P_0$):**
   - `server/server.py` continuously tracks resting ambient pressure using an exponential moving average with anti-latchup recovery when untouched.
   - `analysis/` scripts estimate $P_0$ as the median of initial unpressed frames per shape.
3. **Load Cell Calibration (`experiments/loadcell_calibration.json`):**
   - Stores `counts_per_gram` and HX711 GPIO pins (`dt_pin`, `sck_pin`). Override path via `BUBBLE_LOADCELL_CALIBRATION`.

---

## Summary Dataset Reference

Full summary statistics computed over all 24 shapes (85,485 pressed frames across 120 continuous trials using `ALPHA_PAPER=4.4305`, `GAMMA_PAPER=0.3312`, median resting $P_0$) are provided in [data/all_24_geometries_summary.csv](data/all_24_geometries_summary.csv):
- **Calibration Baseline (`flat`):** 2,925 pressed frames across 5 trials ($R^2 = 98.94\%$, $\text{MAE} = 0.1081\text{ N}$).
- **4-Shape Unseen Subset (`cylinder`, `square`, `sphere`, `triangle_dome25-10`):**
  - Across all 18,702 pressed frames: $R^2 = 97.00\%$, $\text{MAE} = 0.1753\text{ N}$.
  - Across a balanced 2,000-point random sample (500/shape, `default_rng(42)`): $R^2 = 97.17\%$, $\text{MAE} = 0.1632\text{ N}$ (`analysis/generate_unseen_4_geometries_parity_plot.py` on the full dataset).
- **All 24 Geometries Combined:** 85,485 pressed frames ($R^2 = 85.73\%$, $\text{MAE} = 0.3940\text{ N}$).

---

## Open TODOs & Notes
- [ ] Confirm final paper title, author list, and BibTeX citation once accepted/published.
- [ ] Confirm open-source license choice with advisor (currently MIT placeholder in `LICENSE`).
- [ ] Optionally publish full 120-trial raw dataset archive externally (e.g., Zenodo/OSF) and link here.
- [ ] Optionally add the C/C++ `stepper_runner` source used on Raspberry Pi 5 if hardware actuation code is also released.
