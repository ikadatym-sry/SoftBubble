# Hardware Specifications & Wiring Guide

This document describes the mechanical, electrical, and pneumatic specifications of the multi-modal pneumatic-optical soft bubble tactile sensor.

---

## 1. System Architecture

```
                      +-----------------------------------+
                      |   Raspberry Pi 5 (Host Computer)  |
                      +-----------------+-----------------+
                                        |
       +--------------------+-----------+-----------+--------------------+
       | CSI Ribbon Cable   | I2C Bus (0x18)        | GPIO / SPI         | SPI / Serial
       v                    v                       v                    v
+--------------+    +-----------------+     +---------------+    +---------------+
|  IMX296 GS   |    | Honeywell MPRLS |     | TMC2209 /     |    | HX711 Load    |
| 12mm Lens    |    | 24-bit Pressure |     | Stepper Motor |    | Cell Ground   |
| Camera (CSI) |    | Sensor (0-25psi)|     | (Z-carriage)  |    | Truth Rig     |
+--------------+    +-----------------+     +---------------+    +---------------+
       |                    |                       |
       +--------------------+-----------------------+
                            |
                   [ Sealed Silicone Dome ]
```

---

## 2. Bill of Materials (BOM)

| Component | Model / Spec | Purpose | Interface |
| :--- | :--- | :--- | :--- |
| **Main Computer** | Raspberry Pi 5 (4GB/8GB) | Real-time computer vision, pressure telemetry, control | — |
| **Vision Sensor** | Raspberry Pi Global Shutter (IMX296) | Internal marker tracking & contact shadow capture | CSI-2 |
| **Optical Lens** | 12 mm C/CS Mount Lens with adjustable aperture | Macro focus on silicone dome inner surface | Mechanical |
| **Pressure Sensor** | Honeywell MPRLS (0–25 psi digital) | 24-bit precision internal air pressure ($\Delta P$) | I2C (Address `0x18`) |
| **Vertical Actuator** | NEMA 17 Stepper Motor + Lead Screw | Automated vertical loading carriage (0.5–5.0 N) | GPIO (Step/Dir) |
| **Motor Driver** | TMC2209 or A4988 | Microstepping motor driver | GPIO |
| **Reference Sensor** | 1 kg / 5 kg Load Cell + HX711 ADC | Ground truth normal force measurement ($F_{\mathrm{ref}}$) | GPIO / Serial |
| **Internal Lighting** | High-CRI White SMD LEDs + Diffuser Ring | Uniform inner dome illumination | GPIO / PWM |
| **Membrane** | Smooth-On Dragon Skin 10 / Ecoflex 00-30 | Soft elastomeric bubble membrane with printed dots | Mechanical |

---

## 3. Wiring Connections

### Honeywell MPRLS Pressure Sensor -> Raspberry Pi 5
* `VIN`  -> Pin 1 (3.3V)
* `GND`  -> Pin 6 (GND)
* `SDA`  -> Pin 3 (GPIO 2 / I2C1 SDA)
* `SCL`  -> Pin 5 (GPIO 3 / I2C1 SCL)
* `RST`  -> Pin 11 (GPIO 17, optional hardware reset)

### Stepper Motor Driver -> Raspberry Pi 5
* `STEP` -> GPIO 20
* `DIR`  -> GPIO 21
* `EN`   -> GPIO 16
* `LIMIT_SWITCH_TOP` -> GPIO 19 (Pull-up)
* `LIMIT_SWITCH_BOTTOM` -> GPIO 26 (Pull-up)

### Reference Load Cell (HX711) -> Raspberry Pi 5
* `VCC`  -> Pin 2 (5.0V)
* `GND`  -> Pin 9 (GND)
* `DT`   -> GPIO 5
* `SCK`  -> GPIO 6

---

## 4. Firmware

* `firmware/arduino/arduino_firmware.ino`: Optional Arduino firmware if offloading stepper pulse generation.
* `firmware/esp8266/esp8266_mprls.ino`: Optional wireless telemetry node firmware.
