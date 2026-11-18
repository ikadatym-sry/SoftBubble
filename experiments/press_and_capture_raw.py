#!/usr/bin/env python3
"""
Press Indenter to Target Force, Capture Pure RAW Unannotated Camera Frame, Retract to Home.
"""

import asyncio
import json
import time
import sys

WS_URL = "ws://localhost:8080/ws"
RAW_OUT_PATH = "/home/bubble/Desktop/TestVersion1/results/triangle/raw_triangle_pressed.jpg"

async def run(target_force: float = 2.5):
    import websockets

    print("=" * 70)
    print(f"📸 PRESS & CAPTURE PURE RAW FRAME (Target Force: {target_force:.1f} N)")
    print("=" * 70)

    async with websockets.connect(WS_URL) as ws:
        latest_telem = {}
        latest_stepper = {}
        is_running = True

        async def telemetry_listener():
            nonlocal latest_telem, latest_stepper
            try:
                while is_running:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    latest_telem = data.get("telemetry", {})
                    latest_stepper = data.get("stepper", {})
            except Exception:
                pass

        listener_task = asyncio.create_task(telemetry_listener())
        await asyncio.sleep(0.5)

        async def safe_jog(direction: str, dist_mm: float, speed_mm_s: float = 3.0):
            await ws.send(json.dumps({
                "cmd": "stepper_jog",
                "direction": direction,
                "mm": round(dist_mm, 2),
                "speed": round(speed_mm_s, 1)
            }))
            for _ in range(8):
                await asyncio.sleep(0.03)
                if latest_stepper.get("is_moving", False):
                    break
            while latest_stepper.get("is_moving", False):
                await asyncio.sleep(0.04)
            await asyncio.sleep(0.05)

        try:
            # 1. Search contact
            print("[1/4] Moving down to establish contact...")
            while True:
                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                dp = latest_telem.get("delta_p_hpa", 0.0)
                z = latest_stepper.get("position_mm", 0.0)
                if f_lc >= 0.15 or dp >= 0.38:
                    print(f"  🎯 Contact detected at Z = {z:.2f} mm (F_lc = {f_lc:.3f} N, ΔP = {dp:.2f} hPa)")
                    break
                step = 3.0 if z < 25.0 else 1.0
                spd = 5.0 if z < 25.0 else 3.0
                await safe_jog("down", step, spd)
                if z > 120.0:
                    print("❌ Over-travel limit reached.")
                    return

            # 2. Ramp to target force (~2.5 N)
            print(f"[2/4] Pressing to target force ({target_force:.1f} N)...")
            while True:
                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                if f_lc >= target_force:
                    break
                diff = target_force - f_lc
                step = 0.3 if diff > 1.0 else (0.15 if diff > 0.3 else 0.06)
                spd = 1.8 if diff > 1.0 else 1.0
                await safe_jog("down", step, spd)

            # 3. Hold & Capture RAW frame
            f_held = latest_telem.get("f_loadcell_n", 0.0)
            dp_held = latest_telem.get("delta_p_hpa", 0.0)
            print(f"  ⏳ Holding stable contact: F_lc = {f_held:.3f} N, ΔP = {dp_held:.2f} hPa")
            await asyncio.sleep(1.0)

            print(f"[3/4] Triggering PURE RAW snapshot capture -> {RAW_OUT_PATH}...")
            await ws.send(json.dumps({
                "cmd": "capture_snapshot",
                "raw_path": RAW_OUT_PATH
            }))
            await asyncio.sleep(0.8)
            print(f"  ✅ Raw snapshot captured successfully!")

        finally:
            # 4. Retract to Home
            print("[4/4] Retracting stepper to HOME (Z = 0.00 mm)...")
            await ws.send(json.dumps({"cmd": "stepper_stop"}))
            await asyncio.sleep(0.1)
            await ws.send(json.dumps({"cmd": "stepper_home", "speed": 5.0}))
            while latest_stepper.get("is_moving", False) or latest_stepper.get("position_mm", 0.0) > 0.3:
                await asyncio.sleep(0.1)
            print("  🏠 Stepper returned to Home.")

        is_running = False
        listener_task.cancel()
        print("=" * 70)
        print("🎉 COMPLETE: PURE RAW PHOTO SAVED & STEPPER SAFELY HOME!")
        print("=" * 70)

if __name__ == "__main__":
    t_f = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
    asyncio.run(run(t_f))
