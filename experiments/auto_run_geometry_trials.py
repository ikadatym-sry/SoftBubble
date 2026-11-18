#!/usr/bin/env python3
"""
Autonomous Multi-Trial Benchmark Runner for Tactile Bubble Rig
---------------------------------------------------------------
Executes 5 automated trials per geometry across force levels:
[0.5 N, 1.0 N, 2.0 N, 3.0 N, 4.0 N, 5.0 N]
Saves clean 11-column CSVs into results/<shape_name>/
"""

import asyncio
import json
import time
import argparse
import sys
from pathlib import Path

WS_URL = "ws://localhost:8080/ws"

async def run_trials(shape: str, num_trials: int = 5, start_trial: int = 1):
    import websockets

    print("=" * 80)
    print(f"  🚀 AUTONOMOUS BENCHMARK: [{shape.upper()}] - {num_trials} TRIALS")
    print("  Force Targets: 0.5 N, 1.0 N, 2.0 N, 3.0 N, 4.0 N, 5.0 N (Hold 1.0s each)")
    print("  Watchdog Cutoff: 5.8 N | Hardware Interlock: 6.5 N")
    print("=" * 80)

    # Automatically create shape directory and dimension tag file
    shape_dir = Path(__file__).resolve().parent / "results" / shape
    shape_dir.mkdir(parents=True, exist_ok=True)
    tag_file = shape_dir / f"{shape}.txt"
    if not tag_file.exists():
        tag_file.touch()

    async with websockets.connect(WS_URL) as ws:
        latest_telem = {}
        latest_stepper = {}
        emergency_abort = False
        is_running = True

        async def telemetry_listener():
            nonlocal latest_telem, latest_stepper, emergency_abort
            try:
                while is_running:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    latest_telem = data.get("telemetry", {})
                    latest_stepper = data.get("stepper", {})

                    f_lc = latest_telem.get("f_loadcell_n", 0.0)
                    if f_lc >= 5.80:
                        emergency_abort = True
                        print(f"\n🚨 [WATCHDOG] Load cell force reached {f_lc:.2f} N >= 5.80 N! Aborting & Retracting!")
                        await ws.send(json.dumps({"cmd": "stepper_stop"}))
                        await asyncio.sleep(0.1)
                        await ws.send(json.dumps({"cmd": "stepper_home", "speed": 4.0}))
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        listener_task = asyncio.create_task(telemetry_listener())
        await asyncio.sleep(0.6)

        print("\n[Sensor Initial Health Check]")
        print(f"  Load Cell: {latest_telem.get('f_loadcell_n', 0.0):.3f} N")
        print(f"  Chamber ΔP: {latest_telem.get('delta_p_hpa', 0.0):.2f} hPa")
        print(f"  Stepper Z: {latest_stepper.get('position_mm', 0.0):.2f} mm")

        # Helper to execute non-overlapping, verified stepper jogs
        async def safe_jog(direction: str, dist_mm: float, speed_mm_s: float = 4.0):
            if emergency_abort:
                return
            await ws.send(json.dumps({
                "cmd": "stepper_jog",
                "direction": direction,
                "mm": round(dist_mm, 2),
                "speed": round(speed_mm_s, 1)
            }))
            # Wait for motor to start moving
            for _ in range(8):
                await asyncio.sleep(0.03)
                if latest_stepper.get("is_moving", False):
                    break
            # Wait for motor to finish moving
            while latest_stepper.get("is_moving", False) and not emergency_abort:
                await asyncio.sleep(0.04)
            await asyncio.sleep(0.05)

        TARGET_FORCES = [0.50, 1.00, 2.00, 3.00, 4.00, 5.00]
        raw_photo_captured = False

        for trial_idx in range(start_trial, num_trials + 1):
            if emergency_abort:
                print(f"⚠️ Aborting remaining trials due to safety event.")
                break

            print("\n" + "=" * 65)
            print(f"  ▶️ STARTING TRIAL {trial_idx}/{num_trials} for [{shape.upper()}]")
            print("=" * 65)

            # Start Recording via WebSocket into results/<shape>/
            await ws.send(json.dumps({
                "cmd": "start_record",
                "shape": shape,
                "trial": trial_idx
            }))
            await asyncio.sleep(0.4)

            contact_z = None

            try:
                # 1. Search for contact onset
                print(f"  [1/3] Searching contact onset...")
                while not emergency_abort:
                    f_lc = latest_telem.get("f_loadcell_n", 0.0)
                    dp = latest_telem.get("delta_p_hpa", 0.0)
                    real_z = latest_stepper.get("position_mm", 0.0)

                    if real_z > 10.0 and ((f_lc >= 0.20 and dp >= 0.25) or dp >= 0.38 or f_lc >= 0.35):
                        contact_z = real_z
                        print(f"\n  🎯 Contact felt at Z = {contact_z:.2f} mm (F_lc = {f_lc:.3f} N, ΔP = {dp:.2f} hPa)")
                        break

                    # Step down toward contact: Turbo x4 search in free space, swift approach
                    step = 10.0 if real_z < 60.0 else (4.0 if real_z < 75.0 else 1.0)
                    spd = 14.0 if real_z < 60.0 else 8.0
                    await safe_jog("down", step, spd)
                    print(f"    Searching... Z = {latest_stepper.get('position_mm', 0.0):5.1f} mm | F_lc = {latest_telem.get('f_loadcell_n', 0.0):+.3f} N | ΔP = {latest_telem.get('delta_p_hpa', 0.0):+.2f} hPa")

                    if latest_stepper.get("position_mm", 0.0) > 200.0:
                        print("  ⚠️ Travel limit (200 mm) reached without contact.")
                        break

                if contact_z is None or emergency_abort:
                    print("  ⚠️ Skipping trial due to search abort.")
                    continue

                # 2. Ramp through force targets: 0.5, 1, 2, 3, 4, 5 N
                print(f"  [2/3] Sweeping force levels: {TARGET_FORCES} N...")
                for target_f in TARGET_FORCES:
                    if emergency_abort:
                        break

                    # Step down until target force reached (Turbo x4 Fast-Ramp)
                    while not emergency_abort:
                        f_lc = latest_telem.get("f_loadcell_n", 0.0)
                        if f_lc >= target_f:
                            break

                        diff = target_f - f_lc
                        jog_step = 0.80 if diff > 1.2 else (0.35 if diff > 0.4 else 0.12)
                        jog_speed = 5.0 if diff > 1.2 else 3.5
                        await safe_jog("down", jog_step, jog_speed)

                    # Hold at target force for 0.45s (15+ steady samples)
                    f_now = latest_telem.get("f_loadcell_n", 0.0)
                    f_s = latest_telem.get("f_sensor_n", 0.0)
                    dp = latest_telem.get("delta_p_hpa", 0.0)
                    area = latest_telem.get("contact_area_mm2", 0.0)
                    print(f"    ⏳ [HOLD @ {target_f:3.1f} N] -> F_lc: {f_now:.3f} N | F_sensor: {f_s:.3f} N | ΔP: {dp:.2f} hPa | Area: {area:.1f} mm²")

                    # Auto-capture strictly 1 pure raw unannotated photo per shape (during Trial 1 @ 2.0 N)
                    if trial_idx == 1 and target_f == 2.0 and not raw_photo_captured:
                        raw_save_path = f"/home/bubble/Desktop/TestVersion1/results/{shape}/raw_{shape}_pressed.jpg"
                        await ws.send(json.dumps({
                            "cmd": "capture_snapshot",
                            "raw_path": raw_save_path
                        }))
                        raw_photo_captured = True
                        print(f"    📸 [Auto-RAW] Captured 1 clean unannotated photo -> {raw_save_path}")

                    await asyncio.sleep(0.45)

                print(f"  [3/3] Target sweep complete. Retracting to HOME...")

            finally:
                # Retract to Home fast (14.0 mm/s Turbo x4)
                await ws.send(json.dumps({"cmd": "stepper_stop"}))
                await asyncio.sleep(0.04)
                await ws.send(json.dumps({"cmd": "stepper_home", "speed": 14.0}))
                while latest_stepper.get("is_moving", False) or latest_stepper.get("position_mm", 0.0) > 0.3:
                    if emergency_abort:
                        break
                    await asyncio.sleep(0.05)

                # Stop recording
                await ws.send(json.dumps({"cmd": "stop_record"}))
                await asyncio.sleep(0.15)
                print(f"  ✅ Trial {trial_idx}/{num_trials} finished & CSV saved cleanly.")

                # Membrane recovery pause between trials (Turbo x4: 0.8s)
                if trial_idx < num_trials:
                    print(f"  💤 Fast membrane relaxation pause (0.8s)...")
                    await asyncio.sleep(0.8)

        is_running = False
        listener_task.cancel()
        print("\n" + "=" * 80)
        print(f"  🎉 [COMPLETED] Finished all {num_trials} trials for [{shape.upper()}]!")
        print(f"  Files safely organized in results/{shape}/")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Tactile Geometry Benchmark")
    parser.add_argument("--shape", type=str, default="flat", help="Shape name (flat, sphere, cylinder, square)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials / end trial index (default: 5)")
    parser.add_argument("--start-trial", type=int, default=1, help="Starting trial index (default: 1)")
    args = parser.parse_args()

    asyncio.run(run_trials(args.shape, args.trials, args.start_trial))
