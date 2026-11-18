#!/usr/bin/env python3
"""
Autonomous Safe Stepper Force & Hold Benchmark
-----------------------------------------------
Controls the stepper rig via server/server.py WebSocket.
Steps down carefully, searches for contact, presses up to ~3.5 N (Hard limit 4.0 N),
holds at contact to observe sensor relaxation/drift, and retracts to Home.
"""

import argparse
import asyncio
import json

WS_URL = "ws://localhost:8080/ws"

async def run_benchmark(ws_url: str = WS_URL):
    import websockets

    print("=" * 75)
    print("  🚀 AUTONOMOUS TACTILE BUBBLE BENCHMARK: PRESS & HOLD TEST")
    print("  Ceiling Force: 4.0 N max | Hold Duration: 6.0s | Speed: Controlled")
    print("=" * 75)

    async with websockets.connect(ws_url) as ws:
        latest_telem = {}
        latest_stepper = {}
        is_running = True
        emergency_abort = False

        # Background listener
        async def telemetry_listener():
            nonlocal latest_telem, latest_stepper, emergency_abort
            try:
                while is_running:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    latest_telem = data.get("telemetry", {})
                    latest_stepper = data.get("stepper", {})
                    
                    # Real-time safety watchdog (3.8 N safety margin before 4.0 N hard cut)
                    f_lc = latest_telem.get("f_loadcell_n", 0.0)
                    if f_lc >= 3.80:
                        emergency_abort = True
                        print(f"\n🚨 [WATCHDOG] Load cell = {f_lc:.2f} N >= 3.80 N! Triggering STOP & RETRACT!")
                        await ws.send(json.dumps({"cmd": "stepper_stop"}))
                        await asyncio.sleep(0.2)
                        await ws.send(json.dumps({"cmd": "stepper_home", "speed": 3.0}))
            except asyncio.CancelledError:
                pass
            except Exception as e:
                pass

        listener_task = asyncio.create_task(telemetry_listener())

        # Wait for first telemetry frame
        await asyncio.sleep(0.5)
        print("\n[1/5] Initial Sensor Health Check:")
        print(f"  Load Cell Initial: {latest_telem.get('f_loadcell_n', 0.0):.3f} N")
        print(f"  Atmospheric ΔP:   {latest_telem.get('delta_p_hpa', 0.0):.2f} hPa")
        print(f"  Tracked Dots:     {latest_telem.get('total_dots', 0)} dots")
        print(f"  Stepper Position: {latest_stepper.get('position_mm', 0.0):.2f} mm")

        # Start CSV Recording
        print("\n[2/5] Starting CSV Data Recording...")
        await ws.send(json.dumps({"cmd": "toggle_record"}))
        await asyncio.sleep(0.5)

        hold_logs_1 = []
        hold_logs_2 = []
        contact_z = None
        current_z = latest_stepper.get("position_mm", 0.0)

        try:
            print("\n[3/5] Descending to find contact...")
            # Stage A: Fast approach if starting from high up (< 65 mm)
            while current_z < 65.0 and not emergency_abort:
                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                dp = latest_telem.get("delta_p_hpa", 0.0)
                if f_lc >= 0.15 or dp >= 0.35:
                    print(f"\n🎯 Early contact felt at Z = {current_z:.2f} mm!")
                    contact_z = current_z
                    break
                step = 5.0
                await ws.send(json.dumps({"cmd": "stepper_jog", "direction": "down", "mm": step, "speed": 4.0}))
                current_z += step
                # wait for motion to finish
                while latest_stepper.get("is_moving", False) and not emergency_abort:
                    await asyncio.sleep(0.08)
                await asyncio.sleep(0.15)
                print(f"  [Approach] Z: {current_z:5.1f} mm | F_lc: {latest_telem.get('f_loadcell_n', 0.0):+.3f} N | ΔP: {latest_telem.get('delta_p_hpa', 0.0):+.2f} hPa")

            # Stage B: Careful approach (1.0 mm steps)
            while current_z < 98.0 and contact_z is None and not emergency_abort:
                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                dp = latest_telem.get("delta_p_hpa", 0.0)
                if f_lc >= 0.10 or dp >= 0.25:
                    contact_z = current_z
                    print(f"\n🎯 [CONTACT ONSET DETECTED] at Z = {contact_z:.2f} mm!")
                    print(f"    F_lc: {f_lc:.3f} N | ΔP: {dp:.2f} hPa")
                    break
                step = 1.0
                await ws.send(json.dumps({"cmd": "stepper_jog", "direction": "down", "mm": step, "speed": 2.0}))
                current_z += step
                while latest_stepper.get("is_moving", False) and not emergency_abort:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.15)
                print(f"  [Searching] Z: {current_z:5.1f} mm | F_lc: {latest_telem.get('f_loadcell_n', 0.0):+.3f} N | ΔP: {latest_telem.get('delta_p_hpa', 0.0):+.2f} hPa")

            if contact_z is None:
                print("⚠️ Search limit reached without contact onset.")
                return

            print("\n[4/5] Controlled Indentation & Hold Tests...")
            print("-" * 88)
            print(f" {'Step':^6} | {'Depth (mm)':^10} | {'ΔP (hPa)':^10} | {'Area (mm²)':^11} | {'F_loadcell':^11} | {'F_sensor':^11} | {'Err (N)':^8} | {'State':<12}")
            print("-" * 88)

            first_hold_done = False
            second_hold_done = False

            # Fine indentation loop
            while not emergency_abort:
                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                dp = latest_telem.get("delta_p_hpa", 0.0)
                area = latest_telem.get("contact_area_mm2", 0.0)
                f_s = latest_telem.get("f_sensor_n", 0.0)
                err = latest_telem.get("f_error_n", 0.0)
                depth = current_z - contact_z

                # Target 1: ~1.5 - 2.0 N -> HOLD 6 SECONDS
                if f_lc >= 1.50 and not first_hold_done:
                    print(f"\n⏳ *** [HOLD TEST 1 ACTIVE] Holding at F_lc = {f_lc:.2f} N for 6.0 seconds... ***")
                    for i in range(12):
                        if emergency_abort:
                            break
                        await asyncio.sleep(0.5)
                        snap = {
                            "time_s": (i + 1) * 0.5,
                            "depth_mm": current_z - contact_z,
                            "dp": latest_telem.get("delta_p_hpa", 0.0),
                            "area": latest_telem.get("contact_area_mm2", 0.0),
                            "f_lc": latest_telem.get("f_loadcell_n", 0.0),
                            "f_s": latest_telem.get("f_sensor_n", 0.0),
                            "err": latest_telem.get("f_error_n", 0.0),
                        }
                        hold_logs_1.append(snap)
                        print(f"  Hold 1 (+{snap['time_s']:3.1f}s) | Depth: {snap['depth_mm']:.2f}mm | ΔP: {snap['dp']:5.2f}hPa | Area: {snap['area']:5.1f}mm² | F_lc: {snap['f_lc']:.3f}N | F_s: {snap['f_s']:.3f}N | Err: {snap['err']:+.3f}N")
                    first_hold_done = True
                    print("✅ Hold 1 completed. Continuing indentation...\n")

                # Target 2: ~2.8 - 3.2 N -> HOLD 6 SECONDS
                if f_lc >= 2.80 and not second_hold_done:
                    print(f"\n⏳ *** [HOLD TEST 2 ACTIVE] Holding at F_lc = {f_lc:.2f} N for 6.0 seconds... ***")
                    for i in range(12):
                        if emergency_abort:
                            break
                        await asyncio.sleep(0.5)
                        snap = {
                            "time_s": (i + 1) * 0.5,
                            "depth_mm": current_z - contact_z,
                            "dp": latest_telem.get("delta_p_hpa", 0.0),
                            "area": latest_telem.get("contact_area_mm2", 0.0),
                            "f_lc": latest_telem.get("f_loadcell_n", 0.0),
                            "f_s": latest_telem.get("f_sensor_n", 0.0),
                            "err": latest_telem.get("f_error_n", 0.0),
                        }
                        hold_logs_2.append(snap)
                        print(f"  Hold 2 (+{snap['time_s']:3.1f}s) | Depth: {snap['depth_mm']:.2f}mm | ΔP: {snap['dp']:5.2f}hPa | Area: {snap['area']:5.1f}mm² | F_lc: {snap['f_lc']:.3f}N | F_s: {snap['f_s']:.3f}N | Err: {snap['err']:+.3f}N")
                    second_hold_done = True
                    print("✅ Hold 2 completed. Target test profile satisfied!\n")
                    break

                if f_lc >= 3.60:
                    print(f"🏁 Reached upper test threshold {f_lc:.2f} N. Stopping sweep.")
                    break

                # Advance fine step (0.25 mm)
                fine_step = 0.25
                await ws.send(json.dumps({"cmd": "stepper_jog", "direction": "down", "mm": fine_step, "speed": 1.0}))
                current_z += fine_step
                while latest_stepper.get("is_moving", False) and not emergency_abort:
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.25)

                f_lc = latest_telem.get("f_loadcell_n", 0.0)
                dp = latest_telem.get("delta_p_hpa", 0.0)
                area = latest_telem.get("contact_area_mm2", 0.0)
                f_s = latest_telem.get("f_sensor_n", 0.0)
                err = latest_telem.get("f_error_n", 0.0)
                depth = current_z - contact_z
                print(f" {depth:^6.2f} | {depth:^10.2f} | {dp:^10.2f} | {area:^11.1f} | {f_lc:^11.3f} | {f_s:^11.3f} | {err:^8.3f} | Indenting...")

        finally:
            print("\n[5/5] Finishing & Returning Stepper HOME safely...")
            await ws.send(json.dumps({"cmd": "stepper_stop"}))
            await asyncio.sleep(0.2)
            await ws.send(json.dumps({"cmd": "stepper_home", "speed": 3.0}))
            while latest_stepper.get("is_moving", False) or latest_stepper.get("position_mm", 0.0) > 0.5:
                await asyncio.sleep(0.2)
            print("✅ Stepper returned to Home (0.00 mm).")

            # Stop CSV Recording
            await ws.send(json.dumps({"cmd": "toggle_record"}))
            await asyncio.sleep(0.5)

            is_running = False
            listener_task.cancel()

        # Display Final Analytical Report
        print("\n" + "=" * 80)
        print("  📊 SUMMARY ANALYSIS: SENSOR ESTIMATION vs. GROUND TRUTH LOAD CELL")
        print("=" * 80)
        if hold_logs_1:
            h1_flc = [x["f_lc"] for x in hold_logs_1]
            h1_fs = [x["f_s"] for x in hold_logs_1]
            h1_dp = [x["dp"] for x in hold_logs_1]
            h1_area = [x["area"] for x in hold_logs_1]
            print(f"🔹 HOLD 1 (Low-Force Plateau ~1.5 - 2.0 N):")
            print(f"   Load Cell Force: {h1_flc[0]:.3f} N -> {h1_flc[-1]:.3f} N (Δ = {h1_flc[-1]-h1_flc[0]:+.3f} N)")
            print(f"   Sensor Estimate: {h1_fs[0]:.3f} N -> {h1_fs[-1]:.3f} N (Δ = {h1_fs[-1]-h1_fs[0]:+.3f} N)")
            print(f"   Mean Pressure:   {sum(h1_dp)/len(h1_dp):.2f} hPa | Mean Area: {sum(h1_area)/len(h1_area):.1f} mm²")
            print(f"   Mean Error:      {sum(x['err'] for x in hold_logs_1)/len(hold_logs_1):+.3f} N")

        if hold_logs_2:
            h2_flc = [x["f_lc"] for x in hold_logs_2]
            h2_fs = [x["f_s"] for x in hold_logs_2]
            h2_dp = [x["dp"] for x in hold_logs_2]
            h2_area = [x["area"] for x in hold_logs_2]
            print(f"\n🔹 HOLD 2 (High-Force Plateau ~2.8 - 3.2 N):")
            print(f"   Load Cell Force: {h2_flc[0]:.3f} N -> {h2_flc[-1]:.3f} N (Δ = {h2_flc[-1]-h2_flc[0]:+.3f} N)")
            print(f"   Sensor Estimate: {h2_fs[0]:.3f} N -> {h2_fs[-1]:.3f} N (Δ = {h2_fs[-1]-h2_fs[0]:+.3f} N)")
            print(f"   Mean Pressure:   {sum(h2_dp)/len(h2_dp):.2f} hPa | Mean Area: {sum(h2_area)/len(h2_area):.1f} mm²")
            print(f"   Mean Error:      {sum(x['err'] for x in hold_logs_2)/len(hold_logs_2):+.3f} N")

        print("=" * 80)
        print("📁 Raw 11-column data saved to results/")
        print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Stepper Press & Hold Test")
    parser.add_argument("--ws-url", type=str, default=WS_URL, help="WebSocket server URL")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.ws_url))
