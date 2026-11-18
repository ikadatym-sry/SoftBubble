import argparse
import asyncio
import json


async def lift_and_zero(dist_total: float = 15.0, step: float = 5.0, ws_url: str = "ws://localhost:8080/ws"):
    import websockets
    async with websockets.connect(ws_url) as ws:
        steps = max(1, int(round(dist_total / step)))
        for i in range(steps):
            print(f"Jogging UP {step:.1f} mm (Step {i+1}/{steps})...")
            await ws.send(json.dumps({'cmd': 'stepper_jog', 'direction': 'up', 'mm': step, 'speed': 3.5}))
            
            # Wait for movement to register
            await asyncio.sleep(0.3)
            while True:
                msg = await ws.recv()
                d = json.loads(msg)
                st = d.get('stepper', {})
                telem = d.get('telemetry', {})
                if not st.get('is_moving', False):
                    pos = st.get('position_mm')
                    top = st.get('top_limit')
                    f_lc = telem.get('f_loadcell_n')
                    print(f"  Step {i+1} complete: pos={pos} mm, top_limit={top}, LoadCell={f_lc} N")
                    break
                await asyncio.sleep(0.1)

        # Zero the position so this new raised point is the new Home (0.00 mm)
        print("Zeroing stepper position to set new Home reference...")
        await ws.send(json.dumps({'cmd': 'stepper_zero'}))
        await asyncio.sleep(0.5)

        # Final check
        msg = await ws.recv()
        d = json.loads(msg)
        st = d.get('stepper', {})
        telem = d.get('telemetry', {})
        print(f"✅ Final State -> Stepper: {st.get('position_mm')} mm | LoadCell: {telem.get('f_loadcell_n')} N | ΔP: {telem.get('delta_p_hpa')} hPa")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lift Z-carriage and re-zero home position')
    parser.add_argument('--distance', type=float, default=15.0, help='Total lift distance in mm (default: 15.0)')
    parser.add_argument('--step', type=float, default=5.0, help='Step size per jog in mm (default: 5.0)')
    parser.add_argument('--ws-url', type=str, default='ws://localhost:8080/ws', help='WebSocket server URL')
    args = parser.parse_args()
    asyncio.run(lift_and_zero(args.distance, args.step, args.ws_url))
