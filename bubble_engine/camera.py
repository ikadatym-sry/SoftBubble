"""
Threaded Picamera2 camera capture module with configurable manual focus for Raspberry Pi cameras
(tested with Raspberry Pi Camera Module 3 Wide / imx708_wide using fixed macro LensPosition).
"""

import time
import threading
from typing import Optional, Tuple
import numpy as np

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcamera_controls
except ImportError:
    Picamera2 = None
    libcamera_controls = None


class BubbleCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        lens_position: float = 12.0,  # Macro focus (~8.3 cm)
        fps: int = 30,
    ):
        self.width = width
        self.height = height
        self.lens_position = lens_position
        self.fps = fps

        self.picam: Optional[Picamera2] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0

    def start(self):
        if self._running:
            return

        if Picamera2 is None:
            raise RuntimeError(
                "picamera2 is not installed or not available in this Python environment."
            )

        for attempt in range(3):
            try:
                print(f"[BubbleCamera] Initializing Picamera2 (attempt {attempt + 1}/3)...")
                self.picam = Picamera2()
                break
            except Exception as e:
                print(f"[BubbleCamera] Camera busy ({e}). Freeing /dev/media and /dev/video...")
                import os
                os.system("fuser -k -9 /dev/media* /dev/video* 2>/dev/null")
                time.sleep(1.2)
        else:
            raise RuntimeError("Camera is locked by another process and could not be acquired.")

        config = self.picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        self.picam.configure(config)
        self.picam.start()

        # Apply controls for macro focus (if supported) and frame rate
        if libcamera_controls is not None:
            frame_duration_us = int(1_000_000 / self.fps)
            try:
                self.picam.set_controls({
                    "AfMode": libcamera_controls.AfModeEnum.Manual,
                    "LensPosition": float(self.lens_position),
                    "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                })
            except Exception as e:
                print(f"[BubbleCamera] Manual AfMode/LensPosition unsupported ({e}); applying FrameDurationLimits only.")
                try:
                    self.picam.set_controls({"FrameDurationLimits": (frame_duration_us, frame_duration_us)})
                except Exception:
                    pass
        time.sleep(0.5)  # Allow ISP and sensor exposure to settle

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[BubbleCamera] Capture thread started at {self.width}x{self.height} @ {self.fps} FPS.")

    def _capture_loop(self):
        while self._running:
            try:
                frame_rgb = self.picam.capture_array()
                with self._lock:
                    self._latest_frame = frame_rgb
                    self._frame_id += 1
            except Exception as e:
                print(f"[BubbleCamera] Capture error: {e}")
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Returns (success, frame_rgb, frame_id).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0
            return True, self._latest_frame.copy(), self._frame_id

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
            self.picam = None
        print("[BubbleCamera] Camera stopped cleanly.")
