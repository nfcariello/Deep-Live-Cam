"""Optional virtual-camera output for the live face swap.

Lets Deep-Live-Cam appear as a webcam in Teams / Zoom / Meet / etc. by pushing
the swapped frames to a system virtual-camera device via ``pyvirtualcam``.

On macOS the backend device is the **OBS Virtual Camera** (installed by OBS
Studio); on Windows it is OBS or Unity Capture; on Linux it is v4l2loopback.

Everything here is best-effort and never raises into the live pipeline: if
``pyvirtualcam`` isn't installed, or no backend device exists, :meth:`start`
returns ``(False, reason)`` and the caller simply runs without a virtual camera.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class VirtualCamera:
    """Thin, crash-safe wrapper around a ``pyvirtualcam.Camera``."""

    def __init__(self) -> None:
        self._cam = None
        self._width = 0
        self._height = 0

    @property
    def is_running(self) -> bool:
        return self._cam is not None

    def start(self, width: int, height: int, fps: float) -> Tuple[bool, str]:
        """Open the virtual camera. Returns ``(ok, message)``; never raises."""
        if self._cam is not None:
            return True, "Virtual camera already running"

        try:
            import pyvirtualcam
        except Exception as exc:  # ImportError, or a broken install
            return False, (
                "Virtual camera unavailable: pyvirtualcam is not installed "
                f"({exc}). Run: pip install pyvirtualcam"
            )

        fps_int = int(round(fps)) if fps and fps > 0 else 30
        try:
            self._cam = pyvirtualcam.Camera(
                width=int(width), height=int(height), fps=fps_int
            )
        except Exception as exc:
            self._cam = None
            return False, (
                "Virtual camera unavailable: could not open a device "
                f"({type(exc).__name__}: {exc}). On macOS, install OBS to "
                "provide the virtual camera, and make sure OBS's own "
                "'Start Virtual Camera' is off so this app can feed it."
            )

        self._width = int(width)
        self._height = int(height)
        device = getattr(self._cam, "device", "virtual camera")
        return True, f"Virtual camera started ({device})"

    def send(self, frame_bgr: np.ndarray) -> None:
        """Push one BGR frame to the virtual camera. No-op if not running."""
        cam = self._cam
        if cam is None:
            return
        try:
            if (
                frame_bgr.shape[1] != self._width
                or frame_bgr.shape[0] != self._height
            ):
                frame_bgr = cv2.resize(frame_bgr, (self._width, self._height))
            # pyvirtualcam defaults to RGB; cvtColor yields a contiguous array.
            cam.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            # A transient send failure must never take down the live loop.
            pass

    def close(self) -> None:
        cam = self._cam
        self._cam = None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
