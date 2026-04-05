"""
Module 2: Driver / SDK Integration
====================================
Handles device discovery, bias tuning, and provides a unified
driver interface on top of Module 1's EventCamera backends.

Features:
  - Auto-detect connected cameras
  - Load / save / tune bias parameters
  - Apply ROI (Region of Interest) masking
  - Hot-pixel filtering
  - Noise filtering (background activity filter)

Usage:
    driver = CameraDriver(backend="mock")
    driver.configure(sensitivity=7, noise_filter_us=1000)
    driver.start()

    for batch in driver.stream():
        process(batch)

    driver.stop()
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

# Allow importing Module 1 from parent package
sys.path.insert(0, str(Path(__file__).parent.parent / "module1_camera"))
from event_camera import EventCamera, EventBatch, EventCameraBase


# ---------------------------------------------------------------------------
# Bias / configuration profiles
# ---------------------------------------------------------------------------

DEFAULT_BIASES = {
    "sensitivity": 5,          # 1–10: higher = more events
    "noise_filter_us": 2000,   # Background Activity Filter window (µs)
    "refractory_period_us": 0, # Per-pixel refractory (0 = disabled)
    "hot_pixel_threshold": 0,  # Events/s above which pixel is masked (0 = off)
    "roi": None,               # (x0, y0, x1, y1) or None for full frame
}


class BiasConfig:
    """Stores and serialises camera bias/config parameters."""

    def __init__(self, **kwargs):
        self._cfg: Dict[str, Any] = {**DEFAULT_BIASES, **kwargs}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._cfg.get(name)

    def update(self, **kwargs):
        self._cfg.update(kwargs)
        return self

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._cfg, f, indent=2)
        print(f"[BiasConfig] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BiasConfig":
        with open(path) as f:
            data = json.load(f)
        print(f"[BiasConfig] Loaded from {path}")
        return cls(**data)

    def __repr__(self):
        return f"BiasConfig({self._cfg})"


# ---------------------------------------------------------------------------
# Filters applied on the CPU before buffering
# ---------------------------------------------------------------------------

class BackgroundActivityFilter:
    """
    Per-pixel timestamp filter. Suppresses isolated events that have
    no neighbour within `dt_us` microseconds in an 8-connected window.
    This is the standard BAF used in event-camera research.
    """

    def __init__(self, sensor_size: tuple, dt_us: int = 2000):
        W, H = sensor_size
        self.dt_us = dt_us
        # Timestamp of last event per pixel (initialised to large negative)
        self._last_t = np.full((H, W), -10**9, dtype=np.int64)

    def apply(self, batch: EventBatch) -> EventBatch:
        if self.dt_us <= 0:
            return batch

        keep = np.zeros(batch.count, dtype=bool)
        W, H = batch.sensor_size

        xs = batch.x.astype(np.int32)
        ys = batch.y.astype(np.int32)

        for i in range(batch.count):
            x, y, t = xs[i], ys[i], batch.t[i]

            # Check 8-connected neighbourhood
            x0, x1 = max(0, x - 1), min(W - 1, x + 1)
            y0, y1 = max(0, y - 1), min(H - 1, y + 1)

            neighbours = self._last_t[y0:y1+1, x0:x1+1]
            if np.any(t - neighbours <= self.dt_us):
                keep[i] = True

            self._last_t[y, x] = t

        return EventBatch(
            x        = batch.x[keep],
            y        = batch.y[keep],
            t        = batch.t[keep],
            polarity = batch.polarity[keep],
            sensor_size = batch.sensor_size,
        )


class HotPixelFilter:
    """
    Masks pixels that fire more than `threshold` events/second.
    Uses a rolling 1-second histogram to detect pathological pixels.
    """

    def __init__(self, sensor_size: tuple, threshold: int = 5000):
        W, H = sensor_size
        self.threshold = threshold
        self._counts   = np.zeros((H, W), dtype=np.int32)
        self._window_t = None
        self._masked   = np.zeros((H, W), dtype=bool)

    def apply(self, batch: EventBatch) -> EventBatch:
        if self.threshold <= 0 or batch.count == 0:
            return batch

        # Initialise window start
        if self._window_t is None:
            self._window_t = batch.t[0]

        # Roll window every 1 second (1e6 µs)
        if batch.t[-1] - self._window_t > 1_000_000:
            self._masked = self._counts > self.threshold
            self._counts[:] = 0
            self._window_t = batch.t[-1]

        # Accumulate
        np.add.at(self._counts, (batch.y, batch.x), 1)

        # Remove masked pixels
        mask = ~self._masked[batch.y, batch.x]
        return EventBatch(
            x        = batch.x[mask],
            y        = batch.y[mask],
            t        = batch.t[mask],
            polarity = batch.polarity[mask],
            sensor_size = batch.sensor_size,
        )


class ROIFilter:
    """Crops events to a rectangular region of interest."""

    def __init__(self, x0: int, y0: int, x1: int, y1: int):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def apply(self, batch: EventBatch) -> EventBatch:
        mask = ((batch.x >= self.x0) & (batch.x <= self.x1) &
                (batch.y >= self.y0) & (batch.y <= self.y1))
        return EventBatch(
            x        = batch.x[mask] - self.x0,
            y        = batch.y[mask] - self.y0,
            t        = batch.t[mask],
            polarity = batch.polarity[mask],
            sensor_size = (self.x1 - self.x0, self.y1 - self.y0),
        )


# ---------------------------------------------------------------------------
# CameraDriver — high-level driver wrapping Module 1 + filters
# ---------------------------------------------------------------------------

class CameraDriver:
    """
    High-level driver. Combines:
      - EventCamera (Module 1) for hardware/mock streaming
      - BiasConfig for camera tuning
      - Optional filter chain (BAF → HotPixel → ROI)
    """

    def __init__(self, backend: str = "mock", bias: Optional[BiasConfig] = None,
                 **camera_kwargs):
        self.bias   = bias or BiasConfig()
        self._cam   = EventCamera(backend=backend, **camera_kwargs)
        self._filters = []
        self._stats = {"in": 0, "out": 0, "batches": 0}

    def configure(self,
                  sensitivity: int = 5,
                  noise_filter_us: int = 2000,
                  hot_pixel_threshold: int = 0,
                  roi: Optional[tuple] = None):
        """
        Apply configuration and build the filter chain.

        Args:
            sensitivity:         1–10, bias for event rate
            noise_filter_us:     BAF window in microseconds (0 = disabled)
            hot_pixel_threshold: max events/s per pixel (0 = disabled)
            roi:                 (x0, y0, x1, y1) or None
        """
        self.bias.update(
            sensitivity       = sensitivity,
            noise_filter_us   = noise_filter_us,
            hot_pixel_threshold = hot_pixel_threshold,
            roi               = roi,
        )

        # Build filter chain
        self._filters = []
        if noise_filter_us > 0:
            self._filters.append(
                BackgroundActivityFilter(self._cam.sensor_size, noise_filter_us))
        if hot_pixel_threshold > 0:
            self._filters.append(
                HotPixelFilter(self._cam.sensor_size, hot_pixel_threshold))
        if roi is not None:
            self._filters.append(ROIFilter(*roi))

        print(f"[CameraDriver] Config: {self.bias}")
        print(f"[CameraDriver] Filter chain: {[type(f).__name__ for f in self._filters]}")

    def start(self):
        self._cam.start()

    def stop(self):
        self._cam.stop()
        retention = 100 * self._stats["out"] / max(self._stats["in"], 1)
        print(f"[CameraDriver] Stats — in: {self._stats['in']:,}  "
              f"out: {self._stats['out']:,}  "
              f"retention: {retention:.1f}%  "
              f"batches: {self._stats['batches']:,}")

    def stream(self, duration_ms: float = 10.0):
        """Yield filtered EventBatch objects."""
        for raw in self._cam.stream(duration_ms=duration_ms):
            self._stats["in"]    += raw.count
            self._stats["batches"] += 1

            filtered = raw
            for f in self._filters:
                filtered = f.apply(filtered)

            self._stats["out"] += filtered.count

            if filtered.count > 0:
                yield filtered

    @staticmethod
    def list_devices() -> list:
        """
        Attempt to enumerate connected event cameras.
        Returns list of dicts with device info.
        """
        devices = []

        # Try pyaer
        try:
            from pyaer import libcaer
            # libcaer doesn't expose a clean enumeration API;
            # we probe device IDs 0–7
            for dev_id in range(8):
                try:
                    from pyaer.davis import DAVIS
                    d = DAVIS(device_id=dev_id)
                    info = d.get_camera_info()
                    devices.append({
                        "backend":   "pyaer",
                        "device_id": dev_id,
                        "name":      info.deviceString,
                        "serial":    str(info.deviceSerialNumber),
                    })
                    d.shutdown()
                except Exception:
                    pass
        except ImportError:
            pass

        # Try Metavision
        try:
            import metavision_sdk_driver as mv_driver
            for cam_info in mv_driver.Camera.list_online_cameras():
                devices.append({
                    "backend": "metavision",
                    "serial":  cam_info.serial,
                    "name":    cam_info.integrator_name,
                })
        except Exception:
            pass

        if not devices:
            devices.append({
                "backend": "mock",
                "name":    "MockEventCamera (no hardware found)",
            })

        return devices


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 2: CameraDriver Demo ===\n")

    print("Scanning for devices...")
    for d in CameraDriver.list_devices():
        print(f"  Found: {d}")

    driver = CameraDriver(
        backend       = "mock",
        sensor_size   = (346, 260),
        event_rate_hz = 800_000,
    )

    driver.configure(
        sensitivity         = 7,
        noise_filter_us     = 1500,
        hot_pixel_threshold = 10_000,
        roi                 = (50, 30, 296, 230),   # 246×200 crop
    )

    driver.start()

    print(f"\n{'Batch':>6}  {'Raw N':>10}  {'Filtered N':>12}  {'Retention':>10}")
    print("-" * 50)

    for i, batch in enumerate(driver.stream(duration_ms=10)):
        if i < 10:
            print(f"{i+1:>6}  {'?':>10}  {batch.count:>12,}  {'–':>10}")
            print(f"         sensor_size after ROI: {batch.sensor_size}")
        if i >= 9:
            break

    driver.stop()
