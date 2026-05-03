"""
Module 1: Event Camera Hardware Interface
==========================================
Supports: iniVation DAVIS/DVS (via libcaer), Prophesee (via Metavision SDK)

Install:
    # libcaer (iniVation)
    pip install pyaer
    # OR Metavision SDK (Prophesee)
    pip install metavision-sdk-core metavision-sdk-driver

Usage:
    cam = EventCamera(backend="pyaer")   # or "metavision"
    cam.start()
    for batch in cam.stream(duration_ms=10):
        print(batch)   # numpy array (N, 4): [x, y, t, polarity]
    cam.stop()
"""

import numpy as np
import threading
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class EventBatch:
    """A batch of raw events from the sensor."""
    x:        np.ndarray   # shape (N,), uint16
    y:        np.ndarray   # shape (N,), uint16
    t:        np.ndarray   # shape (N,), int64  — microseconds
    polarity: np.ndarray   # shape (N,), bool   — True=ON, False=OFF
    sensor_size: tuple     # (width, height)

    @property
    def count(self) -> int:
        return len(self.x)

    def to_array(self) -> np.ndarray:
        """Return (N, 4) array: [x, y, t, polarity]"""
        return np.stack([self.x, self.y, self.t, self.polarity.astype(np.uint8)], axis=1)

    def __repr__(self):
        return (f"EventBatch(N={self.count}, "
                f"t=[{self.t.min()}..{self.t.max()}] µs, "
                f"ON={self.polarity.sum()}, OFF={(~self.polarity).sum()})")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EventCameraBase(ABC):
    def __init__(self, sensor_size: tuple = (346, 260)):
        self.sensor_size = sensor_size
        self._running = False
        self._queue: queue.Queue = queue.Queue(maxsize=128)

    @abstractmethod
    def _open_device(self): ...

    @abstractmethod
    def _close_device(self): ...

    @abstractmethod
    def _poll_events(self) -> Optional[EventBatch]: ...

    def start(self):
        self._open_device()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[EventCamera] Started — sensor {self.sensor_size}")

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._close_device()
        print("[EventCamera] Stopped")

    def _capture_loop(self):
        while self._running:
            batch = self._poll_events()
            if batch is not None and batch.count > 0:
                try:
                    self._queue.put_nowait(batch)
                except queue.Full:
                    pass   # drop oldest — real-time priority

    def stream(self, duration_ms: float = 10.0) -> Iterator[EventBatch]:
        """Yield EventBatch objects continuously while running."""
        while self._running:
            try:
                batch = self._queue.get(timeout=duration_ms / 1000.0)
                yield batch
            except queue.Empty:
                continue


# ---------------------------------------------------------------------------
# Backend: pyaer (iniVation DAVIS / DVS cameras)
# ---------------------------------------------------------------------------

class PyAERCamera(EventCameraBase):
    """
    Uses the pyaer Python binding for libcaer.
    Supports: DVS128, DAVIS240, DVS346, DAVIS346, DVS-Nano, etc.
    """

    def __init__(self, device_id: int = 0, bias_sensitivity: int = 5):
        super().__init__()
        self.device_id = device_id
        self.bias_sensitivity = bias_sensitivity
        self._device = None

    def _open_device(self):
        try:
            from pyaer import libcaer
            from pyaer.davis import DAVIS
        except ImportError:
            raise RuntimeError("pyaer not installed. Run: pip install pyaer")

        self._device = DAVIS(device_id=self.device_id)
        self._device.start_data_stream()

        info = self._device.get_camera_info()
        self.sensor_size = (info.dvsSizeX, info.dvsSizeY)

        # Apply bias for sensitivity
        self._device.set_bias_from_json(f"./biases/davis_sensitivity_{self.bias_sensitivity}.json",
                                         verbose=False)
        print(f"[PyAER] Device opened: {info.deviceString}, sensor={self.sensor_size}")

    def _close_device(self):
        if self._device:
            self._device.shutdown()

    def _poll_events(self) -> Optional[EventBatch]:
        if self._device is None:
            return None

        data, _ = self._device.get_event_packet()
        if data is None:
            return None

        events = data.get("events", None)
        if events is None or len(events) == 0:
            return None

        return EventBatch(
            x        = events["x"].astype(np.uint16),
            y        = events["y"].astype(np.uint16),
            t        = events["timestamp"].astype(np.int64),
            polarity = events["polarity"].astype(bool),
            sensor_size = self.sensor_size,
        )


# ---------------------------------------------------------------------------
# Backend: Metavision SDK (Prophesee cameras)
# ---------------------------------------------------------------------------

class MetavisionCamera(EventCameraBase):
    """
    Uses Prophesee Metavision SDK.
    Supports: EVK3, EVK4, SilkyEvCam, VGA, HD cameras.
    """

    def __init__(self, serial: str = "", raw_file: str = ""):
        super().__init__()
        self.serial   = serial
        self.raw_file = raw_file
        self._controller = None
        self._buf: list = []

    def _open_device(self):
        try:
            import metavision_sdk_core as mv_core
            import metavision_sdk_driver as mv_driver
        except ImportError:
            raise RuntimeError(
                "Metavision SDK not installed.\n"
                "  pip install metavision-sdk-core metavision-sdk-driver"
            )

        if self.raw_file:
            self._controller = mv_driver.Camera.from_file(self.raw_file)
        elif self.serial:
            self._controller = mv_driver.Camera.from_serial(self.serial)
        else:
            self._controller = mv_driver.Camera.from_first_available()

        geom = self._controller.get_camera_configuration().resolution
        self.sensor_size = (geom.width, geom.height)

        # Register callback
        self._controller.cd.add_callback(self._mv_callback)
        self._controller.start()
        print(f"[Metavision] Device opened: sensor={self.sensor_size}")

    def _mv_callback(self, evts):
        self._buf.append(evts.copy())

    def _close_device(self):
        if self._controller:
            self._controller.stop()

    def _poll_events(self) -> Optional[EventBatch]:
        if not self._buf:
            time.sleep(0.001)
            return None

        chunk = self._buf.pop(0)
        return EventBatch(
            x        = chunk["x"].astype(np.uint16),
            y        = chunk["y"].astype(np.uint16),
            t        = chunk["t"].astype(np.int64),
            polarity = chunk["p"].astype(bool),
            sensor_size = self.sensor_size,
        )


# ---------------------------------------------------------------------------
# Mock camera — for development / testing without hardware
# ---------------------------------------------------------------------------

class MockEventCamera(EventCameraBase):
    """
    Synthetic event stream — useful for testing the full pipeline
    without physical hardware.

    Generates a moving bright edge (Gaussian profile) across the sensor,
    producing realistic ON/OFF event distributions.
    """

    def __init__(self, sensor_size=(346, 260), event_rate_hz=1_000_000,
                 time_window_us=10_000):
        super().__init__(sensor_size)
        self.event_rate_hz  = event_rate_hz
        self.time_window_us = time_window_us
        self._t_current     = 0

    def _open_device(self):
        print(f"[MockCamera] Synthetic sensor {self.sensor_size} @ {self.event_rate_hz/1e6:.1f}M ev/s")

    def _close_device(self):
        pass

    def _poll_events(self) -> Optional[EventBatch]:
        W, H = self.sensor_size
        dt   = self.time_window_us

        n_events = int(self.event_rate_hz * dt / 1e6)
        if n_events == 0:
            return None

        # Moving vertical edge: x-position oscillates sinusoidally
        phase  = (self._t_current % 2_000_000) / 2_000_000 * 2 * np.pi
        edge_x = int(W / 2 + (W * 0.4) * np.sin(phase))

        # Events cluster near the edge with Gaussian spread
        x_raw  = np.random.normal(loc=edge_x, scale=8, size=n_events)
        x      = np.clip(x_raw, 0, W - 1).astype(np.uint16)
        y      = np.random.randint(0, H, n_events, dtype=np.uint16)
        t      = np.sort(
            np.random.randint(self._t_current,
                              self._t_current + dt,
                              n_events, dtype=np.int64)
        )
        # ON events on leading edge, OFF on trailing
        polarity = (x_raw - edge_x) > 0

        self._t_current += dt
        time.sleep(dt / 1e6)   # simulate real-time

        return EventBatch(x=x, y=y, t=t, polarity=polarity,
                          sensor_size=self.sensor_size)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def EventCamera(backend: str = "mock", **kwargs) -> EventCameraBase:
    """
    Factory function.

    Args:
        backend: "mock" | "pyaer" | "metavision"
        **kwargs: forwarded to the backend constructor

    Returns:
        An EventCameraBase instance (not yet started).
    """
    backends = {
        "mock":       MockEventCamera,
        "pyaer":      PyAERCamera,
        "metavision": MetavisionCamera,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(backends)}")
    return backends[backend](**kwargs)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 1: Event Camera Demo ===\n")

    cam = EventCamera(backend="mock", sensor_size=(346, 260), event_rate_hz=500_000)
    cam.start()

    total_events = 0
    t0 = time.time()

    print(f"{'Batch':>6}  {'N events':>10}  {'t_start (µs)':>14}  {'ON%':>6}")
    print("-" * 50)

    for i, batch in enumerate(cam.stream(duration_ms=10)):
        total_events += batch.count
        on_pct = 100 * batch.polarity.sum() / max(batch.count, 1)
        print(f"{i+1:>6}  {batch.count:>10,}  {batch.t.min():>14,}  {on_pct:>5.1f}%")
        if i >= 9:
            break

    elapsed = time.time() - t0
    cam.stop()

    print(f"\nTotal events: {total_events:,} in {elapsed:.2f}s "
          f"({total_events/elapsed/1e6:.2f}M ev/s)")
    print("\nEventBatch array shape:", batch.to_array().shape)
    print("Sample rows (x, y, t, pol):\n", batch.to_array()[:5])
