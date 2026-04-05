"""
Module 3: Event Buffer
========================
Accumulates raw events into structured representations suitable
for downstream GPU processing.

Representations provided:
  1. RingBuffer      — low-latency circular buffer for streaming ingestion
  2. TimeWindowBuffer — accumulates events within a fixed time window [t, t+Δt]
  3. EventFrame      — projects events onto a 2D frame (event count / surface)
  4. VoxelGrid       — 3D spatiotemporal tensor (x, y, time_bins)
  5. EventSurface    — time-surface / SAE (Surface of Active Events)

Usage:
    buf = TimeWindowBuffer(sensor_size=(346, 260), window_us=10_000, n_bins=5)
    buf.push(batch)
    voxel = buf.get_voxel_grid()   # torch.Tensor (2, n_bins, H, W)
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "module1_camera"))
from event_camera import EventBatch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 1. Ring Buffer (raw event storage)
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    Lock-free circular buffer for raw events.
    Oldest events are silently overwritten when capacity is exceeded.

    Args:
        capacity:     max total events stored
        sensor_size:  (W, H) used for downstream conversions
    """

    def __init__(self, capacity: int = 2_000_000, sensor_size: tuple = (346, 260)):
        self.capacity    = capacity
        self.sensor_size = sensor_size
        self._x   = np.zeros(capacity, dtype=np.uint16)
        self._y   = np.zeros(capacity, dtype=np.uint16)
        self._t   = np.zeros(capacity, dtype=np.int64)
        self._p   = np.zeros(capacity, dtype=bool)
        self._head = 0
        self._size = 0

    def push(self, batch: EventBatch):
        n = batch.count
        if n == 0:
            return
        # Wrap-around write
        idx = np.arange(self._head, self._head + n) % self.capacity
        self._x[idx] = batch.x
        self._y[idx] = batch.y
        self._t[idx] = batch.t
        self._p[idx] = batch.polarity
        self._head = (self._head + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def peek(self, n: Optional[int] = None) -> EventBatch:
        """Return the most recent `n` events (or all if n is None)."""
        n = n or self._size
        n = min(n, self._size)
        end   = self._head
        start = (end - n) % self.capacity
        if start < end:
            idx = np.arange(start, end)
        else:
            idx = np.concatenate([np.arange(start, self.capacity),
                                  np.arange(0, end)])
        return EventBatch(
            x        = self._x[idx].copy(),
            y        = self._y[idx].copy(),
            t        = self._t[idx].copy(),
            polarity = self._p[idx].copy(),
            sensor_size = self.sensor_size,
        )

    @property
    def size(self) -> int:
        return self._size

    def clear(self):
        self._head = 0
        self._size = 0


# ---------------------------------------------------------------------------
# 2. Time Window Buffer
# ---------------------------------------------------------------------------

class TimeWindowBuffer:
    """
    Accumulates events for exactly `window_us` microseconds,
    then exposes them as various tensor formats.

    Args:
        sensor_size:  (W, H)
        window_us:    time window length in microseconds
        n_bins:       number of temporal bins for voxel grid encoding
        overlap_us:   overlap between consecutive windows (0 = no overlap)
    """

    def __init__(self, sensor_size: tuple = (346, 260),
                 window_us: int = 10_000,
                 n_bins: int    = 5,
                 overlap_us: int = 0):
        self.sensor_size = sensor_size
        self.window_us   = window_us
        self.n_bins      = n_bins
        self.overlap_us  = overlap_us

        self._xs: list = []
        self._ys: list = []
        self._ts: list = []
        self._ps: list = []
        self._window_start: Optional[int] = None
        self._ready_window: Optional[EventBatch] = None

    def push(self, batch: EventBatch) -> bool:
        """
        Push a batch of events. Returns True if a complete
        window is now available via get_*() methods.
        """
        if batch.count == 0:
            return False

        if self._window_start is None:
            self._window_start = int(batch.t[0])

        self._xs.append(batch.x)
        self._ys.append(batch.y)
        self._ts.append(batch.t)
        self._ps.append(batch.polarity)

        # Check if window is complete
        latest_t = int(batch.t[-1])
        if latest_t - self._window_start >= self.window_us:
            # Extract exactly one window
            all_x = np.concatenate(self._xs)
            all_y = np.concatenate(self._ys)
            all_t = np.concatenate(self._ts)
            all_p = np.concatenate(self._ps)

            in_window = (all_t >= self._window_start) & \
                        (all_t < self._window_start + self.window_us)

            self._ready_window = EventBatch(
                x        = all_x[in_window],
                y        = all_y[in_window],
                t        = all_t[in_window],
                polarity = all_p[in_window],
                sensor_size = self.sensor_size,
            )

            # Slide window (keep overlap)
            next_start = self._window_start + self.window_us - self.overlap_us
            carry_mask = all_t >= next_start
            self._xs = [all_x[carry_mask]]
            self._ys = [all_y[carry_mask]]
            self._ts = [all_t[carry_mask]]
            self._ps = [all_p[carry_mask]]
            self._window_start = next_start
            return True

        return False

    def get_raw(self) -> Optional[EventBatch]:
        """Return the completed window as a raw EventBatch."""
        return self._ready_window

    def get_event_frame(self) -> np.ndarray:
        """
        Project events onto a 2-channel frame (ON, OFF counts).
        Shape: (2, H, W), dtype float32.
        """
        b = self._ready_window
        if b is None:
            return None
        W, H = self.sensor_size
        frame = np.zeros((2, H, W), dtype=np.float32)
        np.add.at(frame[0], (b.y[b.polarity],  b.x[b.polarity]),  1)
        np.add.at(frame[1], (b.y[~b.polarity], b.x[~b.polarity]), 1)
        return frame

    def get_voxel_grid(self) -> np.ndarray:
        """
        Encode events into a (2, T, H, W) voxel grid using
        bilinear temporal interpolation (Zhu et al. 2019).

        Returns np.ndarray shape (2, n_bins, H, W), dtype float32.
        """
        b = self._ready_window
        if b is None or b.count == 0:
            W, H = self.sensor_size
            return np.zeros((2, self.n_bins, H, W), dtype=np.float32)

        W, H = self.sensor_size
        T    = self.n_bins
        grid = np.zeros((2, T, H, W), dtype=np.float32)

        t0  = b.t.min()
        t1  = b.t.max()
        dt  = max(t1 - t0, 1)

        # Normalised time in [0, T-1]
        t_norm = (b.t - t0) / dt * (T - 1)

        t_low  = np.floor(t_norm).astype(int)
        t_high = t_low + 1
        w_high = t_norm - t_low
        w_low  = 1.0 - w_high

        for pol in [0, 1]:
            mask = (b.polarity == (pol == 0))
            if not np.any(mask):
                continue
            xs = b.x[mask]
            ys = b.y[mask]
            tl = t_low[mask]
            th = t_high[mask]
            wl = w_low[mask]
            wh = w_high[mask]

            # Low bin
            valid_l = tl < T
            np.add.at(grid[pol], (tl[valid_l], ys[valid_l], xs[valid_l]), wl[valid_l])

            # High bin
            valid_h = th < T
            np.add.at(grid[pol], (th[valid_h], ys[valid_h], xs[valid_h]), wh[valid_h])

        return grid

    def get_time_surface(self, decay_us: float = 30_000) -> np.ndarray:
        """
        Surface of Active Events (SAE / time surface).
        Each pixel holds exp(-(t_now - t_last) / decay).
        Shape: (2, H, W), dtype float32.
        """
        b = self._ready_window
        if b is None or b.count == 0:
            W, H = self.sensor_size
            return np.zeros((2, H, W), dtype=np.float32)

        W, H    = self.sensor_size
        surface = np.full((2, H, W), -np.inf, dtype=np.float64)

        # Latest timestamp per pixel per polarity
        for pol in [0, 1]:
            mask = (b.polarity == (pol == 0))
            if not np.any(mask):
                continue
            xs = b.x[mask]
            ys = b.y[mask]
            ts = b.t[mask].astype(np.float64)
            # Vectorised max using sorting trick
            order = np.argsort(ts)
            np.maximum.at(surface[pol], (ys[order], xs[order]), ts[order])

        t_now = float(b.t.max())
        out   = np.exp((surface - t_now) / decay_us)
        out[surface == -np.inf] = 0.0
        return out.astype(np.float32)

    def to_torch(self, representation: str = "voxel") -> "torch.Tensor":
        """
        Convert the current window to a PyTorch tensor.

        Args:
            representation: "voxel" | "frame" | "surface"

        Returns:
            torch.Tensor on CPU (move to GPU in Module 4)
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed: pip install torch")

        fn = {
            "voxel":   self.get_voxel_grid,
            "frame":   self.get_event_frame,
            "surface": self.get_time_surface,
        }
        if representation not in fn:
            raise ValueError(f"Unknown representation: {representation}")

        arr = fn[representation]()
        return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# 3. Sliding-window stream helper
# ---------------------------------------------------------------------------

def sliding_windows(event_stream,
                    sensor_size: tuple,
                    window_us: int     = 10_000,
                    n_bins: int        = 5,
                    representation: str = "voxel",
                    overlap_us: int    = 0):
    """
    Generator that wraps an EventBatch stream and yields
    (window_idx, tensor) pairs.

    Args:
        event_stream:    iterable of EventBatch
        sensor_size:     (W, H)
        window_us:       window length in µs
        n_bins:          temporal bins
        representation:  "voxel" | "frame" | "surface"
        overlap_us:      overlap between windows

    Yields:
        (int, np.ndarray) — window index and representation array
    """
    buf = TimeWindowBuffer(
        sensor_size = sensor_size,
        window_us   = window_us,
        n_bins      = n_bins,
        overlap_us  = overlap_us,
    )
    window_idx = 0
    fn = {
        "voxel":   buf.get_voxel_grid,
        "frame":   buf.get_event_frame,
        "surface": buf.get_time_surface,
    }[representation]

    for batch in event_stream:
        if buf.push(batch):
            yield window_idx, fn()
            window_idx += 1


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 3: Event Buffer Demo ===\n")

    # Simulate upstream modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "module2_driver"))
    from camera_driver import CameraDriver

    driver = CameraDriver(backend="mock", sensor_size=(346, 260),
                          event_rate_hz=500_000)
    driver.configure(noise_filter_us=0)
    driver.start()

    ring = RingBuffer(capacity=1_000_000, sensor_size=(346, 260))
    buf  = TimeWindowBuffer(sensor_size=(346, 260), window_us=10_000, n_bins=5)

    print(f"{'Win':>4}  {'Events':>8}  {'Voxel shape':>16}  {'Frame max':>10}  {'Surface max':>12}")
    print("-" * 60)

    win_count = 0
    for batch in driver.stream(duration_ms=5):
        ring.push(batch)

        if buf.push(batch):
            voxel   = buf.get_voxel_grid()
            frame   = buf.get_event_frame()
            surface = buf.get_time_surface()

            print(f"{win_count:>4}  {buf.get_raw().count:>8,}  "
                  f"{str(voxel.shape):>16}  "
                  f"{frame.max():>10.2f}  "
                  f"{surface.max():>12.4f}")
            win_count += 1

        if win_count >= 8:
            break

    driver.stop()

    # Show ring buffer
    recent = ring.peek(n=500)
    print(f"\nRing buffer peek: {recent}")

    # Torch conversion
    if HAS_TORCH:
        t = buf.to_torch("voxel")
        print(f"\nPyTorch voxel tensor: shape={t.shape}, dtype={t.dtype}")
    else:
        print("\n(PyTorch not installed — skipping tensor demo)")
