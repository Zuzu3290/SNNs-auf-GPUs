"""
Module 4: GPU Memory Management
=================================
Handles host→device transfers for event tensors using:
  - PyTorch pinned memory for fast DMA transfers
  - CUDA stream-based async copies (overlapping compute and transfer)
  - Pre-allocated device buffers to avoid per-frame allocation overhead
  - Double buffering to hide transfer latency

Usage:
    gm = GPUMemoryManager(device="cuda:0", sensor_size=(346,260), n_bins=5)
    gm.allocate()

    # In the streaming loop:
    device_tensor = gm.push(voxel_grid_np)   # returns torch.Tensor on GPU
    # device_tensor is ready for Module 5 CUDA kernel
"""

import numpy as np
import time
from typing import Optional, Tuple

try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def get_best_device() -> str:
    """Return 'cuda:0' if available, else 'cpu'."""
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def gpu_info() -> dict:
    """Return dict with basic GPU info."""
    if not (HAS_TORCH and torch.cuda.is_available()):
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available":    True,
        "name":         props.name,
        "total_mem_gb": props.total_memory / 1e9,
        "free_mem_gb":  torch.cuda.mem_get_info()[0] / 1e9,
        "sm_count":     props.multi_processor_count,
        "compute_cap":  f"{props.major}.{props.minor}",
    }


# ---------------------------------------------------------------------------
# GPUMemoryManager
# ---------------------------------------------------------------------------

class GPUMemoryManager:
    """
    Manages pre-allocated GPU buffers and async host→device transfers
    for event camera voxel grids / frames.

    Args:
        device:      torch device string, e.g. "cuda:0"
        sensor_size: (W, H)
        n_bins:      temporal bins (for voxel grid shape)
        n_polarities: 2 (ON/OFF) or 1 for frame
        dtype:       torch.float32 recommended
        double_buffer: use two alternating device buffers to overlap
                       H2D transfer with kernel execution
    """

    def __init__(self,
                 device:       str   = "cuda:0",
                 sensor_size:  tuple = (346, 260),
                 n_bins:       int   = 5,
                 n_polarities: int   = 2,
                 dtype               = None,
                 double_buffer: bool = True):

        if not HAS_TORCH:
            raise RuntimeError("PyTorch required: pip install torch")

        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sensor_size   = sensor_size
        self.n_bins        = n_bins
        self.n_polarities  = n_polarities
        self.dtype         = dtype or torch.float32
        self.double_buffer = double_buffer

        W, H = sensor_size
        # Shape: (polarity, time_bins, H, W)
        self._shape = (n_polarities, n_bins, H, W)

        self._pinned: Optional[torch.Tensor]       = None  # page-locked host buf
        self._gpu_bufs: list                       = []    # device buffers
        self._streams:  list                       = []    # CUDA streams
        self._buf_idx:  int                        = 0     # current buffer slot
        self._allocated: bool                      = False

        self._stats = {
            "transfers": 0,
            "total_bytes": 0,
            "total_ms": 0.0,
        }

    def allocate(self):
        """
        Pre-allocate pinned host memory and device buffers.
        Call once before the streaming loop.
        """
        n = 2 if self.double_buffer else 1

        # Pinned (page-locked) memory — enables DMA, bypasses OS paging
        if self.device.type == "cuda":
            self._pinned = torch.zeros(self._shape, dtype=self.dtype).pin_memory()
        else:
            self._pinned = torch.zeros(self._shape, dtype=self.dtype)

        # Device buffers
        self._gpu_bufs = [
            torch.zeros(self._shape, dtype=self.dtype, device=self.device)
            for _ in range(n)
        ]

        # Per-buffer CUDA streams
        if self.device.type == "cuda":
            self._streams = [torch.cuda.Stream(device=self.device) for _ in range(n)]
        else:
            self._streams = [None] * n

        self._allocated = True
        bytes_total = (
            self._pinned.nelement() * self._pinned.element_size() +
            sum(b.nelement() * b.element_size() for b in self._gpu_bufs)
        )
        print(f"[GPUMemory] Allocated {bytes_total/1024:.1f} KB "
              f"({'double' if self.double_buffer else 'single'} buffer, "
              f"device={self.device}, shape={self._shape})")

    def push(self, data: np.ndarray) -> torch.Tensor:
        """
        Transfer a numpy voxel grid to GPU (async if CUDA available).

        Args:
            data: numpy array, shape matching self._shape

        Returns:
            torch.Tensor on self.device — ready for CUDA kernel launch
        """
        if not self._allocated:
            self.allocate()

        t0   = time.perf_counter()
        slot = self._buf_idx % len(self._gpu_bufs)
        buf  = self._gpu_bufs[slot]
        stream = self._streams[slot]

        # Copy numpy → pinned host buffer
        self._pinned.copy_(torch.from_numpy(data))

        # Async H2D transfer on this slot's stream
        if self.device.type == "cuda" and stream is not None:
            with torch.cuda.stream(stream):
                buf.copy_(self._pinned, non_blocking=True)
            # Caller can overlap with computation on default stream;
            # insert an event if strict ordering needed
        else:
            buf.copy_(self._pinned)

        self._buf_idx += 1

        dt_ms = (time.perf_counter() - t0) * 1000
        nbytes = data.nbytes
        self._stats["transfers"]   += 1
        self._stats["total_bytes"] += nbytes
        self._stats["total_ms"]    += dt_ms

        return buf

    def push_batch(self, data: np.ndarray) -> torch.Tensor:
        """
        Transfer a batched voxel tensor (B, pol, T, H, W) to GPU.
        Allocates a one-off device tensor (no pre-alloc benefit).
        """
        t = torch.from_numpy(data)
        if self.device.type == "cuda":
            t_pinned = t.pin_memory()
            return t_pinned.to(self.device, non_blocking=True)
        return t.to(self.device)

    def synchronize(self):
        """Wait for all pending CUDA transfers to complete."""
        if self.device.type == "cuda":
            for s in self._streams:
                if s:
                    s.synchronize()

    def free(self):
        """Release GPU buffers."""
        self._gpu_bufs.clear()
        self._pinned = None
        self._allocated = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print("[GPUMemory] Buffers freed.")

    def stats(self) -> dict:
        n = self._stats["transfers"]
        total_mb = self._stats["total_bytes"] / 1e6
        avg_ms   = self._stats["total_ms"] / max(n, 1)
        bw_gbps  = (self._stats["total_bytes"] / max(self._stats["total_ms"]/1000, 1e-9)) / 1e9
        return {
            "transfers":    n,
            "total_mb":     total_mb,
            "avg_ms":       avg_ms,
            "bandwidth_gbps": bw_gbps,
        }


# ---------------------------------------------------------------------------
# EventTensorCache — device-side frame cache
# ---------------------------------------------------------------------------

class EventTensorCache:
    """
    Maintains a rolling cache of the last N GPU tensors.
    Useful for models that need temporal context across windows.

    Args:
        capacity: number of windows to keep on GPU
        shape:    tensor shape (pol, bins, H, W)
        device:   torch device
    """

    def __init__(self, capacity: int = 8,
                 shape: tuple = (2, 5, 260, 346),
                 device: str  = "cuda:0"):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")
        self.capacity = capacity
        dev           = torch.device(device if torch.cuda.is_available() else "cpu")
        # Pre-allocate (capacity, *shape)
        self._buf   = torch.zeros((capacity, *shape), dtype=torch.float32, device=dev)
        self._idx   = 0
        self._count = 0

    def push(self, tensor: torch.Tensor):
        """Add a tensor to the cache (overwrites oldest)."""
        slot = self._idx % self.capacity
        self._buf[slot] = tensor
        self._idx   += 1
        self._count  = min(self._count + 1, self.capacity)

    def get_sequence(self) -> torch.Tensor:
        """
        Return a (T, pol, bins, H, W) tensor of the last T cached frames
        in chronological order.
        """
        n = self._count
        if n == 0:
            return self._buf[:0]
        slots = [(self._idx - n + i) % self.capacity for i in range(n)]
        return self._buf[slots]

    @property
    def size(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 4: GPU Memory Demo ===\n")

    info = gpu_info()
    print("GPU info:", info)
    print(f"Best device: {get_best_device()}\n")

    # Simulate a stream of voxel grids from Module 3
    sensor_size = (346, 260)
    n_bins      = 5
    shape       = (2, n_bins, sensor_size[1], sensor_size[0])

    gm = GPUMemoryManager(
        device       = get_best_device(),
        sensor_size  = sensor_size,
        n_bins       = n_bins,
        double_buffer = True,
    )
    gm.allocate()

    cache = EventTensorCache(
        capacity = 8,
        shape    = shape,
        device   = get_best_device(),
    )

    print(f"\n{'Frame':>6}  {'Shape':>20}  {'Device':>10}  {'ms':>8}")
    print("-" * 55)

    for i in range(12):
        # Simulate voxel grid from Module 3
        dummy = np.random.rand(*shape).astype(np.float32)

        gpu_t = gm.push(dummy)
        cache.push(gpu_t)

        print(f"{i+1:>6}  {str(tuple(gpu_t.shape)):>20}  "
              f"{str(gpu_t.device):>10}  "
              f"{gm.stats()['avg_ms']:>8.3f}")

    gm.synchronize()
    s = gm.stats()
    print(f"\nTransfer stats:")
    print(f"  Transfers:   {s['transfers']}")
    print(f"  Total data:  {s['total_mb']:.2f} MB")
    print(f"  Avg latency: {s['avg_ms']:.3f} ms")
    print(f"  Bandwidth:   {s['bandwidth_gbps']:.2f} GB/s")

    seq = cache.get_sequence()
    print(f"\nCache sequence shape: {tuple(seq.shape)}  (last {cache.size} frames)")

    gm.free()
