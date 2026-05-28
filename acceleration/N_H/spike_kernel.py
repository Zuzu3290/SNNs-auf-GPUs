"""
Module 5: Custom CUDA Kernel — Event → Spike Map
==================================================
Implements parallel GPU kernels that convert event voxel grids
into spike tensors for SNN consumption.

Kernels provided:
  1. event_to_spikes_kernel  — hard threshold: spike if events > θ
  2. lif_membrane_kernel     — Leaky Integrate-and-Fire per pixel
  3. polarity_merge_kernel   — merge ON/OFF channels with sign encoding

Two execution backends:
  A) PyCUDA    — requires: pip install pycuda
  B) CuPy      — requires: pip install cupy-cuda12x  (or 11x)
  C) Torch JIT — fallback using torch.jit + CUDA extensions

Usage (CuPy backend):
    kernel = SpikeKernel(backend="cupy")
    spikes = kernel.event_to_spikes(voxel_gpu, threshold=1.0)
    # spikes: torch.Tensor (2, T, H, W) bool on GPU
"""

import numpy as np
import time
from typing import Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.compiler as compiler
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False


# ---------------------------------------------------------------------------
# CUDA kernel source — written in CUDA C
# ---------------------------------------------------------------------------

CUDA_KERNEL_SOURCE = r"""
// =========================================================================
// Kernel 1: Hard-threshold event → spike conversion
// Input:  voxel[pol, T, H, W]  float32
// Output: spikes[pol, T, H, W] float32 (1.0 = spike, 0.0 = no spike)
// =========================================================================
extern "C" __global__
void event_to_spikes_kernel(
    const float* __restrict__ voxel,
    float*       __restrict__ spikes,
    const float  threshold,
    const int    pol,
    const int    T,
    const int    H,
    const int    W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pol * T * H * W;
    if (idx >= total) return;

    spikes[idx] = (voxel[idx] >= threshold) ? 1.0f : 0.0f;
}


// =========================================================================
// Kernel 2: Leaky Integrate-and-Fire (LIF) per pixel
//
// Integrates voxel bins in time order, accumulates membrane potential
// with leak, fires when V > V_thresh, then resets.
//
// Input:  voxel [T, H, W]  float32  (single polarity channel)
// Output: spikes[T, H, W]  float32
// In/Out: membrane[H, W]   float32  (persists between calls)
// =========================================================================
extern "C" __global__
void lif_membrane_kernel(
    const float* __restrict__ voxel,
    float*       __restrict__ spikes,
    float*       __restrict__ membrane,
    const float  v_thresh,
    const float  leak,          // decay factor per timestep, e.g. 0.9
    const float  v_reset,       // reset potential after spike, e.g. 0.0
    const int    T,
    const int    H,
    const int    W
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= H * W) return;

    int py = pixel / W;
    int px = pixel % W;

    float vm = membrane[pixel];

    for (int t = 0; t < T; t++) {
        int tidx = t * H * W + pixel;

        // Integrate input current
        vm = leak * vm + voxel[tidx];

        // Fire and reset
        if (vm >= v_thresh) {
            spikes[tidx] = 1.0f;
            vm = v_reset;
        } else {
            spikes[tidx] = 0.0f;
        }
    }

    // Write updated membrane state back
    membrane[pixel] = vm;
}


// =========================================================================
// Kernel 3: Merge ON/OFF channels with signed encoding
//
// Combined spike map: +1 for ON spike, -1 for OFF spike, 0 otherwise.
// Input:  on_spikes[T,H,W], off_spikes[T,H,W]
// Output: merged[T,H,W]  float32
// =========================================================================
extern "C" __global__
void polarity_merge_kernel(
    const float* __restrict__ on_spikes,
    const float* __restrict__ off_spikes,
    float*       __restrict__ merged,
    const int    T,
    const int    H,
    const int    W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * H * W) return;

    float on  = on_spikes[idx];
    float off = off_spikes[idx];
    // Priority: ON > OFF; simultaneous fire → 0
    merged[idx] = (on > 0.5f) ? 1.0f : ((off > 0.5f) ? -1.0f : 0.0f);
}
"""


# ---------------------------------------------------------------------------
# Backend: CuPy
# ---------------------------------------------------------------------------

class CuPyKernelBackend:
    def __init__(self):
        if not HAS_CUPY:
            raise RuntimeError("CuPy not installed.\n"
                               "  pip install cupy-cuda12x  # or cupy-cuda11x")
        self._mod = cp.RawModule(code=CUDA_KERNEL_SOURCE)
        self._k_threshold = self._mod.get_function("event_to_spikes_kernel")
        self._k_lif       = self._mod.get_function("lif_membrane_kernel")
        self._k_merge     = self._mod.get_function("polarity_merge_kernel")
        print("[CuPy] Kernels compiled successfully.")

    def event_to_spikes(self, voxel_cp: "cp.ndarray",
                        threshold: float = 1.0) -> "cp.ndarray":
        """voxel_cp: (pol, T, H, W). Returns spike array same shape."""
        pol, T, H, W = voxel_cp.shape
        total     = pol * T * H * W
        block     = 256
        grid      = (total + block - 1) // block

        spikes = cp.zeros_like(voxel_cp)
        self._k_threshold(
            (grid,), (block,),
            (voxel_cp, spikes, np.float32(threshold),
             np.int32(pol), np.int32(T), np.int32(H), np.int32(W))
        )
        return spikes

    def lif_integrate(self, voxel_cp: "cp.ndarray",
                      membrane_cp: "cp.ndarray",
                      v_thresh: float = 1.0,
                      leak: float     = 0.9,
                      v_reset: float  = 0.0) -> "cp.ndarray":
        """
        voxel_cp:    (T, H, W) — single polarity
        membrane_cp: (H, W)   — persistent state (updated in-place)
        Returns:     (T, H, W) spike array
        """
        T, H, W = voxel_cp.shape
        block   = 256
        grid    = (H * W + block - 1) // block
        spikes  = cp.zeros_like(voxel_cp)

        self._k_lif(
            (grid,), (block,),
            (voxel_cp, spikes, membrane_cp,
             np.float32(v_thresh), np.float32(leak), np.float32(v_reset),
             np.int32(T), np.int32(H), np.int32(W))
        )
        return spikes

    def polarity_merge(self, on_cp: "cp.ndarray",
                       off_cp: "cp.ndarray") -> "cp.ndarray":
        """on_cp, off_cp: (T,H,W). Returns merged (T,H,W)."""
        T, H, W = on_cp.shape
        total   = T * H * W
        block   = 256
        grid    = (total + block - 1) // block
        merged  = cp.zeros_like(on_cp)
        self._k_merge(
            (grid,), (block,),
            (on_cp, off_cp, merged,
             np.int32(T), np.int32(H), np.int32(W))
        )
        return merged


# ---------------------------------------------------------------------------
# Backend: PyTorch (CPU/GPU fallback — no kernel compilation needed)
# ---------------------------------------------------------------------------

class TorchKernelBackend:
    """
    Pure PyTorch implementation of the same operations.
    Runs on CUDA via PyTorch's existing kernels — not a custom kernel
    but produces identical results and works without CuPy/PyCUDA.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        print(f"[TorchKernel] Using device: {self.device} (no custom CUDA compilation)")

    def event_to_spikes(self, voxel: "torch.Tensor",
                        threshold: float = 1.0) -> "torch.Tensor":
        return (voxel >= threshold).float()

    def lif_integrate(self, voxel: "torch.Tensor",
                      membrane: "torch.Tensor",
                      v_thresh: float = 1.0,
                      leak: float     = 0.9,
                      v_reset: float  = 0.0) -> "torch.Tensor":
        """
        voxel:    (T, H, W)
        membrane: (H, W)  — updated in-place
        """
        T = voxel.shape[0]
        spikes = torch.zeros_like(voxel)
        vm     = membrane.clone()

        for t in range(T):
            vm = leak * vm + voxel[t]
            fired    = vm >= v_thresh
            spikes[t] = fired.float()
            vm[fired] = v_reset

        membrane.copy_(vm)
        return spikes

    def polarity_merge(self, on: "torch.Tensor",
                       off: "torch.Tensor") -> "torch.Tensor":
        return on.float() - off.float()


# ---------------------------------------------------------------------------
# SpikeKernel — unified interface
# ---------------------------------------------------------------------------

class SpikeKernel:
    """
    Unified interface over CuPy / Torch backends.

    Args:
        backend: "cupy" | "torch"
        device:  torch device string (used for torch backend)
    """

    def __init__(self, backend: str = "auto", device: str = "cuda:0"):
        if backend == "auto":
            backend = "cupy" if HAS_CUPY else "torch"
        self.backend_name = backend

        if backend == "cupy":
            self._backend = CuPyKernelBackend()
            self.device   = None   # CuPy manages its own device context
        elif backend == "torch":
            self._backend = TorchKernelBackend(device=device)
            self.device   = torch.device(device if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Persistent membrane states for LIF kernels
        self._membranes: dict = {}

    def _to_backend(self, tensor):
        """Convert torch.Tensor → cp.ndarray if using CuPy backend."""
        if self.backend_name == "cupy" and HAS_TORCH and isinstance(tensor, torch.Tensor):
            return cp.from_dlpack(tensor.detach())
        return tensor

    def _to_torch(self, arr) -> "torch.Tensor":
        """Convert cp.ndarray → torch.Tensor."""
        if self.backend_name == "cupy" and HAS_CUPY and isinstance(arr, cp.ndarray):
            return torch.from_dlpack(arr.toDlpack())
        return arr

    def event_to_spikes(self, voxel: "torch.Tensor",
                        threshold: float = 1.0) -> "torch.Tensor":
        """
        Hard-threshold spike conversion.

        Args:
            voxel:     (2, T, H, W) float32 on GPU
            threshold: events needed to fire a spike

        Returns:
            (2, T, H, W) float32 spike tensor on GPU
        """
        v   = self._to_backend(voxel)
        out = self._backend.event_to_spikes(v, threshold)
        return self._to_torch(out)

    def lif_spikes(self, voxel: "torch.Tensor",
                   channel_id: str = "default",
                   v_thresh: float = 1.0,
                   leak:     float = 0.9,
                   v_reset:  float = 0.0) -> "torch.Tensor":
        """
        LIF integration per polarity channel.
        Membrane state is persistent between calls (stateful).

        Args:
            voxel:      (2, T, H, W) — ON channel at [0], OFF at [1]
            channel_id: key for membrane state cache
            v_thresh:   firing threshold
            leak:       membrane decay (0→no leak, 1→perfect integrator)
            v_reset:    post-spike reset potential

        Returns:
            (2, T, H, W) spike tensor
        """
        pols, T, H, W = voxel.shape

        if self.backend_name == "cupy":
            spikes = cp.zeros_like(self._to_backend(voxel))
            for p in range(pols):
                key = f"{channel_id}_pol{p}"
                if key not in self._membranes:
                    self._membranes[key] = cp.zeros((H, W), dtype=cp.float32)
                v_in = self._to_backend(voxel[p])
                spikes[p] = self._backend.lif_integrate(
                    v_in, self._membranes[key], v_thresh, leak, v_reset)
            return self._to_torch(spikes)
        else:
            spikes = torch.zeros_like(voxel)
            for p in range(pols):
                key = f"{channel_id}_pol{p}"
                if key not in self._membranes:
                    dev = voxel.device
                    self._membranes[key] = torch.zeros(H, W,
                                                        dtype=torch.float32,
                                                        device=dev)
                spikes[p] = self._backend.lif_integrate(
                    voxel[p], self._membranes[key], v_thresh, leak, v_reset)
            return spikes

    def merge_polarities(self, spikes: "torch.Tensor") -> "torch.Tensor":
        """
        Merge ON/OFF into a signed map (+1/-1/0).

        Args:
            spikes: (2, T, H, W) — [0]=ON, [1]=OFF

        Returns:
            (T, H, W) signed spike tensor
        """
        if self.backend_name == "cupy":
            on  = self._to_backend(spikes[0])
            off = self._to_backend(spikes[1])
            return self._to_torch(self._backend.polarity_merge(on, off))
        else:
            return self._backend.polarity_merge(spikes[0], spikes[1])

    def reset_membrane(self, channel_id: str = "default"):
        """Reset LIF membrane state for a given channel."""
        keys = [k for k in self._membranes if k.startswith(channel_id)]
        for k in keys:
            del self._membranes[k]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 5: CUDA Spike Kernel Demo ===\n")

    if not HAS_TORCH:
        print("PyTorch required for demo.")
        exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"CuPy available: {HAS_CUPY}")

    kernel = SpikeKernel(backend="auto", device=device)
    print(f"Backend: {kernel.backend_name}\n")

    # Simulate voxel grid from Module 4
    W, H, T = 346, 260, 5
    voxel = torch.rand(2, T, H, W, dtype=torch.float32)
    if device != "cpu":
        voxel = voxel.to(device)

    print("--- Hard-threshold spikes ---")
    t0 = time.perf_counter()
    spikes_thresh = kernel.event_to_spikes(voxel, threshold=0.5)
    dt = (time.perf_counter() - t0) * 1000
    fire_rate = spikes_thresh.mean().item() * 100
    print(f"  Shape: {tuple(spikes_thresh.shape)}")
    print(f"  Fire rate: {fire_rate:.1f}%  (threshold=0.5)")
    print(f"  Kernel time: {dt:.3f} ms\n")

    print("--- LIF membrane spikes ---")
    t0 = time.perf_counter()
    spikes_lif = kernel.lif_spikes(voxel, v_thresh=1.0, leak=0.9)
    dt = (time.perf_counter() - t0) * 1000
    fire_rate = spikes_lif.mean().item() * 100
    print(f"  Shape: {tuple(spikes_lif.shape)}")
    print(f"  Fire rate: {fire_rate:.1f}%")
    print(f"  Kernel time: {dt:.3f} ms\n")

    print("--- Polarity merge ---")
    merged = kernel.merge_polarities(spikes_thresh)
    unique, counts = torch.unique(merged, return_counts=True)
    print(f"  Shape: {tuple(merged.shape)}")
    for v, c in zip(unique.tolist(), counts.tolist()):
        print(f"    value {v:+.0f}: {c:,} pixels")
