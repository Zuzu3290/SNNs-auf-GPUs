"""
Python interface to the GPU event preprocessing kernels.

Replaces the CPU tonic pipeline (Denoise + ToFrame) for the GPU-only
configuration. The CPU+GPU hybrid path in event_data_workflow/ is unchanged.

Requires the CUDA extension to be built:
    cd src/gpu_kernel_implementation
    python build.py

Usage:
    from gpu_kernel_implementation.event_preprocessing import GPUEventPreprocessor

    pre = GPUEventPreprocessor(H=34, W=34, n_time_bins=16)
    frame = pre.process(events_numpy, device="cuda")  # [2, T, H, W] float32
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional


class GPUEventPreprocessor:
    """
    Converts a raw tonic-style numpy event array to a frame tensor entirely
    on GPU — no CPU tonic transforms involved.

    Parameters
    ----------
    H, W         : Sensor height and width in pixels.
    n_time_bins  : Number of temporal bins in the output frame.
    filter_time_us: Denoise filter window in microseconds (matches tonic default).
    search_window : Max events to scan left/right per event during denoise.
    """

    def __init__(
        self,
        H: int,
        W: int,
        n_time_bins: int = 16,
        filter_time_us: int = 10_000,
        search_window: int = 200,
    ):
        self.H              = H
        self.W              = W
        self.n_time_bins    = n_time_bins
        self.filter_time_us = filter_time_us
        self.search_window  = search_window
        self._ops           = self._load_ops()

    def _load_ops(self):
        try:
            import snn_gpu_preproc as ops  # type: ignore[import]
            return ops
        except ImportError:
            return None

    def process(
        self,
        events: np.ndarray,    # structured array with fields x, y, t, p
        device: str = "cuda",
    ) -> Optional[torch.Tensor]:
        """
        Run GPU denoise + ToFrame on a raw event array.

        Returns [2, n_time_bins, H, W] float32 tensor on `device`,
        or None if the CUDA extension is not built.
        """
        if self._ops is None:
            raise RuntimeError(
                "[GPUEventPreprocessor] CUDA extension not built. "
                "Run: python src/gpu_kernel_implementation/build.py"
            )

        if events is None or len(events) == 0:
            return torch.zeros(
                2, self.n_time_bins, self.H, self.W,
                dtype=torch.float32, device=device
            )

        ev_x = torch.from_numpy(events["x"].astype(np.int16)).to(device)
        ev_y = torch.from_numpy(events["y"].astype(np.int16)).to(device)
        ev_t = torch.from_numpy(events["t"].astype(np.int64)).to(device)
        ev_p = torch.from_numpy(events["p"].astype(np.int8)).to(device)

        t_start = int(ev_t[0].item())
        t_end   = int(ev_t[-1].item())
        if t_end <= t_start:
            t_end = t_start + 1

        keep_mask = self._ops.gpu_denoise_events(
            ev_x, ev_y, ev_t,
            self.filter_time_us,
            self.search_window,
        )

        frame = self._ops.gpu_events_to_frame(
            ev_x, ev_y, ev_t, ev_p, keep_mask,
            self.H, self.W, self.n_time_bins,
            t_start, t_end,
        )

        return frame
