"""
Probes RAM, GPU, and disk availability for cache/worker decisions.
"""
from __future__ import annotations
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch

logger = logging.getLogger(__name__)

GPU_PRESSURE_THRESHOLD = 0.75


@dataclass
class CacheMetrics:
    """Snapshot of system resource availability."""
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    disk_available_gb: float
    disk_exists: bool
    gpu_memory_gb: float
    gpu_available_gb: float


def gpu_pressure(metrics: CacheMetrics) -> float:
    """VRAM utilisation as 0–1. 0 when no GPU is present."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)


def is_gpu_under_pressure(metrics: CacheMetrics, threshold: float = GPU_PRESSURE_THRESHOLD) -> bool:
    return gpu_pressure(metrics) > threshold


class SystemResourceMonitor:
    """Probes RAM/GPU/disk on demand. Used by data_pipeline.dataloader_config()
    and AdaptiveCacheController before every caching decision."""

    def __init__(self, cache_path: str = "./cache", device_idx: int = 0, cuda_enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.device_idx = device_idx
        # False forces GPU fields to 0 even when CUDA is physically present —
        # set when the run is explicitly CPU-only, so a GPU that exists but
        # isn't requested never affects cache/worker decisions.
        self.cuda_enabled = cuda_enabled

    def snapshot(self) -> CacheMetrics:
        vm = psutil.virtual_memory()

        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024 ** 3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False

        if self.cuda_enabled and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device_idx)
            gpu_total = props.total_memory / (1024 ** 3)

            # Most conservative available-VRAM estimate: driver-reported free
            # space vs. total minus PyTorch's own reserved pool, whichever is smaller.
            free_driver, _ = torch.cuda.mem_get_info(self.device_idx)
            reserved = torch.cuda.memory_reserved(self.device_idx)
            gpu_available = min(free_driver, props.total_memory - reserved) / (1024 ** 3)
        else:
            gpu_total     = 0.0
            gpu_available = 0.0

        return CacheMetrics(
            total_ram_gb=vm.total / (1024 ** 3),
            available_ram_gb=vm.available / (1024 ** 3),
            ram_usage_percent=vm.percent,
            disk_available_gb=disk_available,
            disk_exists=disk_exists,
            gpu_memory_gb=gpu_total,
            gpu_available_gb=gpu_available,
        )
