"""
System Resource Monitor — centralized RAM, GPU, and disk probing.

Exports:
    CacheMetrics            — snapshot dataclass
    SystemResourceMonitor   — probes RAM, GPU, disk
    gpu_pressure()          — VRAM utilisation fraction from a snapshot
    is_gpu_under_pressure() — threshold check on a snapshot
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
    """Snapshot of system resource availability for cache decision-making."""
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    disk_available_gb: float
    disk_exists: bool
    gpu_memory_gb: float
    gpu_available_gb: float


def gpu_pressure(metrics: CacheMetrics) -> float:
    """VRAM utilisation as 0–1 derived from a CacheMetrics snapshot. Returns 0 if no GPU."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)


def is_gpu_under_pressure(metrics: CacheMetrics, threshold: float = GPU_PRESSURE_THRESHOLD) -> bool:
    """True when GPU VRAM usage exceeds threshold (default 75%)."""
    return gpu_pressure(metrics) > threshold


class SystemResourceMonitor:
    """
    Centralized probe for RAM, GPU, and disk availability.

    Both PipelineMemoryCoordinator and AdaptiveCacheController call snapshot()
    to get a current view of system resources before making caching decisions.
    """

    def __init__(self, cache_path: str = "./cache", device_idx: int = 0):
        self.cache_path = Path(cache_path)
        self.device_idx = device_idx

    def snapshot(self) -> CacheMetrics:
        """Probe live system state and return a CacheMetrics snapshot."""
        vm = psutil.virtual_memory()

        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024 ** 3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False

        gpu_total = 0.0
        gpu_available = 0.0
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device_idx)
            gpu_total = props.total_memory / (1024 ** 3)

            # Three views of VRAM — take the most conservative to avoid OOM surprises:
            # - memory_allocated : PyTorch tensors only (underestimates real usage)
            # - memory_reserved  : PyTorch allocator pool (includes fragmentation slack)
            # - mem_get_info     : CUDA driver level — captures other processes, context overhead
            free_driver, _ = torch.cuda.mem_get_info(self.device_idx)
            reserved = torch.cuda.memory_reserved(self.device_idx)
            # driver free is the hard ceiling; total-reserved is PyTorch's own view of headroom
            gpu_available = min(free_driver, props.total_memory - reserved) / (1024 ** 3)

        return CacheMetrics(
            total_ram_gb=vm.total / (1024 ** 3),
            available_ram_gb=vm.available / (1024 ** 3),
            ram_usage_percent=vm.percent,
            disk_available_gb=disk_available,
            disk_exists=disk_exists,
            gpu_memory_gb=gpu_total,
            gpu_available_gb=gpu_available,
        )
