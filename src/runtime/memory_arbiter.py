# Pure-Python VRAM zone manager — fallback for environments where snn_runtime
# (the C++ cuMemPool extension) is not compiled. Mirrors the C++ API exactly
# so callers need no conditional logic. No cuMemPool API is available here.
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

import torch


class Zone(IntEnum):
    """Mirrors the SnnZone enum from the C++ extension — same integer values."""
    DATASET_CACHE    = 0
    MODEL_PARAMS     = 1
    KERNEL_WORKSPACE = 2
    EMERGENCY        = 3


DEFAULT_ZONE_RATIOS: Dict[Zone, float] = {
    Zone.DATASET_CACHE:    0.40,
    Zone.MODEL_PARAMS:     0.30,
    Zone.KERNEL_WORKSPACE: 0.20,
    Zone.EMERGENCY:        0.10,
}

HARD_ZONES = {Zone.MODEL_PARAMS, Zone.KERNEL_WORKSPACE, Zone.EMERGENCY}


@dataclass
class ZoneRecord:
    name:          str
    soft_limit_mb: float
    hard_limit_mb: float
    used_mb:       float = field(default=0.0)


class MemoryArbiter:
    """
    Pure-Python VRAM zone arbiter — fallback when snn_runtime is not compiled.
    Tracks zone usage with a threading.Lock; no cuMemPool API access.

    Usage::

        arb = MemoryArbiter.get()
        ok  = arb.request(Zone.KERNEL_WORKSPACE, mb=256)
        arb.release(Zone.KERNEL_WORKSPACE, mb=256)
        arb.print_status()
    """

    registry: Dict[int, "MemoryArbiter"] = {}
    class_lock = threading.Lock()
    pending_ratios: Dict[Zone, float] = {}
    pending_overhead_mb: int = 512

    def __init__(self, device_idx: int = 0,
                 zone_ratios: Dict[Zone, float] | None = None,
                 overhead_mb: int = 512) -> None:
        self.device_idx = device_idx
        self.lock       = threading.Lock()

        ratios = zone_ratios if zone_ratios else DEFAULT_ZONE_RATIOS
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        budget_mb = (total_bytes - overhead_mb * 1024 * 1024) / (1024.0 * 1024.0)

        self.zones: Dict[Zone, ZoneRecord] = {}
        for z, ratio in ratios.items():
            soft = budget_mb * ratio
            self.zones[z] = ZoneRecord(
                name          = z.name.lower(),
                soft_limit_mb = soft,
                hard_limit_mb = soft * 0.85,
            )

    @classmethod
    def configure(cls, dataset_ratio: float, model_ratio: float,
                  workspace_ratio: float, emergency_ratio: float,
                  overhead_mb: int = 512) -> None:
        """Call before get() to set ratios from SNN_module.yaml. No-op after first get()."""
        cls.pending_ratios = {
            Zone.DATASET_CACHE:    dataset_ratio,
            Zone.MODEL_PARAMS:     model_ratio,
            Zone.KERNEL_WORKSPACE: workspace_ratio,
            Zone.EMERGENCY:        emergency_ratio,
        }
        cls.pending_overhead_mb = overhead_mb

    @classmethod
    def get(cls, device_idx: int = 0) -> "MemoryArbiter":
        with cls.class_lock:
            if device_idx not in cls.registry:
                cls.registry[device_idx] = cls(
                    device_idx,
                    zone_ratios = cls.pending_ratios if cls.pending_ratios else None,
                    overhead_mb = cls.pending_overhead_mb,
                )
            return cls.registry[device_idx]

    def soft_limit_mb(self, zone: Zone) -> float:
        return self.zones[zone].soft_limit_mb

    def hard_limit_mb(self, zone: Zone) -> float:
        return self.zones[zone].hard_limit_mb

    def used_mb(self, zone: Zone) -> float:
        with self.lock:
            return self.zones[zone].used_mb

    def over_soft(self, zone: Zone) -> bool:
        with self.lock:
            z = self.zones[zone]
            return z.used_mb > z.soft_limit_mb

    def request(self, zone: Zone, mb: float) -> bool:
        rec = self.zones[zone]
        with self.lock:
            if zone in HARD_ZONES and rec.used_mb + mb > rec.hard_limit_mb:
                return False
            rec.used_mb += mb
            if rec.used_mb > rec.soft_limit_mb:
                warnings.warn(
                    f"[MemoryArbiter] {rec.name} soft limit exceeded: "
                    f"{rec.used_mb:.0f} MB > {rec.soft_limit_mb:.0f} MB",
                    stacklevel=2,
                )
        return True

    def release(self, zone: Zone, mb: float) -> None:
        with self.lock:
            rec = self.zones[zone]
            rec.used_mb = max(0.0, rec.used_mb - mb)

    # set_release_threshold / get_release_threshold_mb are no-ops in Python fallback
    def set_release_threshold(self, mb: float) -> None:
        pass

    def get_release_threshold_mb(self) -> float:
        return 0.0

    def stats(self) -> dict:
        with self.lock:
            zones = [
                {
                    "name":    rec.name,
                    "used_mb": rec.used_mb,
                    "soft_mb": rec.soft_limit_mb,
                    "hard_mb": rec.hard_limit_mb,
                    "pct":     rec.used_mb / rec.soft_limit_mb * 100.0
                               if rec.soft_limit_mb else 0.0,
                }
                for rec in self.zones.values()
            ]
        return {
            "zones":            zones,
            "pool_used_mb":     0.0,
            "pool_reserved_mb": 0.0,
            "pool_high_mb":     0.0,
        }

    def print_status(self) -> None:
        print(f"[MemoryArbiter] VRAM zones (device:{self.device_idx}, Python fallback)")
        for info in self.stats()["zones"]:
            bar = "#" * int(info["pct"] / 5)
            print(f"  {info['name']:<20} {info['used_mb']:>6.0f} / {info['soft_mb']:.0f} MB"
                  f"  ({info['pct']:.0f}%)  [{bar:<20}]")
