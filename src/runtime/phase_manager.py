# Training phase coordinator — single source of truth for the current execution phase.
# Any component (memory arbiter, cache engine, kernel dispatcher) can read the phase
# without importing from learning/, event_data_workflow/, or acceleration/.
from __future__ import annotations

import threading
import time
from enum import Enum


class Phase(Enum):
    IDLE     = "idle"
    WARMUP   = "warmup"
    TRAIN    = "train"
    BACKWARD = "backward"
    EVAL     = "eval"


class PhaseManager:
    """
    Singleton phase coordinator.

    Usage::

        pm = PhaseManager.get()
        pm.enter(Phase.TRAIN)
        if pm.is_training():
            ...
    """

    singleton: "PhaseManager | None" = None
    class_lock = threading.Lock()

    def __init__(self) -> None:
        self.phase      = Phase.IDLE
        self.entered_at = time.monotonic()
        self.lock       = threading.Lock()

    @classmethod
    def get(cls) -> "PhaseManager":
        with cls.class_lock:
            if cls.singleton is None:
                cls.singleton = cls()
            return cls.singleton

    def enter(self, phase: Phase) -> None:
        with self.lock:
            if self.phase == phase:
                return
            self.phase      = phase
            self.entered_at = time.monotonic()

    @property
    def current(self) -> Phase:
        return self.phase

    def is_training(self) -> bool:
        return self.phase in (Phase.TRAIN, Phase.BACKWARD)

    def elapsed_s(self) -> float:
        """Seconds spent in the current phase."""
        return time.monotonic() - self.entered_at

    def __repr__(self) -> str:
        return f"PhaseManager(phase={self.phase.value}, elapsed={self.elapsed_s():.2f}s)"
