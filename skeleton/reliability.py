# MTBF / MTBI reliability tracker for the SNN training pipeline.
# Persists session history to a JSON file so metrics accumulate across runs.
#
# MTBF  = total operational time / number of failures
# MTBI  = total operational time / number of interruptions
#
# Failure     : unrecoverable crash — training stops (OOM, unhandled exception, kernel error)
# Interruption: recoverable disruption — training continues (memory warning, kernel fallback,
#               GPU pressure event, spike rate anomaly)
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class ReliabilityEvent:
    kind:      str    # "failure" | "interruption"
    reason:    str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    started_at:   float
    ended_at:     float | None = None
    outcome:      str = "running"  # "success" | "failure" | "interrupted"
    events:       List[ReliabilityEvent] = field(default_factory=list)
    operational_s: float = 0.0  # filled on session end


class ReliabilityTracker:
    """
    Tracks MTBF and MTBI across training sessions.

    Usage::

        rt = ReliabilityTracker()
        rt.start_session()

        # during training
        rt.record_interruption("memory arbiter refused kernel_workspace 256 MB")
        rt.record_failure("OOM during backward pass")

        rt.end_session(outcome="failure")
        rt.print_report()
    """

    DEFAULT_PATH = "./outputs/data/reliability.json"

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path        = path
        self.current:    Session | None = None
        self.history:    List[Session]  = []
        self._load()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        self.current = Session(started_at=time.monotonic())

    def end_session(self, outcome: str = "success") -> None:
        if self.current is None:
            return
        self.current.ended_at     = time.monotonic()
        self.current.outcome      = outcome
        self.current.operational_s = self.current.ended_at - self.current.started_at
        self.history.append(self.current)
        self.current = None
        self._save()

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_failure(self, reason: str) -> None:
        """Record an unrecoverable failure — training stops."""
        ev = ReliabilityEvent(kind="failure", reason=reason)
        if self.current is not None:
            self.current.events.append(ev)
        print(f"[Reliability] FAILURE recorded: {reason}")

    def record_interruption(self, reason: str) -> None:
        """Record a recoverable interruption — training continues."""
        ev = ReliabilityEvent(kind="interruption", reason=reason)
        if self.current is not None:
            self.current.events.append(ev)
        print(f"[Reliability] INTERRUPTION recorded: {reason}")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _all_events(self) -> List[ReliabilityEvent]:
        events = []
        for s in self.history:
            events.extend(s.events)
        if self.current:
            events.extend(self.current.events)
        return events

    def _total_operational_s(self) -> float:
        total = sum(s.operational_s for s in self.history)
        if self.current:
            total += time.monotonic() - self.current.started_at
        return total

    def mtbf(self) -> float:
        """Mean time between failures in seconds. Returns inf if no failures recorded."""
        failures = sum(1 for e in self._all_events() if e.kind == "failure")
        if failures == 0:
            return float("inf")
        return self._total_operational_s() / failures

    def mtbi(self) -> float:
        """Mean time between interruptions (failures + interruptions) in seconds."""
        count = len(self._all_events())
        if count == 0:
            return float("inf")
        return self._total_operational_s() / count

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        events     = self._all_events()
        failures   = [e for e in events if e.kind == "failure"]
        interrupts = [e for e in events if e.kind == "interruption"]
        op_s       = self._total_operational_s()

        print("\n[ReliabilityTracker] Summary")
        print(f"  Sessions recorded    : {len(self.history)}")
        print(f"  Total operational    : {op_s / 60:.2f} min")
        print(f"  Failures             : {len(failures)}")
        print(f"  Interruptions        : {len(interrupts)}")
        mtbf = self.mtbf()
        mtbi = self.mtbi()
        print(f"  MTBF                 : {'∞' if mtbf == float('inf') else f'{mtbf/60:.2f} min'}")
        print(f"  MTBI                 : {'∞' if mtbi == float('inf') else f'{mtbi/60:.2f} min'}")

        if failures:
            print("  Recent failures:")
            for e in failures[-3:]:
                print(f"    - {e.reason}")
        if interrupts:
            print("  Recent interruptions:")
            for e in interrupts[-3:]:
                print(f"    - {e.reason}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        data = [
            {
                "started_at":    s.started_at,
                "ended_at":      s.ended_at,
                "outcome":       s.outcome,
                "operational_s": s.operational_s,
                "events": [
                    {"kind": e.kind, "reason": e.reason, "timestamp": e.timestamp}
                    for e in s.events
                ],
            }
            for s in self.history
        ]
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for s in data:
                session = Session(
                    started_at    = s["started_at"],
                    ended_at      = s.get("ended_at"),
                    outcome       = s.get("outcome", "unknown"),
                    operational_s = s.get("operational_s", 0.0),
                    events        = [
                        ReliabilityEvent(
                            kind      = e["kind"],
                            reason    = e["reason"],
                            timestamp = e.get("timestamp", 0.0),
                        )
                        for e in s.get("events", [])
                    ],
                )
                self.history.append(session)
        except (json.JSONDecodeError, KeyError):
            self.history = []
