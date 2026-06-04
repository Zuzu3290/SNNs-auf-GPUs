# MTBF and MTBI — What They Are and How We Use Them

## Definitions

**MTBF — Mean Time Between Failures**

The average time the system operates successfully between unrecoverable failures.

```
MTBF = Total operational time / Number of failures
```

A *failure* is an event that stops the training run entirely and requires manual intervention:
- Unhandled Python exception during training
- GPU out-of-memory that cannot be recovered
- Kernel import failure that prevents any forward pass
- CUDA driver error

A high MTBF means the system is stable. If MTBF is infinite, no failures have occurred across all recorded sessions.

---

**MTBI — Mean Time Between Interruptions**

The average time between any disruption to normal operation — including both failures and recoverable events.

```
MTBI = Total operational time / Number of interruptions
```

An *interruption* is any event that deviates from the expected operating path, even if training continues:
- GPU memory pressure above 90% peak
- MemoryArbiter refusing a kernel workspace request
- Kernel falling back to the non-accelerated path (snn_forward not found)
- Spike rate anomaly (abnormal neural activity detected)

MTBI is always ≤ MTBF. A gap between them reveals how often the system is disrupted without fully failing.

---

## Why Both Matter

| Metric | What it tells you |
|---|---|
| MTBF | How often training crashes completely |
| MTBI | How often hardware or memory pressure is affecting the run |
| MTBF − MTBI gap | How much silent degradation is happening that isn't a crash |

If MTBF is high but MTBI is low, training is completing but frequently hitting resource pressure. That means results are valid but performance is inconsistent — the kind of instability that shows up as training variance or inconsistent convergence speed.

---

## In This Project

`skeleton/reliability.py` — `ReliabilityTracker`

Sessions are tracked from the start of `SNNTrainer.train()` to its completion. Events are recorded as they happen. The history persists across multiple runs in `outputs/data/reliability.json`, so MTBF and MTBI improve in accuracy with each session.

At the end of every training run the report prints:

```
[ReliabilityTracker] Summary
  Sessions recorded    : 5
  Total operational    : 12.40 min
  Failures             : 0
  Interruptions        : 2
  MTBF                 : ∞
  MTBI                 : 6.20 min
  Recent interruptions :
    - GPU memory pressure — epoch 1 peak 91% of VRAM
    - kernel import failed — falling back to framework forward
```

This maps directly to the NVIDIA infrastructure reliability engineering requirement: tracking, calculating, and reporting MTBF and MTBI to drive infrastructure improvements.
