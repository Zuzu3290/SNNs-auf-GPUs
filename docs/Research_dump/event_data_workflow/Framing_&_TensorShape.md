# Temporal Framing and Tensor Format

This document covers how raw neuromorphic events are converted into fixed-size tensors, and how the tensor layout is kept consistent from the DataLoader through to the model.

---

## Event Framing

Raw N-MNIST data consists of sparse asynchronous events `(x, y, polarity, timestamp)`. Before training, these must be converted into dense frame tensors using `tonic.transforms.ToFrame`.

Two modes are available, configured in `configuration/data_workflow.yaml`:

```yaml
framing:
  mode: n_time_bins    # or: time_window
  n_time_bins: 16
  time_window_ms: 15.0
```

| Mode | Behaviour | Use case |
|---|---|---|
| `n_time_bins` | Each recording is divided into exactly T bins regardless of duration | Training — fixed T per sample, required for batching |
| `time_window` | Each bin covers a fixed duration in microseconds; T varies per recording | Analysis — preserves real-time structure |

`n_time_bins` is the standard choice for Conv-SNN training because the DataLoader requires all samples in a batch to have the same T dimension.

---

## Canonical Tensor Shape: `[T, B, C, H, W]`

The DataLoader always outputs tensors in **time-first** format:

```
T — timesteps (number of time bins, e.g. 16)
B — batch size
C — polarity channels (2 for N-MNIST: ON / OFF events)
H — sensor height (34)
W — sensor width  (34)
```

This is set by `tonic.collation.PadTensors(batch_first=False)` in the pipeline. The pipeline has no knowledge of which framework is running — it always produces `[T, B, C, H, W]`.

---

## Per-Framework Tensor Adaptation

Different frameworks may expect different tensor layouts in their `forward()` method. Rather than making the pipeline framework-aware, the adaptation happens in `training.py` right before the model receives each batch.

`ModelInterface` declares a `tensor_format()` method that defaults to `"TB"`:

```python
def tensor_format(self) -> str:
    """
    "TB" — [T, B, C, H, W]  time-first  (default — SNNTorch, Norse, SpikingJelly)
    "BT" — [B, T, C, H, W]  batch-first
    Only override if your framework needs batch-first input.
    """
    return "TB"
```

`SNNTrainer.forward_pass()` reads this and transposes if needed:

```python
if self.model.tensor_format() == "BT":
    data = data.permute(1, 0, 2, 3, 4).contiguous()
return self.model(data)
```

All three current frameworks use `"TB"` (the default), so no transpose is applied. A new framework that requires batch-first input only needs to override `tensor_format()` — no changes to the pipeline or training loop are needed.

---

## Why All Three Frameworks Use Time-First

- **SNNTorch** — `forward()` loops `for step in range(data.size(0))`, indexing dim-0 as T.
- **Norse** — `LIFCell` is called per timestep; the outer loop iterates over dim-0.
- **SpikingJelly** (current, single-step mode) — same manual T loop: `for step in range(data.size(0)): self.net(data[step])`. SpikingJelly never sees the T dimension directly.

If SpikingJelly is switched to multi-step mode (`step_mode='m'`), it would still expect `[T, B, ...]` in the `activation_based` API — so no layout change would be needed even then.

---

## Adding a New Framework

1. Implement `forward(data: torch.Tensor)` expecting whichever layout your framework needs.
2. If it differs from `[T, B, C, H, W]`, override `tensor_format()` in your class:
   ```python
   def tensor_format(self) -> str:
       return "BT"
   ```
3. The trainer adapts automatically. No pipeline changes required.
