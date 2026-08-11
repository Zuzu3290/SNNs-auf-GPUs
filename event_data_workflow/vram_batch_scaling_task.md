# Open task — VRAM doesn't scale with sensor resolution (batch_size is per-dataset, not global)

Status: **open, not yet resolved.** Discovered while smoke-testing N-Caltech101 through the
real pipeline (`diagnostics/verify_dataset_load.py`). Filed here (not `docs/`) because it's an
`event_data_workflow` / training-config concern, not a written-up report.

## The concern, in one line

`SNN_module.yaml`'s `training.batch_size` is a single global number applied to every dataset
in `dataset_registry.py`. That's fine for datasets with a similar sensor resolution to
N-MNIST, but it silently breaks for a dataset with a much larger sensor — the same batch size
that trains N-MNIST fine causes a CUDA out-of-memory crash on N-Caltech101, on this machine's
8GB GPU.

## What's actually running on the GPU during training (and what isn't)

Three things could plausibly live in VRAM. Only one of them is the actual problem:

1. **Model weights** — a few MB. Not the issue.
2. **The dataset** — confirmed NOT the issue. The pipeline's cache strategy for N-Caltech101
   is `DiskCachedDataset` (see `AdaptiveCacheController` in `cache_engine.py`), which keeps
   cached recordings on disk/host RAM, not VRAM. Only the *current batch* ever gets
   transferred to the GPU (`data.to(device, ...)`).
3. **Per-timestep activations, held for backprop-through-time (BPTT)** — this is the actual
   cause. Every framework in this project trains SNNs with `credit_assignment() ==
   "BPTT+SG"` (surrogate-gradient BPTT): the forward pass runs the model once per timestep
   (`T=16` by default, `n_time_bins` in `data_workflow.yaml`), and autograd must keep *every*
   timestep's intermediate feature maps alive simultaneously, because backward() walks back
   through all of them. This is not "the dataset in VRAM" — it's T copies of the model's own
   output, for the whole batch, at once.

## N-MNIST vs. N-Caltech101 — why one is fine and the other isn't

Architecture is fixed in `configuration/network_architecture.yaml`: Conv1 (2→12ch, 5×5) →
pool 2×2 → Conv2 (12→32ch, 5×5) → pool 2×2 → FC. Only the input sensor size changes per
dataset (`dataset_registry.py`'s `sensor_size`, wired through `cfg.apply_dataset_shape()`).

| | N-MNIST (34×34) | N-Caltech101 (240×180) | Ratio |
|---|---|---|---|
| Conv1 output | 30×30×12 = 10,800 | 176×236×12 = 498,432 | ~46× |
| FC_IN (after both pools) | 5×5×32 = 800 | 42×57×32 = 76,608 | ~96× |

Conv1's output alone, held across all 16 timesteps for one batch, at `batch_size=128`
(today's global default):

```
498,432 elements × 128 (batch) × 16 (timesteps) × 4 bytes (fp32) ≈ 4.08 GB
```

— for one layer, before Conv2's activations, LIF membrane-potential state per timestep, or
backward-pass gradients are even counted. That's why it crashes on an 8GB card. At
`batch_size=16` the same layer is ~510MB, which fits comfortably — confirmed empirically:
`--batch-size 128` → `torch.OutOfMemoryError` inside `SNN_TORCH.forward()`;
`--batch-size 16` → 3 real batches complete forward+backward on `cuda:0`,
shape `(16, 16, 2, 180, 240)`.

## Why this matters beyond one crash — scalability / real-time

`configuration/data_workflow.yaml` already has a `realtime.deadline_ms` section keyed per
dataset (e.g. DVS128 Gesture's 105ms, from the original IBM paper). The moment training
config is meant to generalize across datasets/cameras of different resolutions — which a
real-time deployment story implies — a single global `batch_size` stops being a config value
and becomes a silent crash waiting for whichever dataset has the biggest sensor. This is the
same shape of problem `dataset_registry.py`'s per-dataset `epochs`/`batch_size`/`iterations`
fields were added to solve; N-Caltech101 is the first dataset that actually needs a value
there instead of `None`.

## Candidate techniques to resolve (not yet chosen/applied)

1. **Set a smaller per-dataset `batch_size` in `dataset_registry.py`** — simplest, no
   architecture or training-loop change. Empirically, 16 fits on this 8GB GPU for
   N-Caltech101; the real ceiling (and whether it's worth trading batch size for something
   else below) is a decision for whoever owns the hyperparameter values, not something to
   infer from one successful run.
2. **Mixed precision (AMP)** — `cfg.USE_AMP` already exists as a config toggle
   (`SNN_module.yaml`) but isn't confirmed wired through the BPTT loop for this path; fp16/bf16
   activations would roughly halve every number in the VRAM math above.
3. **Gradient accumulation** — `cfg.GRAD_ACCUM_STEPS` already exists. Run a smaller physical
   batch through the GPU, accumulate gradients over several micro-batches, then step the
   optimizer once — same effective batch size for training dynamics, lower peak VRAM.
4. **Truncated BPTT / gradient checkpointing** — don't keep all 16 timesteps' graphs live at
   once; recompute forward activations during backward instead of storing them. Trades extra
   compute time for a large memory cut, without touching batch size at all.
5. **An extra pool/stride stage (or input downsampling) for large-sensor datasets** — shrinks
   Conv1's output size directly, the single biggest term in the memory math above. Changes
   the effective architecture per dataset, so it's a bigger decision than the others.
6. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces allocator fragmentation
   (suggested directly in PyTorch's own OOM message); doesn't reduce the actual memory
   requirement, just how efficiently the allocator packs it, so it's a minor mitigation at
   best, not a fix on its own.

None of the above has been applied to the production pipeline (`data_pipeline.py`,
`learning/training.py`) — this file only records the finding and the option space so the
decision doesn't get lost.
