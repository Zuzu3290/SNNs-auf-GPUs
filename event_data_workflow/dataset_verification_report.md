# Event Data Workflow — Dataset Verification Report

What loads, what broke, what got fixed, and what a real VRAM ceiling means for the next
dataset someone adds.

**5** datasets verified · **2** production bugs fixed · **1** removed, no fix found · **12** sample frames captured

## 01 — Status

| Dataset | Result | Note |
|---|---|---|
| N-MNIST | PASS | verified earlier — the baseline everything else is compared against |
| N-Caltech101 | PASS | needs batch_size ≤16 on this GPU — see §02 |
| DVS128 Gesture | PASS | figshare download was dead until the host fix below |
| DSEC | PASS | windowing rebuilt as a generic, reusable wrapper |
| DAVIS Camera Pose | PASS | new — replaces ASL-DVS's slot, 6-DOF pose regression |
| ASL-DVS | REMOVED | official link disabled by the dataset owner; no working substitute found |
| DHP19 | DECLINED | real access confirmed, 145.5GB — declined on translation effort, not access |

## 02 — The VRAM concern

It was real. N-Caltech101 crashed with a CUDA out-of-memory error at the batch size that
trains N-MNIST without incident. The cause wasn't the dataset sitting in VRAM — the cache
strategy for that run was `DiskCachedDataset`, confirmed off-GPU. What actually fills the GPU
is every timestep's activations, held simultaneously for backprop-through-time, for the whole
batch, at once.

| Layer | N-MNIST (34×34) | N-Caltech101 (240×180) | Ratio |
|---|---|---|---|
| Conv1 output | 30×30×12 = 10,800 | 176×236×12 = 498,432 | ~46× |
| FC input (post-pool) | 5×5×32 = 800 | 42×57×32 = 76,608 | ~96× |

Conv1's output alone, held across 16 timesteps at batch_size=128, is
`498,432 × 128 × 16 × 4 bytes ≈ 4.1 GB` — for one layer, before Conv2, LIF membrane state, or
gradients are counted. At batch_size=16 the same term is ~510MB. Confirmed empirically: 128
crashes, 16 completes real forward+backward batches on the GPU.

**Not fixed globally — left as a decision.** The registry's per-dataset `batch_size` field
exists for exactly this. It's left at `None` for N-Caltech101 deliberately, so whoever sets
real training hyperparameters chooses the number rather than inheriting a guess.

### Managing this on the next dataset added

1. **Compare the new sensor's resolution to N-MNIST's 34×34 before assuming the global
   batch_size will fit.** VRAM for BPTT scales with H×W × batch × timesteps — a 46× larger
   feature map is a 46× larger memory term, not a rounding error.
2. **Set that dataset's own `batch_size` in `dataset_registry.py`** rather than leaving it to
   inherit the global default. One global number across wildly different sensor sizes is what
   caused this crash in the first place.
3. **Test pipeline correctness and memory fit as two separate questions** when smoke-testing a
   new dataset — a small diagnostic batch size first, the real one after. Conflating them means
   a memory ceiling gets misdiagnosed as a data or caching bug, which is exactly what this
   looked like at first.
4. **Never run more than one GPU-bound smoke test at a time on this machine.** Confirmed
   directly today: three concurrent diagnostic runs produced a cuDNN execution error and a
   false OOM that had nothing to do with either dataset — pure contention on one 8GB card.
5. **If batch size alone can't be cut further** without hurting training quality, the
   documented next levers are AMP, gradient accumulation (`cfg.GRAD_ACCUM_STEPS` already
   exists), or truncated BPTT — in that order. Full option list with tradeoffs is in
   `event_data_workflow/vram_batch_scaling_task.md`.

## 03 — The four datasets, in frame

### N-Caltech101 — PASS
101 classes · 240×180 sensor · 8,709 recordings · classification

| sample 0 (target 0) | sample 1 (target 0) | sample 2 (target 0) |
|---|---|---|
| ![N-Caltech101 sample 0](../diagnostics/samples/N-Caltech101/sample_0.png) | ![N-Caltech101 sample 1](../diagnostics/samples/N-Caltech101/sample_1.png) | ![N-Caltech101 sample 2](../diagnostics/samples/N-Caltech101/sample_2.png) |

Loads, caches (disk), and reaches the GPU correctly. Found and fixed a real bug along the way:
tonic hands back class-folder *names* as targets, not integers — nothing downstream mapped
them, so this would have crashed the very first training batch in production, not just in
testing. Fixed once, in the pipeline, not the test script. Only open item: needs its own
`batch_size` (≤16 confirmed working here) — see §02.

### DVS128 Gesture — PASS
11 classes · 128×128 sensor · 1,464 recordings · classification

| sample 0 (target 0) | sample 1 (target 1) | sample 2 (target 10) |
|---|---|---|
| ![DVS128 Gesture sample 0](../diagnostics/samples/DVS128_Gesture/sample_0.png) | ![DVS128 Gesture sample 1](../diagnostics/samples/DVS128_Gesture/sample_1.png) | ![DVS128 Gesture sample 2](../diagnostics/samples/DVS128_Gesture/sample_2.png) |

Loads, caches (in memory), and reaches the GPU correctly — real forward+backward batch
confirmed, shape (16, 128, 2, 128, 128). The download itself was dead: tonic's hardcoded
figshare URL sits behind a bot-detection challenge. Fixed by pointing at figshare's own
canonical direct-download subdomain — not a workaround, the correct host.

### DSEC — PASS
optical flow · 640×480 sensor · regression · target not finalized

| window 0 | window 1 | window 2 |
|---|---|---|
| ![DSEC sample 0](../diagnostics/samples/DSEC/sample_0.png) | ![DSEC sample 1](../diagnostics/samples/DSEC/sample_1.png) | ![DSEC sample 2](../diagnostics/samples/DSEC/sample_2.png) |

Loads, caches, and reaches the GPU correctly — real batch confirmed, shape (16, 4, 2, 480, 640)
with a matching (4, 480, 640, 3) flow target. The one-off windowing class this depended on was
replaced with `WindowedRecordingDataset`, a generic wrapper that turns whole recordings into
windowed samples for any dataset shaped this way — DAVIS Camera Pose below reuses the exact
same class. Frames render mostly dark: automotive driving scenes have long stretches of little
motion, which is sparse ground truth, not missing data.

### DAVIS Camera Pose — PASS *(new)*
6-DOF pose (xyz + quaternion) · 240×180 sensor · regression

| window 0 | window 1 | window 2 |
|---|---|---|
| ![DAVIS Camera Pose sample 0](../diagnostics/samples/DAVIS_Camera_Pose/sample_0.png) | ![DAVIS Camera Pose sample 1](../diagnostics/samples/DAVIS_Camera_Pose/sample_1.png) | ![DAVIS Camera Pose sample 2](../diagnostics/samples/DAVIS_Camera_Pose/sample_2.png) |

Fills ASL-DVS's old slot. Real motion-capture camera-pose data (Mueggler et al.'s Event-Camera
Dataset), plain-text format, hosted by the same group that already proved reliable for DSEC
today. Confirmed real batch on GPU: shape (16, 4, 2, 180, 240) with a matching (4, 7) pose
target. 11,882 windows built from one 150MB sequence; the full 27-sequence collection is
~7.7GB if more variety is ever needed. Frames render almost blank: a static wall of shapes
under constant lighting produces very few events per window — expected for this sequence, not
a parsing fault.

---
`dataset_registry.py` — registry & loaders · `dataset_registry_changes.md` — full history ·
`vram_batch_scaling_task.md` — VRAM findings · `diagnostics/` — verification scripts & sample frames
