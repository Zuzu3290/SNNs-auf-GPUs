# Additional Frameworks — Sinabs

A fourth SNN backend beyond the original three (Norse, snnTorch, SpikingJelly),
added to give the GPU benchmarking work a DVS-native framework shape to compare.
Switch to it the same way as the original three: set `training.framework: sinabs`
in `configuration/SNN_module.yaml`.

| `sinabs` | `SNN_SINABS` | `frameworks/snn_sinabs.py` | Verified end-to-end on GPU |

"Verified end-to-end" means: instantiated through the real `Settings()` config pipeline, ran a real forward pass at production `BATCH_SIZE`/sequence length, confirmed
the loss computes, confirmed weights actually change after one training step, and confirmed eval mode works.

`training.framework` accepts `norse`/`torch`/`sj`/`sinabs` — `main.py`'s `MODELS`
dispatch wires up exactly these four.

---

## Sinabs

PyTorch-based, but DVS-first: every spiking layer is **batch-first** `(B, T, ...)`
rather than the time-first `(T, B, ...)` layout the other backends use — hence
`tensor_format()` returns `"BT"` for this one, the first real use of that branch
in `ModelInterface`. It also ships a genuine export path to SynSense's Speck chip,
relevant if this project's DVS pipeline ever needs an edge-deployment story.

Trains via standard PyTorch autograd — `is_differentiable()` is `True`, same as the
original three. Conv2d/MaxPool2d only understand 4D tensors, so the batch and time
dims are flattened before each conv stage and unflattened before each LIF stage —
see `SNN_SINABS.forward()`.

