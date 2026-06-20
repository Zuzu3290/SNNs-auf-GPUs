# Additional Frameworks — Sinabs, BindsNET, Spyx, Lava-dl

Four more SNN backends beyond the original three (Norse, snnTorch, SpikingJelly),
added to give the GPU benchmarking work more than one framework "shape" to compare —
DVS-native, STDP-trained, JAX-based, and Loihi-targeted. Switch to any of them the
same way as the original three: set `training.framework` in `configuration/SNN_module.yaml`.

| Selector | Class | File | Status |
|---|---|---|---|
| `sinabs`   | `SNN_SINABS`   | `src/learning/frameworks/snn_sinabs.py`   | Verified end-to-end on GPU |
| `bindsnet` | `SNN_BINDSNET` | `src/learning/frameworks/snn_bindsnet.py` | Verified end-to-end on GPU |
| `spyx`     | `SNN_SPYX`     | `src/learning/frameworks/snn_spyx.py`     | Verified end-to-end on CPU (JAX has no Windows CUDA wheels) |
| `lava`     | `SNN_LAVA`     | `src/learning/frameworks/snn_lava.py`     | **Unverified** — written from memory, never run. See below. |

"Verified end-to-end" means: instantiated through the real `Settings()` config
pipeline, ran a real forward pass at production `BATCH_SIZE`/`TIMESTEPS`, confirmed
the loss computes, confirmed weights actually change after one training step, and
confirmed eval mode works.

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

---

## BindsNET

The odd one out: it doesn't train via backprop at all. Weights update through real
**STDP** (the `PostPre` rule — pre-before-post strengthens, post-before-pre weakens),
applied directly inside `Network.run()`. This project's `activity_reg.py` STDP loss
term only *approximates* that as a soft backprop-compatible penalty; BindsNET does
the real thing. `is_differentiable()` returns `False`, so the adversarial evaluator
skips attack generation for it — same pattern documented for JAX/TF in `README.md`.

### A real bug found in BindsNET 0.2.7 (not a config issue)

`MaxPool2dConnection.compute()` allocates its firing-rate buffer as
`torch.ones(source.shape)` — no batch dimension — then does
`self.firing_rates += s.float()` where `s` *does* have a batch dimension. That throws
a broadcast error for any `batch_size > 1`:

```
RuntimeError: output with shape [12, 30, 30] doesn't match the broadcast shape [4, 12, 30, 30]
```

Confirmed by direct reproduction against the installed package — this isn't a
Windows quirk or a missing kwarg. BindsNET's pooling connection is simply unusable
at batch sizes above 1 in the latest PyPI release (unmaintained since ~2022).

**Workaround used here:** skip `MaxPool2dConnection` entirely. The architecture is
split into 3 independently-trained `bindsnet.Network` instances (conv1→lif1,
conv2→lif2, fc→lif_out), chained by plain `F.max_pool2d` calls on the recorded spike
tensors between them. Each stage runs with `one_step=True` so a single timestep
propagates fully through that stage's one connection — matching how the other
backends compute a full conv→lif pass per timestep rather than per-layer. See
`_BindsNetStage` in `snn_bindsnet.py` for the full reasoning.

### The `torch._six` shim

BindsNET 0.2.7 imports `torch._six`, which PyTorch removed years ago. A 4-line shim
at the top of `snn_bindsnet.py` restores just enough of it (`container_abcs`,
`string_classes`, `int_classes`) for BindsNET's import chain to succeed. Narrow and
load-bearing — not a style choice.

---

## Spyx

The only backend here that isn't PyTorch — it's JAX + Haiku. Two consequences:

1. **Tensor boundary.** PyTorch tensors cross into JAX via DLPack
   (`jax.dlpack.from_dlpack(torch_tensor)` / `torch.from_dlpack(jax_array)`) — modern
   torch/jax support this directly via the `__dlpack__` protocol, no capsule dance
   needed. On this machine `jax.devices()` reports CPU-only (JAX has no Windows CUDA
   wheels), so the conversion goes through a CPU copy. Under the `container` branch's
   Linux Docker image with `pip install jax[cuda12]`, the same code gets a real
   zero-copy GPU→GPU handoff.
2. **Gradient boundary.** PyTorch autograd cannot trace through JAX/XLA, so
   `backward_pass()` is a no-op. Gradients are computed with `jax.value_and_grad`
   and applied via `optax` inside `loss_fn()` — the one place in the trainer's loop
   that has access to both the model's output and the batch's targets (`forward()`
   only receives `data`). See `docs/frameworks/README.md` → "JAX backend" for the
   pattern this follows.

**Known limitation:** the optax update fires on every micro-batch, so
`GRAD_ACCUM_STEPS > 1` is not honoured for this backend, unlike the PyTorch backends
which accumulate via `backward_pass(do_step=False)`.

A real bug was found and fixed while verifying this one too: `hk.max_pool`'s
`window_shape`/`strides` need a tuple with the **same rank as the tensor**
(`(1, k, k, 1)` for a 4D `(B,H,W,C)` array), not just the 2D pooling extent
`(k, k)` — passing the 2-tuple silently pools the wrong axes (width and channel
instead of height and width) without raising an error.

---

## Lava-dl — unverified, and why

`pip install lava-dl` pins `torch<2.4.0,>=2.3.1`. This project runs
`torch==2.10.0+cu128`. Installing lava-dl **silently downgraded torch to a CPU-only
2.3.1 build** (and torchvision to 0.18.1) to satisfy that pin — pip doesn't block on
version conflicts by default, it just resolves to whatever the newest-requested
package needs and prints a warning afterward. That would have killed GPU training
for every other framework in this project, not just a new one.

This was caught immediately (`torch.cuda.is_available()` checked right after
install, came back `False`) and fixed by reinstalling the pinned
`torch==2.10.0+cu128` / `torchvision==0.25.0` from the PyTorch CUDA wheel index,
then uninstalling `lava-dl`/`lava-nc` entirely. It is not currently installed
anywhere in this environment.

**This is a hard version-pin conflict, not a Linux-vs-Windows issue** — lava-dl
needs its own fully isolated environment (separate venv/conda env, or its own stage
in the `container` branch's Docker image) to ever coexist with the rest of this
project's modern-torch dependency tree.

`snn_lava.py` was written anyway, at the user's request, as a best-effort sketch of
the integration — `slayer.block.cuba.{Conv,Pool,Dense}` blocks, `(B,C,H,W,T)`
time-last tensor layout (SLAYER convolves the synaptic current decay directly along
the time axis, which is genuinely different from how every other backend in this
project treats time), full PyTorch-autograd training since SLAYER's spiking
nonlinearity is a custom `Function` with a surrogate backward (unlike BindsNET/Spyx,
`is_differentiable()` should be `True` here). **None of this has been run.** Treat
every API name in that file as something to verify against the real package once
it's installed in an isolated environment — not as confirmed-working code. `main.py`
imports it inside a `try/except ImportError` so its absence doesn't break anything
else.

---

## Known gap this surfaced (not fixed here — flagging for a follow-up)

`training.py`'s TRADES adversarial-training path (`generate_trades_adversarial`,
gated by `cfg.TRADES_ENABLED`, **`true` by default** in `SNN_module.yaml`) calls
`torch.autograd.grad(kl, adv)` unconditionally — it never checks
`model.is_differentiable()` the way `adversarial_robustness.py`'s post-training
evaluator does. For BindsNET (and Spyx, and any future JAX/TF backend), that call
would receive a tensor with no gradient path back to `adv` and fail.

**Until that's fixed in `training.py`,** set `trades_enabled: false` when running
with `framework: bindsnet` or `framework: spyx`.

---

## Two real incidents from installing these (read before adding more frameworks)

1. **Never run two `pip install` commands in parallel against the same environment.**
   Installing `lava-dl` and `jax`/`spyx` at the same time caused a race on
   `numpy`'s metadata files mid-uninstall, corrupting the `jax` install. Re-ran them
   one at a time and it was clean.
2. **Check `torch.cuda.is_available()` immediately after installing any new package
   that touches `torch` as a dependency.** pip will silently satisfy a stricter pin
   from a new package by downgrading torch out from under everything else already
   relying on it. Caught here within one command; would otherwise have been a
   confusing "training is suddenly on CPU" bug discovered much later.
