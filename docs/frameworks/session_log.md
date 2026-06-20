# Session Log — Adding Sinabs / BindsNET / Spyx / Lava-dl

A narrative account of what was actually done, what broke, and why — written
because two things came up mid-session that looked alarming out of context
(a TensorFlow/TensorBoard detour, and a library loudly complaining about itself
on import). Reference docs for the frameworks themselves live in
`additional_frameworks.md`; this file is the "what happened and why" version.

---

## 1. Installing Sinabs + BindsNET

```
pip install sinabs bindsnet
```

Both installed cleanly. Output included this line, easy to miss:

```
tonic 1.6.0 requires numpy<2.0.0, but you have numpy 2.2.6 which is incompatible.
tensorflow 2.10.1 requires protobuf<3.20,>=3.9.2, but you have protobuf 7.35.1 which is incompatible.
tensorboard 2.10.1 requires protobuf<3.20,>=3.9.2, but you have protobuf 7.35.1 which is incompatible.
```

`sinabs` pulled in `tensorboardX` (a *different* package from `tensorboard` — it's
a third-party library that writes TensorBoard-format logs without needing
TensorFlow installed). That package wanted a much newer `protobuf` than the
`tensorboard` 2.10.1 already sitting in this environment supported. At the time
this looked harmless — `tonic` and `torch` both still imported fine, so I moved on.

**It wasn't harmless.** It came back two steps later.

---

## 2. Why TensorBoard got touched (this project uses PyTorch, not TensorFlow)

When verifying the new framework files against the real config pipeline, importing
the `learning` package crashed:

```python
from skeleton import Settings
```
```
File ".../norse/torch/__init__.py", line 331, in <module>
    from norse.torch.utils.tensorboard.tensorboard import (...)
File ".../torch/utils/tensorboard/writer.py", line 13, in <module>
    from tensorboard.compat.proto import event_pb2
...
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date...
```

Here's the chain, since it's not obvious:

- **`torch.utils.tensorboard`** is a real part of PyTorch itself — it's PyTorch's
  built-in wrapper for writing TensorBoard log files, with zero TensorFlow
  involvement. It just happens to depend on the separately-pip-installed
  `tensorboard` *package* to do the actual file writing.
- **`norse`** (one of the original three frameworks, already in this project)
  imports `torch.utils.tensorboard` unconditionally at package-import time, as
  part of its own logging utilities — not something this project asked for, just
  baked into `import norse.torch`.
- The `tensorboard` 2.10.1 package already installed here has its protocol-buffer
  message classes **generated against an old protobuf compiler**. Sinabs's
  `tensorboardX` dependency upgraded the `protobuf` *runtime* to 7.35.1. Old
  generated code + new protobuf runtime = the `TypeError` above. This is a known,
  common protobuf breakage pattern, unrelated to anything SNN-specific.

So: **Norse broke because of a transitive dependency of Sinabs, mediated by a
PyTorch built-in module that happens to share a name with TensorFlow's most famous
tool.** No TensorFlow code was added to this project, and nothing here now depends
on TensorFlow.

### The fix

```
pip install -U tensorboard      # 2.10.1 → 2.20.0, built against modern protobuf
pip uninstall -y tensorflow tensorflow-estimator tensorflow-io-gcs-filesystem
```

The tensorboard upgrade alone wasn't enough — upgrading it made `tensorboard`'s
internal lazy-loader try `import tensorflow` (to check for a real TF installation),
and the `tensorflow` 2.10.1 package *already sitting in this environment* — itself,
independently of tensorboard — had the exact same old-protobuf-generated-code
problem. `grep -r "import tensorflow"` across this repo turned up nothing except
one mention in `docs/frameworks/README.md`'s JAX/TF documentation — so that
`tensorflow` install was dead weight from some unrelated past experiment, not a
project dependency. Removing it let tensorboard's lazy-loader fall back to its
"no TF" stub, and everything imported cleanly.

**Net effect on this project:** `tensorboard` went from 2.10.1 → 2.20.0 (still just
a logging backend PyTorch can optionally use — nothing imports it directly here),
and an unused `tensorflow` install was removed. No TensorFlow dependency was added.

---

## 3. The Gym warning

Every time `bindsnet` is imported, this prints:

```
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other
critical functionality. Please upgrade to Gymnasium, the maintained drop-in
replacement of Gym, or contact the authors of your software and request that they
upgrade. Users of this version of Gym should be able to simply replace
'import gym' with 'import gymnasium as gym' in the vast majority of cases.
```

`gym` is OpenAI's original reinforcement-learning environment library (think
"CartPole," Atari benchmarks — nothing to do with spiking neurons). `bindsnet`
pulls it in as a dependency for some of its own RL-flavored example utilities,
which this project doesn't use at all. The warning is `gym` nagging about its own
unmaintained status on every import; it's cosmetic noise from a sub-dependency of a
sub-dependency, not a sign anything is wrong with BindsNET or this integration.
It's been left as-is rather than suppressed, since silencing warnings from code we
don't control can hide a real one later.

---

## 4. The actually-dangerous incident: Lava-dl

```
pip install lava-dl
```

```
Attempting uninstall: torch
    Found existing installation: torch 2.10.0+cu128
    Uninstalling torch-2.10.0+cu128:
      Successfully uninstalled torch-2.10.0+cu128
...
Successfully installed ... torch-2.3.1 torchvision-0.18.1 ...
```

`lava-dl` pins `torch<2.4.0`. pip doesn't refuse to install a package whose pin
conflicts with what's already there — it just downgrades the existing package to
satisfy the new one, silently, and only warns about *other* conflicts afterward.

```
python -c "import torch; print(torch.cuda.is_available())"
False
```

That confirmed it: the `2.3.1` build pip pulled had no CUDA support at all. Every
other framework in this project — Norse, snnTorch, SpikingJelly, the two newly
verified backends — would have silently lost GPU access.

### The fix

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
pip uninstall -y lava-dl lava-nc
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
torch 2.10.0+cu128 True
```

Restored, verified, and `lava-dl` removed entirely — it cannot be installed
alongside the rest of this project's dependencies without repeating this exact
problem. `snn_lava.py` exists as an unverified, never-run sketch for whenever it
gets its own isolated environment (separate venv or its own Docker stage).

### A second, smaller version of the same mistake

`jax`/`jaxlib`/`spyx` were installed in a background shell *while* the `lava-dl`
install above was also running in the background, in parallel. `lava-dl`'s install
was mid-uninstall of `numpy` at the exact moment the `jax` install tried to read
`numpy`'s package metadata:

```
ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory:
'...\numpy-2.2.6.dist-info\METADATA'
```

Re-running the `jax`/`spyx` install alone (no other pip process running
concurrently) fixed it cleanly with zero conflicts reported. Lesson applied for the
rest of the session: one `pip install` at a time, full stop.

---

## 5. What was actually verified, and how

For each of Sinabs, BindsNET, and Spyx — not just "it imports," but, against the
real `Settings()` config (production `BATCH_SIZE=128`, `TIMESTEPS=25`, real
`FC_IN=800` from the conv/pool dims in `network_architecture.yaml`):

1. Instantiated the model class.
2. Ran a real forward pass, checked the output shape was `[T, B, NUM_CLASSES]`.
3. Computed the loss.
4. Ran one training step and confirmed weights actually changed
   (`torch.allclose(w_before, w_after)` is `False`) — for BindsNET this meant
   confirming STDP fired, not backprop.
5. Switched to eval mode and ran forward again.

This is also how the two real bugs documented in `additional_frameworks.md` were
found — BindsNET's `MaxPool2dConnection` batch-size bug, and the `hk.max_pool`
rank-mismatch in the Spyx pooling call — neither would have surfaced from import
checks alone.

Lava-dl got none of this — see `additional_frameworks.md` for why, and what's
unverified about it.
