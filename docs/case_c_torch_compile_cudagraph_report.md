# Case C — `torch.compile` and CUDA Graph Capture Pilot on the Shared SNN Model

**Status: CLOSED for `torch.compile`, OPEN (deferred) for raw CUDA graphs.**

**Summary verdict:** `torch.compile(mode="default")` gives a real, verified 1.3–1.9x
per-batch speedup for two of the four framework backends (Norse, SpikingJelly). It is
unreliable for snntorch and consistently counterproductive for Sinabs, so it is **not**
adopted as a universal, always-on setting — doing so would also break this project's
existing rule that every framework trains under an identical recipe for the comparison
to mean anything. Raw CUDA graph capture, attempted by hand with `torch.cuda.graph()`,
failed outright for two backends and — more importantly — silently produced **wrong
gradients** for the other two. That second finding is the one to remember: a technique
that looks like a 20-80x speedup in a naive test can be completely worthless if it isn't
computing the right thing, and only a built-in correctness check caught it here.

---

## 1 — What was being diagnosed

The original question came from a real blocker: on native Windows, `torch.compile`
failed on the real `SNN_NORSE` model with `InvalidCxxCompiler: Compiler: cl is not
found` — Inductor's backend needs a C++ compiler for CPU-side codegen even when the
model itself runs on CUDA, and this machine has no MSVC toolchain installed. Installing
Visual Studio Build Tools is a multi-GB, system-level change, so rather than do that
unprompted, the pilot moved to WSL2 (Ubuntu 22.04), which already has GPU passthrough,
`gcc`, and a working CUDA-enabled PyTorch install (`torch 2.13.0+cu130`) with no
Windows-toolchain dependency at all.

Two rounds of testing were run, both using the real `SNNModel` class and the real
`sum_over_time_cross_entropy` loss the training loop actually uses — not a toy network —
so every finding below transfers directly to `learning/training.py`/`learning/inference.py`
without needing to modify either file first:

- **Round 1** (`diagnostics/case_c_norse_cuda_graph_probe.py`): Norse only, two dataset
  shapes (N-Caltech101, DVS128 Gesture), `torch.compile` in both `default` and
  `reduce-overhead` mode, 10 steady-state calls per configuration.
- **Round 2** (`diagnostics/case_c_compile_cudagraph_small_eval.py`): a small screening
  pass — all **four** framework backends (torch/snntorch, Norse, SpikingJelly, Sinabs),
  **three** dataset shapes (N-MNIST, N-Caltech101, DVS128 Gesture), 5 steady-state calls
  per configuration, comparing eager vs. `torch.compile(default)` vs. a hand-rolled raw
  `torch.cuda.CUDAGraph()` capture. All data is synthetic, at each dataset's real
  registry shape (`sensor_size`/`num_classes`) — no download needed, the same pattern
  `learning/utilities.measure_batch_vram()` already uses for its own probing.

Both scripts live in `diagnostics/`, which is gitignored (personal/local-only), so they
are not part of the tracked repo — this document is the durable record of what they
found.

---

## 2 — `torch.compile(mode="default")`: real, but not universal

| Framework | N-MNIST | N-Caltech101 | DVS128 Gesture | Verdict |
|---|---|---|---|---|
| Norse | 1.78x faster | 1.53x faster | 1.87x faster | **Consistent win** |
| SpikingJelly | 1.27x faster | 1.32x faster | 1.53x faster | **Consistent win** |
| snntorch (torch) | 0.21x — **5x slower** | 1.19x faster | 1.24x faster | **Unreliable** |
| Sinabs | 0.46x slower | 0.50x slower | 0.43x slower | **Consistent loss** |

Loss and gradient norm matched the eager baseline to within 1e-6–1e-11 relative
difference for Norse and SpikingJelly in every case — the speedup is real, not a
shortcut that skips work.

**snntorch's failure on N-MNIST is explained, not mysterious**: the run log showed
`torch._dynamo hit config.recompile_limit (8)` for snntorch's `Leaky` neuron
(`snntorch/_neurons/leaky.py:228`), meaning Dynamo kept re-tracing and re-compiling
instead of reusing one compiled graph — the 749ms mean is mostly repeated compilation
cost, not the compiled kernel's real steady-state speed. The larger datasets didn't hit
the limit as visibly, but their gradient match (4e-4 relative, versus Norse's typical
1e-9–1e-11) is still five orders of magnitude looser — a sign of a partially-degraded
compile path, not a clean one. **This should be treated as "not yet safe to compile,"
not "confirmed slower."**

**Sinabs's loss is consistent and not a fluke** — the same ~2x slowdown appeared on all
three dataset shapes. Sinabs's LIF layer resets its buffers to zero-size and reinfers
their shape on every call (`frameworks/adapters/sinabs_lif.py`'s `reset()` — see the
in-code comment on why it deliberately avoids sinabs' own `reset_states()`), which is a
structurally different, allocation-heavy pattern the other three don't have. Compiling
around that doesn't help and adds its own overhead.

**Why not just turn it on for the two that benefit?** Because this project has an
explicit, load-bearing rule that every framework shares one optimizer, one loss, and one
training recipe specifically so a cross-framework comparison stays apples-to-apples
(`learning/utilities.py`'s `build_optimizer`/`build_loss` docstrings). Compiling only
Norse and SpikingJelly during a comparison run would reintroduce exactly the kind of
per-framework asymmetry that rule exists to prevent. `torch.compile(default)` is
therefore documented here as an **available, optional, WSL-only performance mode** —
worth reaching for on a dedicated throughput run of one framework, not baked into the
standard multi-framework comparison pipeline.

---

## 3 — Raw CUDA graph capture: two failures, two false positives

`torch.compile(mode="reduce-overhead")` (Round 1) already showed CUDA-graph-based
execution struggling for Norse — slower than eager, not faster, most likely because the
LIF layer's state starts as `None` and becomes a real tensor after the first timestep, a
pattern that fights the "identical op sequence every call" assumption CUDA graphs need.
Round 2 tested this directly and by hand — build the model, run a few real eager passes
first so state is already a real tensor, *then* capture with `torch.cuda.graph()` — to
see whether a more careful, manual capture could do better than Inductor's automatic
one. The result was more informative than a simple pass/fail:

| Framework | Result | Root cause |
|---|---|---|
| Norse | **Failed outright** — `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned.` | `frameworks/adapters/norse_lif.py`'s `build_lif()` constructs `LIFBoxParameters` with `torch.as_tensor(...)` and no `device=` argument — these tensors are built on CPU. They aren't registered as `nn.Module` buffers, so `SNNModel`'s own `self.to(self.device)` call never moves them; Norse's `LIFBoxCell` almost certainly copies them to the input's device on every forward call, and that unpinned copy is exactly what graph capture forbids. **Likely fixable** — constructing these tensors already on `cfg.DEVICE` is a small, scoped adapter change, not a fundamental Norse limitation. Not attempted here; noted as a follow-up. |
| snntorch (torch) | **Failed outright** — `AcceleratorError: CUDA error: operation failed due to a previous error during capture` / `cudaErrorStreamCaptureInvalidated` | This is a cascading error: once any single operation inside a capture region genuinely fails, every operation after it reports "invalidated" regardless of what it was. **The real failing operation is not identifiable from this message alone.** A plausible but unconfirmed suspect is snntorch's custom surrogate-gradient `autograd.Function` (`SURROGATES[stype]` in `snntorch_lif.py`) doing something capture-incompatible in its backward pass, but pinning this down would need a dedicated, isolated repro (bisecting the forward/backward region operation by operation) — out of scope for this screening. **Unresolved.** |
| SpikingJelly | Completed without error, but **`rel_loss=1.0`, `rel_gnorm=1.0`** — the post-capture gradient is completely unrelated to the correct one, not close to it. | Not a SpikingJelly-specific problem. The run log shows the actual cause directly: a PyTorch UserWarning about an `AccumulateGrad` node created on the wrong CUDA stream being kept alive across iterations — a well-documented pitfall of hand-capturing `backward()` with plain `torch.cuda.graph()`, which is exactly why PyTorch ships `torch.cuda.make_graphed_callables()` to handle gradient-accumulator lifetime and stream consistency correctly. **This is an implementation-methodology gap, not a SpikingJelly limitation.** |
| Sinabs | Same failure mode as SpikingJelly — completed, but `rel_loss=1.0`, `rel_gnorm=1.0`. | Same root cause as SpikingJelly: the manual capture protocol, not the framework. |

**The headline numbers for SpikingJelly and Sinabs in the raw output (up to 27x and 82x
"speedup") must not be read as real performance results.** They are the byproduct of a
capture that silently computed the wrong thing — caught only because this screening
checked loss and gradient against an eager baseline instead of trusting a fast-looking
number. Had that check not been there, this would have been an easy, plausible-looking
false win to report.

---

## 4 — Conclusion

- **`torch.compile(mode="default")` is a genuine, worthwhile optimization for Norse and
  SpikingJelly specifically** — not a blanket win, and not currently safe to enable for
  snntorch or Sinabs. Keep it as an optional, documented, per-framework, WSL-only mode
  for dedicated throughput work; do not enable it for standard cross-framework
  comparison runs, where the shared-recipe rule takes priority over raw speed.
- **Raw CUDA graph capture is not currently achievable here with a straightforward,
  hand-rolled approach.** Two backends fail outright (one with a likely-fixable,
  identified cause in Norse's adapter; one with an unidentified cause in snntorch). The
  other two "succeed" in a way that is actively misleading — correct, verified
  correctness checking is what separates a real result from this failure mode, and any
  future CUDA-graph work here must keep that check.
- **No universal setup change is warranted at this stage.** The pipeline does not
  currently accommodate either technique as a standing feature; both remain available,
  documented techniques for future, deliberate adoption rather than defaults.

---

## 5 — Open items (deferred, not scheduled)

1. **Redo CUDA graph capture using `torch.cuda.make_graphed_callables()`** instead of a
   hand-rolled `torch.cuda.graph()` block, for SpikingJelly first (it already has a
   clean, verified `torch.compile` win, so a correctly-implemented graph capture is the
   most promising next step) and optionally Sinabs. This is a real, separate
   implementation task — not a rerun of the existing scripts — since the whole point is
   that the naive protocol used here is the thing that needs to change.
2. **Fix Norse's `LIFBoxParameters` device placement** in
   `frameworks/adapters/norse_lif.py` (construct on `cfg.DEVICE` directly instead of
   CPU) as a small, scoped change, then re-test whether that alone unblocks CUDA graph
   capture for Norse.
3. **snntorch's raw-CUDA-graph failure cause remains unidentified.** Would need a
   dedicated, isolated repro to bisect, separate from this screening.
4. Neither item above is scheduled — this section exists so the next session doesn't
   have to rediscover what was already tried and why it stopped here.
