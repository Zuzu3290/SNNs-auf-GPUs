<<<<<<< HEAD
 this is the precise architectural reason why C++/CUDA was the right choice, not just a performance preference.
=======
# C++/CUDA vs JAX/XLA — Architectural Choice

This document explains why C++/CUDA was chosen over JAX/XLA for this codebase.
It does so without dismissing what JAX genuinely does better — both are legitimate GPU
programming models and the comparison below is intentionally unbiased.

---

## What JAX/XLA genuinely gives you

These are real advantages. Dismissing them would be dishonest.

**Automatic kernel fusion.**
Write `y = relu(dot(x, w) + b)` and XLA fuses it into a single GPU kernel automatically.
For standard DNN layers — convolutions, attention, LayerNorm — XLA-generated code is
routinely competitive with hand-written CUDA and sometimes faster, because XLA sees the
full compute graph and optimises memory access patterns globally. A custom CUDA Conv2d
kernel is not going to beat cuBLAS + XLA fusion.

**Functional transforms that compose.**
`jax.grad(jax.jit(jax.vmap(fn)))` works on arbitrary functions without rewriting anything.
Write the function for one sample; `vmap` vectorises over the batch; `jit` compiles it;
`grad` differentiates through it. In raw CUDA each of these is a separate concern managed
manually.

**No mutable state.**
JAX is functionally pure — arrays are immutable. This eliminates an entire class of GPU
synchronisation bugs caused by in-place mutation and shared state across kernel launches.

**Gradient checkpointing for free.**
`jax.checkpoint` rematerialises activations during the backward pass without storing them.
In PyTorch you wrap modules with `torch.utils.checkpoint` manually. In raw CUDA you
implement it yourself.

**TPU portability.**
The same JAX code runs on an NVIDIA GPU and on a TPU without modification.
CUDA code is permanently NVIDIA-only.

**The learning layer in this codebase is already at this level.**
`training.py`, `snn_norse.py`, `snn_torch.py` are PyTorch autograd — conceptually
equivalent to JAX at the operation level. BPTT, the surrogate gradient in `cuda_ops.py`,
the loss functions: none of this required raw CUDA. PyTorch's autograd handles it
correctly. This is worth being explicit about: `src/learning/` does not benefit from raw
CUDA. It benefits from PyTorch, which operates at the same abstraction level as JAX.

---

## Where JAX undermines the case for raw CUDA

These are cases where JAX would have been the equal or better decision:

**Standard layers.** If this project used only Conv2d, Linear, and standard activations,
XLA fusion would match or outperform custom CUDA. There is no value in writing a custom
matrix multiply kernel — cuBLAS already exists and XLA calls it.

**Multi-device scaling.** JAX's `pmap` distributes computation across multiple GPUs with
a single-program model. It is genuinely cleaner than PyTorch DDP. For multi-GPU SNN
training, JAX would be competitive and simpler.

**SNN research prototyping.** For exploring new neuron models and learning rules, JAX's
`grad` through arbitrary Python control flow is faster to iterate on than writing CUDA
backward kernels manually.

**Debugging.** Functional purity makes gradient checking trivial. In-place ops, a real
source of training bugs in PyTorch, do not exist in JAX by design.

---

## Where C++/CUDA is the only option — the original analysis

This is the precise architectural reason why C++/CUDA was the right choice, not just a performance preference.
>>>>>>> Operators

The specific capabilities that make our system what it is sit below where XLA operates. XLA is a compiler that transforms HLO operations into GPU code. It cannot express nor expose what lives at the CUDA driver and runtime level. Here is exactly what is exclusive to our implementation:

__ballot_sync — warp-level hardware instruction. XLA has no warp primitive. Triton has a partial equivalent (tl.vote) but does not expose the raw 32-bit mask. The SpikeRateBus receiving a free popcount from the kernel ballot is not achievable through any compiler abstraction. It requires CUDA PTX.

cudaDeviceGetDefaultMemPool + cudaMemPoolSetAttribute — these are CUDA driver API calls that operate on the allocator itself. XLA manages its own memory pool internally and exposes no hook to set a release threshold or read cudaMemPoolAttrUsedMemCurrent. The entire MemoryArbiter only exists because we are at the driver level. From XLA you cannot touch the pool.

Runtime adaptive grid — dispatch_warp_oriented samples elapsed_ms every 50 calls and adjusts target_blocks_per_sm for the next launch at runtime. Triton's autotuner selects block sizes at compile time, not per-call. XLA has no mechanism for a kernel to adjust its own grid based on a measured runtime signal.

CUDA events for per-call timing — cudaEventCreate, cudaEventRecord, cudaEventElapsedTime are CUDA runtime calls used in the energy profiler and the timed kernel variant. XLA does not expose these inside a compiled computation.

std::atomic<size_t> zone accounting with CAS rollback — the lock-free zone tracking in MemoryArbiter is C++ host code. Technically a compiler does not prevent this, but without the cuMemPool API it is only accounting — there is nothing to enforce. The enforcement mechanism is cudaMemPoolSetAttribute, which is driver-level only.

If we had built on XLA/JAX custom calls — where you write a C++ function and register it as an XLA operation — we would still be writing CUDA. We would have added JAX's overhead on top of our CUDA without gaining the compiler benefits, because JAX's fusion and optimisation only apply inside its traceable operations, not to opaque custom calls.

The conclusion: C++/CUDA was not a performance choice over XLA. It was an access choice. These hardware capabilities do not exist at the abstraction level XLA operates at. They only exist at the driver level. Writing in C++/CUDA was the only way to build what we built.

---

## Side-by-side summary

| Capability | JAX / XLA | C++ / CUDA |
|---|---|---|
| Standard layer fusion (Conv, Linear, Attention) | **Better** — XLA does it automatically | Manual — must write the fused kernel |
| Functional transforms (`vmap`, `pmap`, `grad`) | **Native** | Not applicable |
| Multi-GPU scaling | **Cleaner** (`pmap`) | Requires DDP / manual management |
| TPU portability | **Yes** | No — NVIDIA only |
| SNN framework ecosystem (Norse, SNNTorch, SpikingJelly) | No support | **Native** — all PyTorch-based |
| Warp vote / ballot instructions (`__ballot_sync`) | Not exposed | **Full access** |
| CUDA memory pool control (`cuMemPoolSetAttribute`) | Not exposed | **Full access** |
| Runtime-adaptive kernel grid (elapsed_ms feedback) | Not possible — AOT only | **Possible** |
| NVML power / temperature / ECC telemetry | Not accessible | **Full access** |
| Per-kernel CUDA event timing | Not exposed inside compiled block | **Full access** |
| PTX hot-swap at runtime (Driver API) | Not possible | **Possible** |
| Gradient checkpointing | **Automatic** (`jax.checkpoint`) | Manual (`torch.utils.checkpoint`) |
| In-place mutation bugs | **Eliminated** by functional purity | Possible — developer responsibility |

---

## Final position

The learning layer — training loop, BPTT, framework forward passes — does not need raw
CUDA. PyTorch autograd is sufficient, and JAX would have been a legitimate alternative
at that level.

The hardware layer — warp voting, memory pool control, NVML telemetry, runtime-adaptive
dispatch, PTX hot-swap — cannot be expressed at any abstraction level above the CUDA
driver. JAX cannot access it. Triton cannot access it. XLA cannot access it.

The choice of C++/CUDA was not made because CUDA is universally faster.
It was made because the specific capabilities this system requires only exist at the level
C++/CUDA operates at. JAX would have covered the learning layer equally well.
It would have covered none of the hardware layer.

message from chat :What JAX gives you" — honest acknowledgment that XLA fusion, vmap/pmap/grad, functional purity, and TPU portability are real advantages. Critically: explicitly states that src/learning/ (PyTorch autograd) is already at the same level as JAX and didn't need raw CUDA.

"Where JAX undermines the case for raw CUDA" — the unbiased part. Standard layers, multi-device, prototyping, debugging — JAX would have been the better call. This section didn't exist before.

The original five technical points are untouched — __ballot_sync, cuMemPool, runtime-adaptive grid, CUDA events, CAS atomic accounting. These are preserved word-for-word under the heading "Where C++/CUDA is the only option."

Added after the original content:

Summary table — side-by-side, 13 rows, no winner declared globally. Each row is honest about which tool wins that specific capability.

Final position — replaces the old one-sided conclusion. The new version explicitly says JAX would have covered the learning layer equally well, and would have covered none of the hardware layer. That's the actual truth of the situation.
