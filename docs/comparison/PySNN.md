https://github.com/BasBuller/PySNN

# Comparison — PySNN (BasBuller) vs SNNs-auf-GPUs

---

## What PySNN Is

PySNN is a Python/PyTorch SNN framework built around correlation-based learning (STDP) rather than gradient-based backpropagation. Every component (Neuron, Connection, SNNNetwork) inherits from `nn.Module`, making it feel familiar to PyTorch users. Its design goal is to enable research into spike-timing-dependent plasticity and event-based processing, specifically for temporal data such as event-based video. It explicitly does not support backpropagation.

---

## Direct Comparison

| Dimension | PySNN | SNNs-auf-GPUs |
|---|---|---|
| **Language** | Python + PyTorch | Python application layer + C++/CUDA kernel |
| **LIF implementation** | Python-level nn.Module, GPU via PyTorch ops | Hand-coded CUDA kernel — T-in-register register-resident voltage, single launch for all T steps |
| **GPU acceleration** | PyTorch tensor ops (no custom CUDA) | Custom CUDA kernels — warp ballot, SM-saturation, cuMemPool |
| **Training paradigm** | STDP only — no backpropagation | Backpropagation (BPTT), TRADES adversarial, STDP regularisation, AMP |
| **PyTorch integration** | Deep — nn.Module throughout | Deep — pybind11 binding returns torch.Tensor, sits inside standard training loop |
| **Temporal dimension** | Extra trace tensor dimension added to all ops | T processed inside kernel register loop — no extra dimensions exposed |
| **Gradient support** | Explicitly not supported | Full autograd compatibility — spikes are differentiable via surrogate gradient |
| **Kernel dispatch** | None — uses standard PyTorch dispatch | Adaptive SM scheduling via energy feedback every 50 calls |
| **Memory coordination** | None — relies on PyTorch allocator | cuMemPool MemoryArbiter with zone-level hard limits |
| **Input type** | Event-based video and temporal data | DVS event camera data ([T, B, C, H, W]) |
| **Stars** | 232 | — |
| **Maintenance** | Active | Active |

---

## Architecture Difference

PySNN adds a **trace dimension** to every tensor in the network, turning a standard `[batch, channels, height, width]` tensor into `[batch, channels, height, width, traces]`. This is a PyTorch-only abstraction for tracking spike timing within the existing tensor framework. No custom GPU code is written — everything runs through PyTorch's standard dispatcher.

Our kernel **internalises the temporal loop**. The trace / voltage state is held in a CUDA register across all T steps within a single kernel launch. Python never sees the per-timestep state — it only receives the final `[B, N, T]` spike tensor. This is fundamentally different from PySNN's approach: PySNN surfaces the temporal dimension to Python; we hide it inside the GPU.

---

## Performance

PySNN provides no benchmarks. Its performance ceiling is bounded by PyTorch's dispatcher — every timestep in the STDP update loop pays a Python-to-CUDA boundary crossing cost per tensor operation, identical to the PyTorch baseline we measured.

Our benchmark for the equivalent operation:
- PyTorch baseline (equivalent to PySNN's approach): 2.785 ms / forward pass
- Custom warp-oriented kernel: 0.028 ms / forward pass
- Speedup: 100x at test scale (B=4, N=512, T=25)

At production scale (B=32, N=4096), the gap narrows to 4–8x as both systems become compute-bound rather than dispatch-bound.

---

## Where PySNN Is Better

- **STDP implementation**: PySNN has a complete, usable STDP learning rule system that works out of the box. Our STDP is a loss regularisation term added on top of BPTT gradients — it is not a standalone local learning rule.
- **Modularity**: PySNN's Neuron/Connection separation makes it easy to mix custom neuron types with different synaptic dynamics per layer. Our kernel handles one LIF variant at a time.
- **Ease of use for SNN-only research**: A PySNN user writes pure Python without needing to compile CUDA extensions. Our system requires a C++/CUDA build step.

## Where SNNs-auf-GPUs Is Better

- **Speed**: No custom CUDA code in PySNN means it pays full PyTorch dispatch overhead on every timestep. Our kernel eliminates 124 of 125 dispatch calls per forward pass.
- **Gradient-based training**: PySNN cannot train with backpropagation at all. Our system supports full BPTT with surrogate gradients, compatible with any loss function.
- **Hardware depth**: Warp ballot, cuMemPool zone arbitration, adaptive SM scheduling, SpikeRateBus — none of these exist in PySNN.
- **Adversarial robustness**: TRADES adversarial training and STDP regularisation coexist in our training loop. PySNN has no adversarial training support.
- **Target domain**: DVS ADAS pipeline — PySNN is general-purpose and has no event camera integration.
