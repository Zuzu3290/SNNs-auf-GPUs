https://github.com/UCI-CARL/CARLsim6

# Comparison — CARLsim (UC Irvine) vs SNNs-auf-GPUs

---

## What CARLsim Is

CARLsim (Cognitive Architecture for Research and Learning through Simulation) is a GPU-accelerated SNN simulator from UC Irvine. It targets biologically-detailed large-scale neural simulations using Izhikevich spiking neurons with realistic synaptic dynamics (AMPA, GABA, NMDA conductances, STP, STDP, axonal plasticity). It provides a PyNN-like C/C++ API with a Python wrapper (pyCARL) via SWIG. Multi-GPU distribution is supported. It is actively maintained (latest commit February 2025, version 6.1).

---

## Direct Comparison

| Dimension | CARLsim | SNNs-auf-GPUs |
|---|---|---|
| **Language** | C/C++/CUDA + MATLAB/Python wrapper | C++/CUDA kernel + Python application layer |
| **Primary neuron model** | Izhikevich (4 or 9 parameter), LIF | LIF (T-in-register, warp-oriented) |
| **Synaptic dynamics** | Conductance-based AMPA/GABA/NMDA, STP, axonal plasticity | Weight-based (synaptic dynamics handled by framework layer above kernel) |
| **Learning rules** | E-STDP, I-STDP, DA-STDP, homeostatic plasticity, STP | BPTT, surrogate gradients, TRADES adversarial, STDP regularisation |
| **GPU dispatch** | CUDA kernels in snn_gpu_module.cu, synapse-parallel and neuron-parallel | T-in-register single-launch, SM-saturating grid-stride, adaptive blocks_per_sm |
| **Multi-GPU** | Yes — P2P partitioned distribution across devices | Single GPU (runtime layer designed for per-device expansion) |
| **Python interface** | pyCARL via SWIG — PyNN-compatible | pybind11 — PyTorch-native, returns torch.Tensor |
| **ML framework integration** | None — not designed for backprop | Full PyTorch integration — autograd, AMP, gradient accumulation |
| **Memory management** | Standard CUDA allocator | cuMemPool arbiter with hard/soft zone limits and pool release threshold |
| **Warp ballot** | Not documented | `__ballot_sync` per timestep — free spike rate, feeds adaptive energy feedback |
| **Backpropagation** | Not supported | Full BPTT with surrogate gradients |
| **Target scale** | Millions of neurons (sparse biological networks) | Training batches — B=32, N=4096 at production scale |
| **Input type** | Poisson spike generators, external stimulation | DVS event camera neuromorphic tensors [T, B, C, H, W] |
| **Build system** | CMake, CUDA 11+ | Python extension build (setup.py / JIT load), CUDA 11+ |
| **Stars** | 58 | — |
| **Maintenance** | Active (v6.1, Feb 2025) | Active (2026) |

---

## Architecture Difference

CARLsim is a **simulation engine** with a configure-compile-run workflow:
1. Define neuron groups, connection patterns, synapse types in C++
2. Call `setupNetwork()` — allocates memory, compiles topology
3. Call `runNetwork(seconds, milliseconds)` — advances simulation

The unit of execution is biological time (milliseconds of simulated time). CARLsim tracks spike queues, synaptic delays (up to 20ms), neuron membrane states, and conductance values across the whole network simultaneously. Its GPU kernel is one large monolithic simulation step.

Our project has a **training loop** workflow:
1. One forward pass per mini-batch — kernel processes `[B, N, T]` input in a single CUDA launch
2. Loss computed from spike output tensor
3. Backpropagation through surrogate gradients
4. Optimizer step

The unit of execution is one training iteration. The kernel does not maintain a global spike queue or synaptic delay buffer — each forward pass is stateless at the population level (voltage state is the only persistent variable).

---

## Performance

CARLsim benchmarks measure **simulation throughput** — biological seconds simulated per wall-clock second — tested on networks like 80% excitatory / 20% inhibitory populations. These are neuroscience benchmarks, not ML training benchmarks.

Our benchmarks measure **forward-pass latency** for a training workload:
- Warp-oriented kernel: 0.028 ms wall / 0.035 ms GPU-only (B=4, N=512, T=25)
- 100x vs PyTorch baseline at test scale
- Estimated 4–8x at production scale (B=32, N=4096)

These are not comparable metrics. CARLsim is optimised for minimising simulation clock time on networks with millions of synapses and realistic spike propagation delays. Our kernel is optimised for mini-batch forward-pass latency in a training loop.

---

## Where CARLsim Is Better

- **Biological detail**: Izhikevich neurons (4-parameter: regular spiking, fast spiking, bursting, etc.) and conductance-based synapses are not available in our kernel. CARLsim models biophysics that our LIF kernel explicitly approximates away.
- **Synaptic plasticity completeness**: E-STDP, I-STDP, dopamine-modulated STDP, STP (Tsodyks-Markram), and axonal plasticity form a full computational neuroscience toolkit. Our STDP is a loss regulariser, not a local learning rule.
- **Scale**: CARLsim is designed for millions of neurons with sparse connectivity. Our kernel targets dense mini-batch tensors up to ~100K neurons.
- **Multi-GPU**: CARLsim supports explicit P2P network partitioning across GPUs. Our system is single-GPU.
- **Delay modelling**: CARLsim supports configurable synaptic delays up to 20ms with explicit spike queues. Our kernel has no synaptic delay — input arrives at each timestep through the tensor.

## Where SNNs-auf-GPUs Is Better

- **Gradient-based training**: CARLsim has no backpropagation. Our system trains with BPTT, surrogate gradients, and adversarial objectives. This is the fundamental difference — CARLsim simulates, we train.
- **PyTorch integration**: CARLsim's Python interface (pyCARL via SWIG) is not PyTorch-native. Our pybind11 binding returns `torch.Tensor` and lives inside the standard autograd graph.
- **Kernel depth per forward pass**: T-in-register voltage eliminates 25 global memory round-trips per neuron per forward pass. Warp ballot gives free sparsity measurement. cuMemPool arbiter coordinates VRAM across the pipeline. CARLsim's CUDA implementation is not documented at this level.
- **ADAS domain fit**: DVS event camera data handling, neuromorphic data pipeline, temporal slice encoding — none of this exists in CARLsim.
- **Adversarial robustness**: TRADES training and adversarial evaluation are core features. CARLsim has no concept of adversarial inputs.

---

## Summary

CARLsim and our project occupy different positions in the SNN landscape. CARLsim answers: "What does this biological network do when simulated accurately?" Our project answers: "How do we train an SNN to classify DVS event streams efficiently on a GPU?" The overlap is the LIF neuron and the GPU execution substrate. Everything above and below that shared layer is different.
