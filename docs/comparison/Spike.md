https://github.com/OFTNAI/Spike

# Comparison — Spike (OFTNAI) vs SNNs-auf-GPUs

---

## What Spike Is

Spike is a standalone GPU-accelerated SNN simulator written in C++/CUDA, built with CMake. Its design goal is computational neuroscience simulation — specifically large-scale biological network benchmarks such as the Vogels-Abbott (4,000 LIF neurons) and Brunel (10,000 LIF neurons, 10^7 synapses) networks. It claims to be "one of the fastest SNN simulators available" within its benchmark suite. It has no Python interface, no deep learning framework integration, and no gradient-based training.

---

## Direct Comparison

| Dimension | Spike | SNNs-auf-GPUs |
|---|---|---|
| **Language** | C++/CUDA, CMake | C++/CUDA kernels + Python application layer |
| **Python interface** | None | pybind11 extension (snn_forward, snn_runtime) |
| **Training paradigm** | STDP only — no backpropagation | Backpropagation (BPTT), surrogate gradients, STDP regularisation, TRADES adversarial |
| **LIF implementation** | Standard LIF, timestep simulation | T-in-register (register-resident voltage across T), warp-oriented SM-saturating kernel |
| **GPU dispatch** | Fixed kernel per network call | Adaptive SM scheduling — target_blocks_per_sm adjusted every 50 calls via energy feedback |
| **Warp ballot** | Not used | `__ballot_sync` per timestep — free spike rate measurement, feeds SpikeRateBus |
| **Memory management** | Not documented beyond standard CUDA | cuMemPool arbiter with zone-level hard/soft limits and pool release threshold pinning |
| **Multi-GPU** | Not mentioned | Single GPU (current); runtime layer designed for per-device MemoryArbiter instances |
| **Input type** | Poisson spike generators | DVS event camera data ([T, B, C, H, W] neuromorphic tensors) |
| **Network scale benchmarks** | 4K–10K neurons | B=4, N=512, T=25 (test); architecture targets B=32, N=4096 production scale |
| **Last commit** | March 2020 — abandoned | Active (2026) |
| **Stars** | 48 | — |

---

## Architecture Difference

Spike is a **simulation engine** — you configure a network topology, attach synapse groups and learning rules, then call simulate(). It manages its own internal spike queues, synapse weight matrices, and time-stepping loop.

Our project is a **training substrate** — the kernel is a compute primitive that sits under a PyTorch training loop, feeding into cross-entropy loss, BPTT gradients, and adversarial robustness evaluation. The simulation step is one forward pass in a larger learning system.

---

## Performance

Spike's benchmark claims are against other neuroscience simulators (NEST, Brian2, GeNN). These are not gradient-based training benchmarks — they measure simulation throughput in seconds-of-biological-time-per-second-wall-time, not inference latency or training throughput.

Our kernel benchmarks measure forward-pass wall time for training:
- Warp-oriented kernel: 0.028 ms / forward pass (B=4, N=512, T=25)
- Speedup: 100x vs PyTorch baseline at test scale (dispatch-elimination dominated)
- Estimated 4–8x at production scale (B=32, N=4096)

These are different metrics and not directly comparable. Spike is optimised for long biological simulations with dense synaptic connectivity. Our kernel is optimised for low-latency forward passes inside a mini-batch training loop.

---

## Where Spike Is Better

- **Synaptic realism**: Conductance-based synapses (AMPA, GABA), configurable transmission delays, weight-dependent STDP. Our kernel only implements LIF — synaptic dynamics are handled by the framework layer above it.
- **Network topology tools**: Spike has explicit connection builder infrastructure. Our project does not — topology is defined by the PyTorch model.
- **Simulation fidelity**: Spike is designed to match biological network dynamics precisely. Our kernel is optimised for ML throughput, not biological accuracy.

## Where SNNs-auf-GPUs Is Better

- **Training**: Spike cannot train with backpropagation. Our system trains end-to-end with BPTT, surrogate gradients, TRADES, STDP regularisation, and AMP.
- **Python integration**: Full PyTorch ecosystem — dataloaders, optimisers, gradient scalers, checkpoint management.
- **Kernel depth**: T-in-register voltage (no global memory between timesteps), cuMemPool arbiter, warp ballot, adaptive SM scheduling. Spike's CUDA implementation is not publicly detailed.
- **Maintenance**: Spike has had no commits since 2020. This project is actively developed.
- **Target domain**: DVS event camera ADAS pipeline. Spike has no event camera input support.
