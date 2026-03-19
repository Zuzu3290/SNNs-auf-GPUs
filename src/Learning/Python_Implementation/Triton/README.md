# Triton SNN Starter

A Python-first starter project for building a **Triton-oriented spiking neural network (SNN)** workflow for GPU deep learning.

This project is organized as a normal CPython package and uses:

- **PyTorch** for model structure, autograd, and training loops
- **Triton** for custom GPU kernels
- A lightweight **LIF-style** spiking layer example
- A clean project structure suitable for extension toward hardware-conscious GPU optimization

## Why this structure?

Triton provides a Python-based environment for writing custom deep learning kernels, including `@triton.jit` kernels and autotuning primitives. It is designed to be used from Python and integrated with tensor frameworks such as PyTorch. citeturn967225search0turn967225search1turn967225search7turn967225search22

Spiking neural networks are usually trained in Python with surrogate-gradient style methods and GPU tensors, which makes a Python + PyTorch + Triton stack a reasonable starting point for experimentation. citeturn967225search2turn967225search5turn967225search13

## Project layout

```text
triton_snn_project/
├── pyproject.toml
├── README.md
├── examples/
│   └── run_demo.py
├── src/
│   └── triton_snn/
│       ├── __init__.py
│       ├── config.py
│       ├── train.py
│       ├── kernels/
│       │   ├── __init__.py
│       │   └── lif.py
│       └── models/
│           ├── __init__.py
│           └── snn.py
└── tests/
    └── test_smoke.py
```

## What the example does

This starter includes:

- a **Triton kernel** for one LIF-like update step
- a small **PyTorch module** that calls the Triton kernel
- a simple multi-step **SNN classifier**
- a tiny synthetic training loop

## Important note

This is a **starter architecture**, not a production-ready neuromorphic hardware toolchain.  
It targets **GPU execution via Triton/PyTorch**, not direct deployment to Loihi, FPGA, ASIC, or embedded neuromorphic hardware.

## Install

```bash
pip install -e .
```

## Run

```bash
python examples/run_demo.py
```
