# Event Camera SNN Pipeline — ARM + Python

Real-time event camera → spiking neural network pipeline, with hot-path
processing implemented in ARM64 assembly (NEON SIMD), orchestrated by GNU Make.

```
┌─────────────────────────────────────────────────────┐
│                     make run                        │
│  pipeline.py  (Python orchestrator)                 │
│    │                                                │
│    ├── Module 1+2  camera / driver (Python)         │
│    ├── Module 3    event buffer    (Python)         │
│    ├── Module 4    GPU memory      (Python/PyTorch) │
│    ├── Module 5 ──►spike_kernel.c ──► event_kernel.s│ ◄─ ARM64 NEON
│    ├── Module 6    SNN model       (snnTorch)       │
│    └── Module 7    decoder/output  (Python)         │
└─────────────────────────────────────────────────────┘
```

---

## Repo structure

```
event_camera_arm/
├── Makefile                   ← build system (start here)
├── pipeline.py                ← Python master pipeline
├── requirements.txt
│
├── asm/
│   └── event_kernel.s         ← ARM64 NEON assembly kernels
│       ├── event_threshold_neon
│       ├── lif_membrane_neon
│       ├── voxel_accumulate
│       ├── polarity_split
│       ├── event_count_xy
│       ├── dot_product_f32_neon
│       └── relu_inplace_neon
│
├── include/
│   ├── arm_kernels.h          ← C declarations for asm functions
│   └── event_types.h          ← shared data structures (EventBatch, VoxelGrid)
│
├── src/
│   ├── pipeline.c             ← native C entry point (no Python)
│   └── kernel/
│       └── spike_kernel.c     ← C bridge: NEON dispatch + scalar fallback
│
├── scripts/
│   ├── setup.sh               ← system setup (apt + CUDA + libcaer)
│   └── check_env.py           ← Python env validator
│
├── tests/
│   └── test_asm_kernels.py    ← ctypes-based tests for asm kernels
│
└── docker/
    ├── Dockerfile             ← multi-stage: builder + GPU runtime
    └── docker-compose.yml     ← camera-bridge + snn-compute services
```

---

## Quick start

```bash
# 1. Full system setup (Ubuntu/Debian — run once as root)
sudo bash scripts/setup.sh

# 2. Python environment
make env

# 3. Build C + ARM assembly
make all

# 4. Run (mock camera — no hardware needed)
make run

# 5. Run native C binary (no Python)
make run-arm
```

---

## Make targets

| Target | What it does |
|--------|-------------|
| `make all` | Build everything |
| `make asm` | Assemble ARM `.s` files only |
| `make c` | Compile C files only |
| `make run` | Run Python pipeline (mock camera) |
| `make run-arm` | Run native C/ARM binary |
| `make run-pyaer` | Run with iniVation camera |
| `make run-mv` | Run with Prophesee camera |
| `make test` | Run all tests |
| `make test-asm` | Test ARM kernels via ctypes |
| `make env` | Create Python venv |
| `make setup` | Full system setup |
| `make docker` | Build Docker image |
| `make clean` | Remove build artifacts |
| `make info` | Show toolchain info |

---

## ARM Assembly kernels (`asm/event_kernel.s`)

Each kernel is written in ARM64 AArch64 with NEON SIMD.

| Kernel | Purpose | SIMD width |
|--------|---------|-----------|
| `event_threshold_neon` | float array → binary spikes | 16 floats/iter |
| `lif_membrane_neon` | LIF membrane update per pixel | 4 pixels/iter |
| `voxel_accumulate` | scatter events into (T,H,W) grid | scalar (indexed) |
| `polarity_split` | partition ON/OFF events | scalar (branching) |
| `event_count_xy` | 2D event histogram | scalar (scatter) |
| `dot_product_f32_neon` | fused dot product for FC layer | 8 floats/iter |
| `relu_inplace_neon` | in-place ReLU | 16 floats/iter |

---

## Environment layers

```
┌──────────────────────────────────────┐
│  OS / hardware (managed by setup.sh) │
│  • CUDA drivers + toolkit            │
│  • libcaer (iniVation cameras)       │
│  • Metavision SDK (Prophesee)        │
│  • udev rules + plugdev group        │
├──────────────────────────────────────┤
│  Python env  (managed by make env)   │
│  • torch, snntorch, numpy, cupy      │
│  • pyaer (iniVation Python binding)  │
├──────────────────────────────────────┤
│  Native build  (managed by make all) │
│  • ARM asm → .o via `as`             │
│  • C sources → .o via `gcc`          │
│  • Linked into pipeline_c binary     │
├──────────────────────────────────────┤
│  Application  (you write this)       │
│  • pipeline.py  — Python pipeline    │
│  • pipeline_c   — C/ARM pipeline     │
└──────────────────────────────────────┘
```

---

## Cross-compilation (x86 → ARM64)

```bash
# Install cross-compiler
sudo apt install gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu

# Build for ARM64 from x86 host
make all CROSS=1

# The binary runs on: Jetson Orin, Raspberry Pi 5, any aarch64 Linux
```

---

## Docker

```bash
# Build image
make docker

# Run with GPU + USB camera
docker run --rm --gpus all \
  --device=/dev/bus/usb \
  -v $(pwd):/workspace \
  event-camera-snn:latest make run-pyaer

# Two-service split (camera on edge, SNN on GPU server)
cd docker && docker compose up
```
