# Container

Docker build environment for compiling the custom CRSC CUDA kernel
(`src/learning/setup.py`). Not required for normal training — only needed if
`kernel: ON` in `configuration/SNN_module.yaml`.

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | `ubuntu:22.04` + CUDA 12.8 toolkit (apt) + Python 3.10 + pinned `torch==2.10.0+cu128` |
| `docker.yaml` | Compose file — build + run with one command |
| `build_kernel_docker.bat` | Windows equivalent — build image, run kernel build, drop into a shell |

Source code is **mounted at run time**, not baked into the image — the kernel
build needs `nvidia-smi` for GPU compute-capability detection, which only
resolves once the container actually has `--gpus all`/GPU access at `docker run`,
not during `docker build`.

## Usage

**Docker Compose** (from repo root):

```bash
docker compose -f container/docker.yaml up --build
```

**Windows batch script** (from anywhere):

```cmd
container\build_kernel_docker.bat
```

Both build the image, mount the repo into `/workspace`, and run
`python src/learning/setup.py build_ext --inplace` inside the container, then
drop you into a shell.

## After the kernel builds

The kernel build only needs to happen once (until the kernel source changes).
Training itself does **not** run through Docker — use the normal launchers at
the repo root:

```bash
./launch.sh      # Linux/Colab
launch.bat       # Windows
```

These just run `python src/learning/main.py` directly; they don't touch this
folder. Set `kernel: ON` in `configuration/SNN_module.yaml` first so training
picks up the compiled kernel instead of the default framework forward pass.
