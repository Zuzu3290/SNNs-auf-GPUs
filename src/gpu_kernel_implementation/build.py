"""
Build script for the GPU event preprocessing CUDA extension.

Run once before using GPUEventPreprocessor:
    python src/gpu_kernel_implementation/build.py
"""
from torch.utils.cpp_extension import load
from pathlib import Path

HERE = Path(__file__).parent

snn_gpu_preproc = load(
    name="snn_gpu_preproc",
    sources=[str(HERE / "event_preprocessing.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

print("[build] snn_gpu_preproc built successfully.")
