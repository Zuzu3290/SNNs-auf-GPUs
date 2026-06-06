import os
import subprocess
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Bypass PyTorch's CUDA version check for compatibility
import torch.utils.cpp_extension as cpp_ext
original_check = getattr(cpp_ext, '_check_cuda_version', None)
if original_check:
    cpp_ext._check_cuda_version = lambda *args, **kwargs: None

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.abspath(os.path.join(HERE, '..', '..'))
GPU_ATTRS = os.path.join(ROOT, 'acceleration', 'GPU_attributes')
CRSC_KERN = os.path.join(ROOT, 'src', 'crsc', 'kernels')
RUNTIME   = os.path.join(ROOT, 'src', 'runtime')
SKEL      = os.path.join(ROOT, 'skeleton')

def get_compute_capability_from_nvidia_smi():
    """Auto-detect GPU compute capability from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split('\n')[0]
            major, minor = cap.split('.')
            return int(major), int(minor)
    except Exception:
        pass
    return 8, 0  # fallback to Ampere

def get_cuda_version():
    """Get CUDA version from nvcc that's actually in PATH."""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        print(f"[BUILD] nvcc output: {output[:200]}")
        if 'release' in output:
            version_str = output.split('release')[-1].split(',')[0].split()[0].strip()
            major = int(version_str.split('.')[0])
            minor = int(version_str.split('.')[1] if '.' in version_str else 0)
            print(f"[BUILD] Detected CUDA {major}.{minor}")
            return major
    except Exception as e:
        print(f"[BUILD] Failed to detect CUDA: {e}")
    return 12

major, minor = get_compute_capability_from_nvidia_smi()
cuda_major = get_cuda_version()

# Map compute capability to supported architecture for this CUDA version
def map_to_supported_arch(major, minor, cuda_major):
    """Map GPU compute capability to arch string CUDA supports."""
    gpu_cc = f"{major}{minor}"
    print(f"[BUILD] GPU compute capability: {gpu_cc}, CUDA version: {cuda_major}")
    
    # CUDA 12.x supports up to Ada (compute 89)
    if cuda_major == 12:
        if (major, minor) >= (9, 0):  # Hopper+ → fallback to Ada
            print(f"[BUILD] CUDA 12.x doesn't support Blackwell compute {gpu_cc}, using Ada (8.9) as fallback")
            return "89"
        elif (major, minor) >= (8, 0):  # Ampere or Ada
            print(f"[BUILD] CUDA 12.x supports compute {gpu_cc}")
            return gpu_cc
        else:
            return "75"  # Turing fallback
    # CUDA 13.x has PyTorch compatibility issues, so we still fallback to Ada
    elif cuda_major >= 13:
        if (major, minor) >= (9, 0):  # Hopper+ has PyTorch header issues in CUDA 13
            print(f"[BUILD] Using CUDA 12.6 fallback with Ada (8.9) due to PyTorch compatibility")
            return "89"
        else:
            print(f"[BUILD] CUDA 13.x supports compute {gpu_cc}")
            return gpu_cc
    else:
        return "89"

compute_arch = map_to_supported_arch(major, minor, cuda_major)
print(f"[BUILD] Using compute architecture: {compute_arch}")

try:
    from torch.utils.cpp_extension import CUDA_HOME
    _cuda_inc = [os.path.join(CUDA_HOME, 'include')] if CUDA_HOME else []
except Exception:
    _cuda_inc = []

setup(
    name="snn_cuda",
    ext_modules=[
        CUDAExtension(
            name="snn_cuda.snn_forward",
            sources=[
                os.path.join(CRSC_KERN, "snn_forward.cpp"),
                os.path.join(CRSC_KERN, "snn_forward.cu"),
            ],
            include_dirs=[GPU_ATTRS] + _cuda_inc,
            extra_compile_args={"cxx": _cxx_flags, "nvcc": _nvcc_flags},
        ),
        # --- Runtime extension: CUDAMemoryArbiter + cuMemPool API -------
        CUDAExtension(
            name="snn_runtime",
            sources=[
                os.path.join(RUNTIME,   "runtime_binding.cpp"),
                os.path.join(GPU_ATTRS, "memory_arbiter.cu"),
                os.path.join(SKEL,      "gpu_diagnostics.cu"),
            ],
            include_dirs=[GPU_ATTRS, SKEL] + _cuda_inc,
            extra_compile_args={"cxx": _cxx_flags, "nvcc": _nvcc_flags},
            extra_compile_args={
                "cxx":  ["-O3", "/Zc:preprocessor"],
                "nvcc": ["-O3", "--use_fast_math", "-allow-unsupported-compiler", f"-gencode=arch=compute_{compute_arch},code=sm_{compute_arch}", "-Xcompiler=/Zc:preprocessor"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
