import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE      = os.path.dirname(os.path.abspath(__file__))       # src/learning/
ROOT      = os.path.abspath(os.path.join(HERE, '..', '..'))  # project root
GPU_ATTRS = os.path.join(ROOT, 'acceleration', 'GPU_attributes')
CRSC_KERN = os.path.join(ROOT, 'src', 'crsc', 'kernels')
RUNTIME   = os.path.join(ROOT, 'src', 'runtime')
SKEL      = os.path.join(ROOT, 'skeleton')

try:
    from torch.utils.cpp_extension import CUDA_HOME
    _cuda_inc = [os.path.join(CUDA_HOME, 'include')] if CUDA_HOME else []
except Exception:
    _cuda_inc = []

_cxx_flags  = ["-O3", "-DSNN_HAS_NVML=0"]
_nvcc_flags = ["-O3", "--use_fast_math", "-DSNN_HAS_NVML=0"]

setup(
    name="snn_extensions",
    ext_modules=[
        # --- Kernel extension: LIF kernels + energy/memory/throughput ---
        CUDAExtension(
            name="snn_forward",
            sources=[
                os.path.join(CRSC_KERN, "snn_binding.cpp"),
                os.path.join(CRSC_KERN, "snn_forward.cu"),
                os.path.join(CRSC_KERN, "lif_temporal.cu"),
                os.path.join(CRSC_KERN, "lif_warp_oriented.cu"),
                os.path.join(GPU_ATTRS, "energy_management.cu"),
                os.path.join(GPU_ATTRS, "memory_management.cu"),
                os.path.join(GPU_ATTRS, "throughput_optimiation.cu"),
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
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
