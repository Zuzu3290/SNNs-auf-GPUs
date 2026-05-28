import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE      = os.path.dirname(os.path.abspath(__file__))       # src/learning/
ROOT      = os.path.abspath(os.path.join(HERE, '..', '..'))  # project root
GPU_ATTRS = os.path.join(ROOT, 'acceleration', 'GPU_attributes')
CRSC_KERN = os.path.join(ROOT, 'src', 'crsc', 'kernels')

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
                os.path.join(GPU_ATTRS, "energy_management.cu"),
                os.path.join(GPU_ATTRS, "memory_management.cu"),
                os.path.join(GPU_ATTRS, "throughput_optimiation.cu"),
            ],
            include_dirs=[GPU_ATTRS] + _cuda_inc,
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
