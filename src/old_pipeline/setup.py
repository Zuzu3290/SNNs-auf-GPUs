from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="snn_cuda",
    ext_modules=[
        CUDAExtension(
            name="snn_cuda.snn_forward",
            sources=[
                "kernels/snn_forward.cpp",
                "kernels/snn_forward.cu",
            ],
            include_dirs=["/usr/local/cuda/include"],
            library_dirs=["/usr/local/cuda/lib64"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)