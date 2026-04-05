import os
import torch
from torch.utils.cpp_extension import load

# 1. Point to your source files
_src_path = os.path.join(os.getcwd(), "csrc")
_sources = [
    os.path.join(_src_path, "binding.cpp"), 
    os.path.join(_src_path, "kernel.cu")
]

# 2. Compile and load on the fly
# This will search for 'nvcc' on your system path
snn_backend = load(
    name="snn_cuda_backend",
    sources=_sources,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True
)

# 3. Use your custom kernel in a PyTorch Module
class CustomSNN(torch.nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, mem, input_spikes):
        # Calls the C++ function defined in binding.cpp
        return snn_backend.forward(mem, input_spikes, self.threshold)
    

## with JIT loading 