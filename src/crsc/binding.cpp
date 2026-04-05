#include <torch/extension.h>

// 1. Declare the function defined in your .cu file
torch::Tensor snn_forward(torch::Tensor mem, torch::Tensor input, float threshold);

// 2. Bind the function to a Python name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &snn_forward, "SNN Forward Update (CUDA)");
}