#include <torch/extension.h>
#include "lif_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused LIF CUDA kernel: membrane_update + threshold + spike_gen + reset in one pass.";
    m.def("lif_forward",  &lif_fused_forward,  "Fused LIF forward pass");
    m.def("lif_backward", &lif_fused_backward, "Fused LIF backward pass (surrogate gradient)");
}
