#include <torch/extension.h>

// Declarations from snn_forward.cu
torch::Tensor snn_forward_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
);

std::tuple<torch::Tensor, float, float, float, float, bool>
snn_forward_profiled(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SNN LIF CUDA extension with GPU energy profiling";

    m.def("forward",
          &snn_forward_cuda,
          "LIF forward pass (CUDA). Returns spike tensor [B, N, T].",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);

    m.def("forward_profiled",
          &snn_forward_profiled,
          "LIF forward pass with GPU energy profiling.\n"
          "Returns (spikes, elapsed_ms, power_before_mw, power_after_mw, energy_mj, nvml_available).\n"
          "energy_mj is 0 when NVML is not available; elapsed_ms is always valid.",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);
}
