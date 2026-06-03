// pybind11 binding layer: exposes snn_forward_cuda, snn_forward_profiled, lif_temporal_cuda,
// and lif_temporal_ballot_cuda to Python. Kept in snn_binding.cpp (not snn_forward.cpp) to
// avoid duplicate-symbol linker errors from matching base names with snn_forward.cu.
#include <torch/extension.h>

// --- snn_forward.cu ---
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

// --- lif_temporal.cu ---
torch::Tensor lif_temporal_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
);

std::pair<torch::Tensor, torch::Tensor>
lif_temporal_ballot_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
);

// --- lif_warp_oriented.cu ---
#include "lif_warp_oriented.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SNN LIF CUDA extension — basic, profiled, and temporal-register kernels";

    // ---- Basic path (lif_basic, fixed 256 threads, kept for benchmarking) ----
    m.def("forward",
          &snn_forward_cuda,
          "Basic LIF forward (CUDA). One thread per (b,n,t). Returns spikes [B,N,T].",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);

    m.def("forward_profiled",
          &snn_forward_profiled,
          "Basic LIF forward with NVML energy profiling.\n"
          "Returns (spikes, elapsed_ms, pwr_before_mw, pwr_after_mw, energy_mj, nvml_ok).",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);

    // ---- Temporal path (T-in-register, one thread per neuron, correct LIF) ----
    m.def("temporal_forward",
          &lif_temporal_cuda,
          "Temporal LIF forward (CUDA). One thread per (b,n), T loop in register.\n"
          "Correct sequential dynamics; eliminates race condition in basic forward.\n"
          "Returns spikes [B, N, T].",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);

    m.def("temporal_forward_ballot",
          &lif_temporal_ballot_cuda,
          "Temporal LIF forward with warp ballot output.\n"
          "Returns (spikes [B,N,T], spike_mask [ceil(B*N/32), T] int32).\n"
          "spike_mask[w,t] is the 32-bit __ballot_sync result for warp w at timestep t.",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")    = 1.0f,
          py::arg("tau_inv") = 0.1f);

    // ---- Warp-oriented path (SM-saturation grid-stride, correct LIF) ----
    m.def("warp_oriented_forward",
          [](torch::Tensor input, torch::Tensor voltage,
             float v_th, float tau_inv, int target_bps) {
              auto r = lif_warp_oriented_cuda(input, voltage, v_th, tau_inv, target_bps);
              return py::make_tuple(r.spikes, r.elapsed_ms,
                                   r.blocks_launched, r.neurons_per_thread);
          },
          "SM-saturating warp-oriented LIF forward.\n"
          "Fills all SMs via grid-stride loop; correct sequential LIF at any B*N size.\n"
          "Returns (spikes [B,N,T], elapsed_ms, blocks_launched, neurons_per_thread).",
          py::arg("input"),
          py::arg("voltage"),
          py::arg("v_th")              = 1.0f,
          py::arg("tau_inv")           = 0.1f,
          py::arg("target_blocks_per_sm") = 8);
}
