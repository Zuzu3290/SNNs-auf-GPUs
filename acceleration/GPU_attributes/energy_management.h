#pragma once
#include <cuda_runtime.h>

// Energy consumed and timing for one kernel invocation
struct KernelEnergyResult {
    float elapsed_ms;       // CUDA-event-measured kernel time (ms)
    float power_before_mw;  // GPU power before launch in mW  (0 if NVML absent)
    float power_after_mw;   // GPU power after sync in mW     (0 if NVML absent)
    float energy_mj;        // avg_power_W * elapsed_s * 1000 (0 if NVML absent)
    bool  nvml_available;   // true = real hardware readings; false = timing-only
};

// RAII scoped profiler: call start() before the kernel, stop() after.
// NVML is initialised lazily once per process via a static; no double-init risk.
class EnergyProfiler {
public:
    EnergyProfiler();
    ~EnergyProfiler();

    // Record the start event and sample power (if NVML is available)
    void start(cudaStream_t stream = nullptr);

    // Record stop event, synchronise, compute result and return it
    KernelEnergyResult stop(cudaStream_t stream = nullptr);

    // Print a human-readable summary to stdout
    static void print_result(const KernelEnergyResult& r, const char* label = "kernel");

private:
    cudaEvent_t ev_start_;
    cudaEvent_t ev_stop_;
    float       power_before_mw_;
    bool        nvml_ok_;
    void*       nvml_device_;   // opaque nvmlDevice_t; avoids exposing nvml.h here
};
