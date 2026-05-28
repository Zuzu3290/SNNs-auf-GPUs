#include "energy_management.h"
#include <cstdio>

// Conditional NVML: header lives in the CUDA toolkit (cuda/include/nvml.h).
// The stubs below let the file compile and link when NVML is absent; the
// profiler then reports timing-only results without power readings.
#if __has_include(<nvml.h>)
  #include <nvml.h>
  #define SNN_HAS_NVML 1
#else
  #define SNN_HAS_NVML 0
  typedef void* nvmlDevice_t;
  #define NVML_SUCCESS 0
  static inline int nvmlInit()                                            { return -1; }
  static inline int nvmlDeviceGetHandleByIndex(unsigned, nvmlDevice_t*)   { return -1; }
  static inline int nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int*)  { return -1; }
#endif

// --- Process-wide NVML state (init once, never shut down mid-run) --------
static bool         s_nvml_init   = false;
static nvmlDevice_t s_nvml_device = nullptr;

static void ensure_nvml_init() {
    if (s_nvml_init) return;
#if SNN_HAS_NVML
    if (nvmlInit() == NVML_SUCCESS) {
        nvmlDeviceGetHandleByIndex(0, &s_nvml_device);
        s_nvml_init = true;
    }
#endif
}

static float read_power_mw() {
#if SNN_HAS_NVML
    if (!s_nvml_device) return 0.f;
    unsigned int mw = 0;
    if (nvmlDeviceGetPowerUsage(s_nvml_device, &mw) == NVML_SUCCESS)
        return static_cast<float>(mw);
#endif
    return 0.f;
}

// -------------------------------------------------------------------------

EnergyProfiler::EnergyProfiler()
    : power_before_mw_(0.f), nvml_ok_(false), nvml_device_(nullptr)
{
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_stop_);
    ensure_nvml_init();
    nvml_ok_     = s_nvml_init;
    nvml_device_ = static_cast<void*>(s_nvml_device);
}

EnergyProfiler::~EnergyProfiler() {
    cudaEventDestroy(ev_start_);
    cudaEventDestroy(ev_stop_);
}

void EnergyProfiler::start(cudaStream_t stream) {
    power_before_mw_ = nvml_ok_ ? read_power_mw() : 0.f;
    cudaEventRecord(ev_start_, stream);
}

KernelEnergyResult EnergyProfiler::stop(cudaStream_t stream) {
    cudaEventRecord(ev_stop_, stream);
    cudaEventSynchronize(ev_stop_);

    KernelEnergyResult r{};
    cudaEventElapsedTime(&r.elapsed_ms, ev_start_, ev_stop_);

    r.power_before_mw = power_before_mw_;
    r.power_after_mw  = nvml_ok_ ? read_power_mw() : 0.f;
    r.nvml_available  = nvml_ok_;

    if (nvml_ok_) {
        // energy (J) = average_power (W) * time (s)  =>  convert to mJ
        float avg_w  = (r.power_before_mw + r.power_after_mw) * 0.5f * 1e-3f;
        r.energy_mj  = avg_w * (r.elapsed_ms * 1e-3f) * 1e3f;
    }
    return r;
}

void EnergyProfiler::print_result(const KernelEnergyResult& r, const char* label) {
    printf("[EnergyProfiler] %s\n", label);
    printf("  Kernel time  : %.3f ms\n", r.elapsed_ms);
    if (r.nvml_available) {
        printf("  Power before : %.1f mW\n", r.power_before_mw);
        printf("  Power after  : %.1f mW\n", r.power_after_mw);
        printf("  Energy used  : %.4f mJ\n", r.energy_mj);
    } else {
        printf("  Power/Energy : N/A (compile with NVML headers for real readings)\n");
    }
}
