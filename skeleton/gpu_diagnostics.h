// GPU preflight diagnostic suite — runs once before training to validate hardware state.
// Checks CUDA device properties, pending error state, VRAM headroom, and NVML health
// signals (temperature, ECC errors, throttle reasons, power draw).
// Lives in skeleton/ — infrastructure layer, not kernel-specific.
// Compiled into the snn_runtime extension alongside CUDAMemoryArbiter.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Throttle reason bitmask — mirrors NVML_CLOCKS_THROTTLE_REASON_* values.
enum SnnThrottleReason : uint32_t {
    SNN_THROTTLE_NONE           = 0x00,
    SNN_THROTTLE_GPU_IDLE       = 0x01,
    SNN_THROTTLE_APP_CLOCK      = 0x02,
    SNN_THROTTLE_SW_POWER_CAP   = 0x04,
    SNN_THROTTLE_HW_SLOWDOWN    = 0x08,
    SNN_THROTTLE_SYNC_BOOST     = 0x10,
    SNN_THROTTLE_SW_THERMAL     = 0x20,
    SNN_THROTTLE_HW_THERMAL     = 0x40,
    SNN_THROTTLE_HW_POWER_BRAKE = 0x80,
};

struct GpuDiagnosticReport {
    // --- CUDA device properties -------------------------------------------
    char     device_name[256];
    int      sm_count;
    int      compute_major;
    int      compute_minor;
    size_t   total_vram_bytes;
    size_t   free_vram_bytes;
    bool     ecc_supported;
    bool     cuda_error_cleared;  // true if a stale CUDA error was found and cleared

    // --- NVML fields (valid only when nvml_available == true) -------------
    bool     nvml_available;
    float    temperature_c;
    float    power_draw_w;
    float    power_limit_w;
    uint64_t ecc_uncorrected;    // > 0 means GPU needs retirement
    uint64_t ecc_corrected;      // rising count is a warning signal
    uint32_t throttle_reasons;   // bitmask of active throttle reasons

    // --- Overall verdict --------------------------------------------------
    bool     healthy;
    char     failure_reason[512];
};

GpuDiagnosticReport run_gpu_preflight(int device_idx = 0);
void                print_diagnostic_report(const GpuDiagnosticReport& r);
