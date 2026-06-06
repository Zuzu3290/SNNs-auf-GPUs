// GPU preflight diagnostic implementation.
// NVML calls guarded by SNN_HAS_NVML — on Colab free tier the NVML section
// is skipped and nvml_available is set to false in the report.
#include "gpu_diagnostics.h"

#ifndef SNN_HAS_NVML
  #if __has_include(<nvml.h>)
    #define SNN_HAS_NVML 1
  #else
    #define SNN_HAS_NVML 0
  #endif
#endif

#if SNN_HAS_NVML
  #include <nvml.h>
#endif

#include <cstdio>
#include <cstring>

static void set_failure(GpuDiagnosticReport& r, const char* msg) {
    r.healthy = false;
    snprintf(r.failure_reason, sizeof(r.failure_reason), "%s", msg);
}

GpuDiagnosticReport run_gpu_preflight(int device_idx) {
    GpuDiagnosticReport r{};
    r.healthy           = true;
    r.failure_reason[0] = '\0';

    // --- CUDA device properties -------------------------------------------
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_idx);
    if (err != cudaSuccess) {
        set_failure(r, cudaGetErrorString(err));
        return r;
    }
    snprintf(r.device_name, sizeof(r.device_name), "%s", prop.name);
    r.sm_count         = prop.multiProcessorCount;
    r.compute_major    = prop.major;
    r.compute_minor    = prop.minor;
    r.total_vram_bytes = prop.totalGlobalMem;
    r.ecc_supported    = (prop.ECCEnabled != 0);

    // --- VRAM headroom ----------------------------------------------------
    size_t free_bytes = 0, total_bytes = 0;
    cudaSetDevice(device_idx);
    cudaMemGetInfo(&free_bytes, &total_bytes);
    r.free_vram_bytes  = free_bytes;
    r.total_vram_bytes = total_bytes;

    if (total_bytes > 0 && (100.0 * free_bytes / total_bytes) < 10.0)
        set_failure(r, "VRAM critically low — less than 10% free before training");

    // --- Clear stale CUDA error state -------------------------------------
    if (cudaGetLastError() != cudaSuccess)
        r.cuda_error_cleared = true;

#if SNN_HAS_NVML
    // --- NVML health signals ----------------------------------------------
    if (nvmlInit_v2() == NVML_SUCCESS) {
        r.nvml_available = true;
        nvmlDevice_t dev;

        if (nvmlDeviceGetHandleByIndex_v2(device_idx, &dev) == NVML_SUCCESS) {

            unsigned int temp = 0;
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                r.temperature_c = (float)temp;
                if (temp >= 90 && r.healthy)
                    set_failure(r, "GPU temperature >= 90 C — thermal throttling imminent");
            }

            unsigned int pmw = 0, plmw = 0;
            if (nvmlDeviceGetPowerUsage(dev, &pmw)             == NVML_SUCCESS) r.power_draw_w  = pmw  / 1000.f;
            if (nvmlDeviceGetPowerManagementLimit(dev, &plmw)  == NVML_SUCCESS) r.power_limit_w = plmw / 1000.f;

            unsigned long long unc = 0, cor = 0;
            if (nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                    NVML_AGGREGATE_ECC_COUNTER, &unc) == NVML_SUCCESS) {
                r.ecc_uncorrected = (uint64_t)unc;
                if (unc > 0 && r.healthy)
                    set_failure(r, "Uncorrectable ECC memory errors — GPU requires retirement");
            }
            if (nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                    NVML_AGGREGATE_ECC_COUNTER, &cor) == NVML_SUCCESS)
                r.ecc_corrected = (uint64_t)cor;

            unsigned long long throttle = 0;
            if (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &throttle) == NVML_SUCCESS) {
                r.throttle_reasons = (uint32_t)throttle;
                const uint32_t hard = SNN_THROTTLE_HW_SLOWDOWN
                                    | SNN_THROTTLE_HW_THERMAL
                                    | SNN_THROTTLE_HW_POWER_BRAKE;
                if ((r.throttle_reasons & hard) && r.healthy)
                    set_failure(r, "Hardware clock throttle active — check cooling and power");
            }
        }
        nvmlShutdown();
    }
#endif
    return r;
}

void print_diagnostic_report(const GpuDiagnosticReport& r) {
    printf("\n[GPU Preflight] %s\n", r.healthy ? "HEALTHY" : "ISSUES DETECTED");
    printf("  Device      : %s\n", r.device_name);
    printf("  Compute     : sm_%d%d   SMs: %d\n", r.compute_major, r.compute_minor, r.sm_count);
    printf("  VRAM free   : %.0f / %.0f MB\n",
           r.free_vram_bytes  / (1024.0 * 1024.0),
           r.total_vram_bytes / (1024.0 * 1024.0));
    if (r.cuda_error_cleared)
        printf("  [WARN] Stale CUDA error cleared from previous session\n");
    if (r.nvml_available) {
        printf("  Temp        : %.0f C\n",           r.temperature_c);
        printf("  Power       : %.0f / %.0f W\n",    r.power_draw_w, r.power_limit_w);
        printf("  ECC unc/cor : %llu / %llu\n",
               (unsigned long long)r.ecc_uncorrected,
               (unsigned long long)r.ecc_corrected);
        printf("  Throttle    : 0x%02X%s\n", r.throttle_reasons,
               r.throttle_reasons == 0 ? " (none)" : "");
    } else {
        printf("  NVML        : unavailable (free GPU tier)\n");
    }
    if (!r.healthy)
        printf("  FAILURE     : %s\n", r.failure_reason);
    printf("\n");
}
