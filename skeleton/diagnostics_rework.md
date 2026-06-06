Overall evaluation

The design is sound: a C++/CUDA preflight collects hardware state, the Python wrapper integrates it into training/reliability logic, and NVML is optional. However, the current version has several correctness and production-readiness issues.

Verdict: good prototype, but I would not use it as a hard production training gate until the issues below are fixed.

Critical / high-priority issues
1. CUDA device index may not match NVML device index

This is the most important issue.

nvmlDeviceGetHandleByIndex_v2(device_idx, &dev)

CUDA device indices are affected by CUDA_VISIBLE_DEVICES, device ordering, and MIG. NVML physical indices are not guaranteed to match CUDA indices. This can cause the diagnostic to check the wrong GPU.

Fix: map CUDA device to NVML device using PCI bus ID.

char pci_bus_id[32] = {};
cudaError_t pci_err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_idx);

if (pci_err == cudaSuccess) {
    if (nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &dev) == NVML_SUCCESS) {
        // query NVML fields
    }
}

For MIG environments, you may need additional MIG-aware NVML handling.

2. cudaSetDevice() and cudaMemGetInfo() are not checked

Current code:

cudaSetDevice(device_idx);
cudaMemGetInfo(&free_bytes, &total_bytes);

If either call fails, the report can contain zero VRAM values while still marking the GPU healthy.

Fix:

err = cudaSetDevice(device_idx);
if (err != cudaSuccess) {
    set_failure(r, cudaGetErrorString(err));
    return r;
}

err = cudaMemGetInfo(&free_bytes, &total_bytes);
if (err != cudaSuccess) {
    set_failure(r, cudaGetErrorString(err));
    return r;
}
3. CUDA stale error handling is in the wrong place

Current code clears/checks CUDA error state after several CUDA calls:

if (cudaGetLastError() != cudaSuccess)
    r.cuda_error_cleared = true;

That can accidentally clear an error caused by the diagnostic itself, especially since cudaMemGetInfo() is not checked.

Better sequence:

cudaError_t stale = cudaGetLastError();
if (stale != cudaSuccess) {
    r.cuda_error_cleared = true;
}

Then perform the actual diagnostic calls and check every return value directly.

Also, cudaGetLastError() only reflects the CUDA runtime error state for the current host thread. It does not reliably describe a “previous session” across processes.

The Python warning should probably say:

"Stale CUDA error cleared at startup — check earlier CUDA work in this process"

not “previous session logs.”

4. NVML compile detection is fragile

This block:

#ifndef SNN_HAS_NVML
  #if __has_include(<nvml.h>)
    #define SNN_HAS_NVML 1
  #else
    #define SNN_HAS_NVML 0
  #endif
#endif

detects whether the header exists, not whether the program can link against NVML. If nvml.h is present but libnvidia-ml is not linked, the extension can fail to build.

Better options:

Let the build system define SNN_HAS_NVML only when both header and library are available.
Use runtime dynamic loading with dlopen("libnvidia-ml.so.1") if you want NVML to be truly optional.
Default to disabled unless explicitly enabled:
#ifndef SNN_HAS_NVML
#define SNN_HAS_NVML 0
#endif
5. ecc_supported is misnamed
r.ecc_supported = (prop.ECCEnabled != 0);

prop.ECCEnabled means ECC is currently enabled, not necessarily supported.

Rename the field:

bool ecc_enabled;

If you actually need support status, query it through NVML, where available.

Medium-priority issues
6. Throttle reasons should be uint64_t, not uint32_t

NVML returns throttle reasons as unsigned long long.

Current code:

uint32_t throttle_reasons;
r.throttle_reasons = (uint32_t)throttle;

The current low-bit values fit, but this truncates future or extended throttle flags.

Use:

uint64_t throttle_reasons;

and:

enum SnnThrottleReason : uint64_t { ... };
7. NVML unavailable message is too specific

Current printout:

printf("  NVML        : unavailable (free GPU tier)\n");

NVML can be unavailable for many reasons:

Build disabled it.
Header/library missing.
Driver issue.
Permission issue.
Container restriction.
Unsupported environment.
Free GPU tier.

Better:

printf("  NVML        : unavailable or disabled\n");

Optionally add a field:

char nvml_status[256];
8. NVML init success but handle failure is misleading

Current logic:

if (nvmlInit_v2() == NVML_SUCCESS) {
    r.nvml_available = true;

    if (nvmlDeviceGetHandleByIndex_v2(device_idx, &dev) == NVML_SUCCESS) {
        // fields populated
    }

    nvmlShutdown();
}

If NVML initializes but the handle lookup fails, nvml_available is true, but all NVML values stay zero. That can look like a valid reading.

Add a status message or mark NVML as partially unavailable.

9. Correctable ECC count should probably be treated as a delta

This Python logic:

if report.nvml_available and report.ecc_corrected > 0:
    reliability.record_interruption(...)

uses the aggregate historical ECC counter. That may repeatedly warn about old corrected ECC events.

Better behavior:

Store last-seen corrected ECC count.
Warn only if the count increases.
Treat uncorrected ECC more strictly.
10. Python wrapper fails open if snn_runtime is missing

Current behavior:

except ImportError:
    print("[GPU Preflight] snn_runtime not built — skipping hardware checks.")
    return True

For development, this is convenient. For production, this is risky because training proceeds without the hardware gate.

Consider:

require_runtime: bool = False

Then:

except ImportError as exc:
    msg = "[GPU Preflight] snn_runtime not built — hardware checks skipped."
    if reliability is not None:
        reliability.record_interruption(msg)
    if require_runtime or block_on_failure:
        raise RuntimeError(msg) from exc
    print(msg)
    return True

Depending on your intended behavior, you may want block_on_failure=True to fail closed when the diagnostic runtime is unavailable.

Lower-priority issues
11. Only one failure reason is retained

set_failure() stores one failure reason. Multiple issues are possible: low VRAM, ECC errors, thermal throttling, etc.

Current behavior is acceptable for a first failure, but a diagnostic report is more useful with separate warnings/failures.

Possible improvement:

char warnings[1024];
char failures[1024];

or expose a Python-side list from the binding.

12. Temperature and throttle policy may be too coarse

Current hard failure:

if (temp >= 90 && r.healthy)
    set_failure(r, "GPU temperature >= 90 C — thermal throttling imminent");

Python warning:

if report.temperature_c >= 80:
    reliability.record_interruption(...)

This is reasonable, but policies vary by GPU model. Some accelerators have different operating ranges. Consider making thresholds configurable.

13. SW_THERMAL throttle should probably be treated as serious

Your hard throttle mask currently includes:

SNN_THROTTLE_HW_SLOWDOWN
| SNN_THROTTLE_HW_THERMAL
| SNN_THROTTLE_HW_POWER_BRAKE;

You may also want to treat software thermal slowdown as at least a warning, possibly a failure:

SNN_THROTTLE_SW_THERMAL
File-by-file evaluation
1. Header file
Good
Clear report struct.
Fixed-size fields are simple to expose through a C++ extension.
Default function API is clean:
GpuDiagnosticReport run_gpu_preflight(int device_idx = 0);
void print_diagnostic_report(const GpuDiagnosticReport& r);
Change
bool ecc_supported;
uint32_t throttle_reasons;

to:

bool     ecc_enabled;
uint64_t throttle_reasons;

Also consider adding:

bool nvml_handle_valid;
char nvml_reason[256];
2. Python wrapper
Good
Simple training gate.
Correctly integrates with ReliabilityTracker.
block_on_failure is useful.
Type-check-only import is clean.
Change

Add handling for runtime diagnostic failures, not just import failure:

try:
    report = rt.run_gpu_preflight(device_idx)
except Exception as exc:
    msg = f"[GPU Preflight] Diagnostic failed: {exc}"
    if reliability is not None:
        reliability.record_failure(msg)
    if block_on_failure:
        raise RuntimeError(msg) from exc
    print(msg)
    return False

Also consider logging NVML absence:

if reliability is not None and not report.nvml_available:
    reliability.record_interruption(
        "NVML unavailable — reduced GPU health diagnostics"
    )
3. CUDA/NVML implementation
Good
Uses zero-initialized report.
Handles no-NVML builds.
Checks ECC, thermal, power, throttle, and VRAM.
Keeps print formatting simple.
Must fix
Check cudaSetDevice().
Check cudaMemGetInfo().
Move stale CUDA error clearing to the start.
Map CUDA device to NVML device by PCI bus ID.
Avoid assuming NVML index equals CUDA index.
Avoid header-only NVML detection unless the build system also links NVML.


Before merging, fix these four items:

Map CUDA device to NVML device by PCI bus ID.
Check all CUDA return values.
Move stale CUDA error clearing to the beginning.
Make NVML build/link behavior explicit instead of relying only on __has_include.