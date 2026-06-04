// pybind11 binding layer for the src/runtime/ C++ components.
// Compiled as the 'snn_runtime' extension module (separate from snn_forward).
// Exposes CUDAMemoryArbiter and the SnnZone enum to Python.
#include <torch/extension.h>
#include "../../acceleration/GPU_attributes/memory_arbiter.h"
#include "../../skeleton/gpu_diagnostics.h"

namespace py = pybind11;

// Helper — convert Python float (MB) to bytes for the C++ API.
static inline size_t mb_to_bytes(double mb) {
    return static_cast<size_t>(mb * 1024.0 * 1024.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SNN runtime layer — CUDAMemoryArbiter with cuMemPool API control";

    // ---- Zone enum -------------------------------------------------------
    py::enum_<SnnZone>(m, "Zone",
            "VRAM zone identifiers used as keys for request() / release().")
        .value("DATASET_CACHE",    SNN_ZONE_DATASET_CACHE,
               "40 % of budget — soft only, cache eviction is tolerable.")
        .value("MODEL_PARAMS",     SNN_ZONE_MODEL_PARAMS,
               "30 % of budget — hard enforced, must not bleed into kernel workspace.")
        .value("KERNEL_WORKSPACE", SNN_ZONE_KERNEL_WORKSPACE,
               "20 % of budget — hard enforced, temporary kernel buffers.")
        .value("EMERGENCY",        SNN_ZONE_EMERGENCY,
               "10 % of budget — pinned as cuMemPool release threshold.")
        .export_values();

    // ---- CUDAMemoryArbiter -----------------------------------------------
    py::class_<CUDAMemoryArbiter>(m, "MemoryArbiter",
            "VRAM zone arbiter backed by the CUDA memory pool API.\n\n"
            "Partitions total VRAM into four named zones. Hard-enforced zones\n"
            "refuse request() when the hard limit (85 % of soft) would be exceeded.\n"
            "The emergency zone floor is pinned as the cuMemPool release threshold,\n"
            "guaranteeing fast re-allocation for burst kernel workspace needs.\n\n"
            "Usage::\n\n"
            "    from snn_runtime import MemoryArbiter, Zone\n"
            "    arb = MemoryArbiter(device_idx=0)\n"
            "    ok  = arb.request(Zone.KERNEL_WORKSPACE, mb=256)\n"
            "    arb.release(Zone.KERNEL_WORKSPACE, mb=256)\n"
            "    arb.print_status()")
        .def(py::init<int>(),
             py::arg("device_idx") = 0,
             "Initialise arbiter for device_idx. Queries VRAM once and sets\n"
             "the cuMemPool release threshold to the emergency zone soft limit.")

        // --- Allocation tracking -----------------------------------------
        .def("request",
             [](CUDAMemoryArbiter& self, SnnZone zone, double mb) {
                 return self.request(zone, mb_to_bytes(mb));
             },
             py::arg("zone"), py::arg("mb"),
             "Register mb of VRAM usage in zone.\n"
             "Returns False (no side-effect) if the hard limit would be exceeded.\n"
             "Warns to stderr when the soft limit is crossed.")
        .def("release",
             [](CUDAMemoryArbiter& self, SnnZone zone, double mb) {
                 self.release(zone, mb_to_bytes(mb));
             },
             py::arg("zone"), py::arg("mb"),
             "Deregister mb of VRAM usage from zone. Safe to over-call.")

        // --- Zone queries ------------------------------------------------
        .def("soft_limit_mb",
             [](const CUDAMemoryArbiter& self, SnnZone zone) {
                 return self.soft_limit(zone) / (1024.0 * 1024.0);
             },
             py::arg("zone"),
             "Soft limit for zone in MB (full zone allocation).")
        .def("hard_limit_mb",
             [](const CUDAMemoryArbiter& self, SnnZone zone) {
                 return self.hard_limit(zone) / (1024.0 * 1024.0);
             },
             py::arg("zone"),
             "Hard limit for zone in MB (85 % of soft — request() cap).")
        .def("used_mb",
             [](const CUDAMemoryArbiter& self, SnnZone zone) {
                 return self.used(zone) / (1024.0 * 1024.0);
             },
             py::arg("zone"),
             "Currently registered usage for zone in MB (atomic read).")
        .def("over_soft",
             &CUDAMemoryArbiter::over_soft,
             py::arg("zone"),
             "True if registered usage exceeds the soft limit.")

        // --- cuMemPool controls ------------------------------------------
        .def("set_release_threshold",
             [](CUDAMemoryArbiter& self, double mb) {
                 self.set_release_threshold(mb_to_bytes(mb));
             },
             py::arg("mb"),
             "Override the cuMemPool release threshold in MB.\n"
             "The pool retains at least this many bytes before releasing to OS.")
        .def("get_release_threshold_mb",
             [](const CUDAMemoryArbiter& self) {
                 return self.get_release_threshold() / (1024.0 * 1024.0);
             },
             "Current cuMemPool release threshold in MB.")

        // --- Diagnostics -------------------------------------------------
        .def("stats",
             [](const CUDAMemoryArbiter& self) {
                 auto s = self.stats();
                 const char* names[] = {
                     "dataset_cache", "model_params",
                     "kernel_workspace", "emergency"
                 };
                 py::list zones;
                 for (int i = 0; i < SNN_ZONE_COUNT; ++i) {
                     py::dict z;
                     z["name"]    = names[i];
                     z["used_mb"] = s.used_bytes[i]       / (1024.0 * 1024.0);
                     z["soft_mb"] = s.soft_limit_bytes[i] / (1024.0 * 1024.0);
                     z["hard_mb"] = s.hard_limit_bytes[i] / (1024.0 * 1024.0);
                     z["pct"]     = s.soft_limit_bytes[i] > 0
                                    ? s.used_bytes[i] * 100.0 / s.soft_limit_bytes[i]
                                    : 0.0;
                     zones.append(z);
                 }
                 py::dict d;
                 d["zones"]            = zones;
                 d["pool_used_mb"]     = s.pool_used_bytes      / (1024.0 * 1024.0);
                 d["pool_reserved_mb"] = s.pool_reserved_bytes  / (1024.0 * 1024.0);
                 d["pool_high_mb"]     = s.pool_used_high_bytes / (1024.0 * 1024.0);
                 return d;
             },
             "Return a dict with per-zone accounting and live cuMemPool stats.")
        .def("print_status", &CUDAMemoryArbiter::print_status,
             "Print formatted zone table and pool stats to stdout.");

    // ---- GPU preflight diagnostics ---------------------------------------
    py::class_<GpuDiagnosticReport>(m, "GpuDiagnosticReport")
        .def_readonly("device_name",        &GpuDiagnosticReport::device_name)
        .def_readonly("sm_count",           &GpuDiagnosticReport::sm_count)
        .def_readonly("compute_major",      &GpuDiagnosticReport::compute_major)
        .def_readonly("compute_minor",      &GpuDiagnosticReport::compute_minor)
        .def_readonly("total_vram_bytes",   &GpuDiagnosticReport::total_vram_bytes)
        .def_readonly("free_vram_bytes",    &GpuDiagnosticReport::free_vram_bytes)
        .def_readonly("ecc_supported",      &GpuDiagnosticReport::ecc_supported)
        .def_readonly("cuda_error_cleared", &GpuDiagnosticReport::cuda_error_cleared)
        .def_readonly("nvml_available",     &GpuDiagnosticReport::nvml_available)
        .def_readonly("temperature_c",      &GpuDiagnosticReport::temperature_c)
        .def_readonly("power_draw_w",       &GpuDiagnosticReport::power_draw_w)
        .def_readonly("power_limit_w",      &GpuDiagnosticReport::power_limit_w)
        .def_readonly("ecc_uncorrected",    &GpuDiagnosticReport::ecc_uncorrected)
        .def_readonly("ecc_corrected",      &GpuDiagnosticReport::ecc_corrected)
        .def_readonly("throttle_reasons",   &GpuDiagnosticReport::throttle_reasons)
        .def_readonly("healthy",            &GpuDiagnosticReport::healthy)
        .def_readonly("failure_reason",     &GpuDiagnosticReport::failure_reason);

    m.def("run_gpu_preflight",
          [](int device_idx) { return run_gpu_preflight(device_idx); },
          py::arg("device_idx") = 0,
          "Run GPU hardware health checks before training.\n"
          "Returns a GpuDiagnosticReport. Check .healthy before proceeding.");

    m.def("print_diagnostic_report",
          &print_diagnostic_report,
          py::arg("report"),
          "Print a formatted preflight report to stdout.");
}
