// Shared C++/CUDA configuration struct bridging accel_config.yaml (Python side) and engine.cu (C++ dispatch side).
// Defines KernelConfig with four boolean flags: accelerate, optimize_throughput, optimize_memory, profile_energy.
// Provides KERNEL_CONFIG_DEFAULT (all false) and KERNEL_CONFIG_ACCELERATED (all true) constexpr presets.
#pragma once

// ---------------------------------------------------------------------------
// KernelConfig — single source of truth for execution-path selection.
//
// Python reads accel_config.yaml and passes these flags down to C++.
// C++ engine.cu reads this struct and dispatches to the appropriate path.
// ---------------------------------------------------------------------------

struct KernelConfig {
    bool accelerate;           // master switch: enable full acceleration stack
    bool optimize_throughput;  // auto-tune block/grid via occupancy API
    bool optimize_memory;      // audit GPU memory headroom before launch
    bool profile_energy;       // wrap kernel with NVML energy profiler
};

// Standard mode — no acceleration overhead
constexpr KernelConfig KERNEL_CONFIG_DEFAULT = {
    /*accelerate*/          false,
    /*optimize_throughput*/ false,
    /*optimize_memory*/     false,
    /*profile_energy*/      false
};

// Full acceleration — all three GPU attributes active
constexpr KernelConfig KERNEL_CONFIG_ACCELERATED = {
    /*accelerate*/          true,
    /*optimize_throughput*/ true,
    /*optimize_memory*/     true,
    /*profile_energy*/      true
};
