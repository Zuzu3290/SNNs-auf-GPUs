// C++ VRAM zone arbiter with direct cuMemPool API control.
// Partitions device VRAM into named zones; enforces hard caps via atomic counters.
// Pool release threshold is pinned to the emergency zone floor so the CUDA memory
// pool never drops below that amount when returning memory to the OS.
#pragma once
#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Zone indices — used as flat array keys for O(1) lookup.
enum SnnZone : int {
    SNN_ZONE_DATASET_CACHE    = 0,
    SNN_ZONE_MODEL_PARAMS     = 1,
    SNN_ZONE_KERNEL_WORKSPACE = 2,
    SNN_ZONE_EMERGENCY        = 3,
    SNN_ZONE_COUNT            = 4,
};

// Per-zone static configuration — stored in the .cu file to avoid
// C++14 out-of-class definition requirements for constexpr static members.
struct SnnZoneInfo {
    const char* name;
    double      ratio;          // fraction of total VRAM budget
    bool        hard_enforced;  // refuse request() above hard limit?
};

struct SnnArbiterStats {
    size_t used_bytes[SNN_ZONE_COUNT];
    size_t soft_limit_bytes[SNN_ZONE_COUNT];
    size_t hard_limit_bytes[SNN_ZONE_COUNT];
    // Live readings from the CUDA memory pool (zero if pool unavailable)
    size_t pool_used_bytes;
    size_t pool_reserved_bytes;
    size_t pool_used_high_bytes;
};

class CUDAMemoryArbiter {
public:
    explicit CUDAMemoryArbiter(int device_idx = 0);
    ~CUDAMemoryArbiter() = default;

    // --- Allocation tracking -------------------------------------------

    // Register bytes of VRAM usage for a zone.
    // Returns false (and rolls back) if a hard-enforced zone's hard limit
    // would be exceeded. Warns to stderr when the soft limit is crossed.
    bool request(SnnZone zone, size_t bytes);

    // Deregister bytes — clamped to zero if over-released.
    void release(SnnZone zone, size_t bytes);

    // --- Zone queries --------------------------------------------------

    size_t soft_limit(SnnZone zone) const { return soft_limit_[zone]; }
    size_t hard_limit(SnnZone zone) const { return hard_limit_[zone]; }
    size_t used      (SnnZone zone) const {
        return used_[zone].load(std::memory_order_relaxed);
    }
    bool   over_soft (SnnZone zone) const { return used(zone) > soft_limit(zone); }

    // --- cuMemPool controls -------------------------------------------

    // Set how many bytes the pool holds before returning to the OS.
    // Affects cudaMallocAsync / cudaFreeAsync behaviour globally on this device.
    void   set_release_threshold(size_t bytes);
    size_t get_release_threshold() const;

    // --- Diagnostics --------------------------------------------------

    SnnArbiterStats stats() const;
    void            print_status() const;

private:
    int           device_idx_;
    cudaMemPool_t pool_;
    bool          pool_available_;

    size_t              soft_limit_[SNN_ZONE_COUNT];
    size_t              hard_limit_[SNN_ZONE_COUNT];
    std::atomic<size_t> used_[SNN_ZONE_COUNT];
};
