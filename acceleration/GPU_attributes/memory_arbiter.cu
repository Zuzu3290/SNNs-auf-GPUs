// cuMemPool-backed VRAM zone arbiter.
// Emergency zone soft limit is set as the pool release threshold — the CUDA
// memory pool will never drop below that amount when returning pages to the OS.
// Zone accounting uses lock-free atomics; hard limit checks do a CAS rollback.
#include "memory_arbiter.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Zone definitions — ratio × budget gives soft limit; hard = 85 % of soft.
// dataset_cache is soft-only: cache eviction is tolerable.
// The other three are hard-enforced: kernel workspace and model params must
// not silently bleed into each other.
// ---------------------------------------------------------------------------
static const SnnZoneInfo kZoneInfo[SNN_ZONE_COUNT] = {
    { "dataset_cache",    0.40, false },
    { "model_params",     0.30, true  },
    { "kernel_workspace", 0.20, true  },
    { "emergency",        0.10, true  },
};

// Reserve 512 MB for the PyTorch caching allocator's internal overhead.
static constexpr size_t kAllocatorOverhead = 512ULL * 1024 * 1024;
// Hard limit = 85 % of soft — leaves headroom before the CUDA OOM killer fires.
static constexpr double kHardFraction = 0.85;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CUDAMemoryArbiter::CUDAMemoryArbiter(int device_idx)
    : device_idx_(device_idx), pool_(nullptr), pool_available_(false)
{
    for (int i = 0; i < SNN_ZONE_COUNT; ++i)
        used_[i].store(0, std::memory_order_relaxed);

    cudaSetDevice(device_idx_);

    // One-time VRAM query — compute zone limits from total VRAM.
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    size_t budget = (total_bytes > kAllocatorOverhead)
                    ? total_bytes - kAllocatorOverhead
                    : total_bytes;

    for (int i = 0; i < SNN_ZONE_COUNT; ++i) {
        soft_limit_[i] = static_cast<size_t>(budget * kZoneInfo[i].ratio);
        hard_limit_[i] = static_cast<size_t>(soft_limit_[i] * kHardFraction);
    }

    // Acquire the default CUDA memory pool (requires CUDA 11.2+).
    cudaError_t err = cudaDeviceGetDefaultMemPool(&pool_, device_idx_);
    if (err == cudaSuccess && pool_ != nullptr) {
        pool_available_ = true;

        // Pin emergency zone as the pool release threshold.
        // cudaMallocAsync / cudaFreeAsync will retain at least this many
        // bytes in the pool instead of returning pages to the OS —
        // guaranteeing fast re-allocation for emergency workspace requests.
        size_t threshold = soft_limit_[SNN_ZONE_EMERGENCY];
        cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &threshold);

        printf("[MemoryArbiter] device=%d  budget=%.0f MB  "
               "pool release threshold=%.0f MB (emergency zone)\n",
               device_idx_,
               budget       / (1024.0 * 1024.0),
               threshold    / (1024.0 * 1024.0));
    } else {
        fprintf(stderr,
                "[MemoryArbiter] device=%d  cuMemPool unavailable (%s) "
                "— zone accounting only\n",
                device_idx_, cudaGetErrorString(err));
    }

    printf("[MemoryArbiter] zones initialised:\n");
    for (int i = 0; i < SNN_ZONE_COUNT; ++i) {
        printf("  %-20s  soft=%.0f MB  hard=%.0f MB  enforced=%s\n",
               kZoneInfo[i].name,
               soft_limit_[i] / (1024.0 * 1024.0),
               hard_limit_[i] / (1024.0 * 1024.0),
               kZoneInfo[i].hard_enforced ? "yes" : "no");
    }
}

// ---------------------------------------------------------------------------
// Allocation tracking
// ---------------------------------------------------------------------------
bool CUDAMemoryArbiter::request(SnnZone zone, size_t bytes) {
    // Optimistic add — then validate. Roll back if over the hard cap.
    size_t prev = used_[zone].fetch_add(bytes, std::memory_order_acq_rel);
    size_t next = prev + bytes;

    if (kZoneInfo[zone].hard_enforced && next > hard_limit_[zone]) {
        used_[zone].fetch_sub(bytes, std::memory_order_acq_rel);
        fprintf(stderr,
                "[MemoryArbiter] REFUSED  zone=%-20s  "
                "requested=%.0f MB  used=%.0f MB  hard=%.0f MB\n",
                kZoneInfo[zone].name,
                bytes / (1024.0 * 1024.0),
                prev  / (1024.0 * 1024.0),
                hard_limit_[zone] / (1024.0 * 1024.0));
        return false;
    }

    if (next > soft_limit_[zone]) {
        fprintf(stderr,
                "[MemoryArbiter] SOFT EXCEEDED  zone=%-20s  "
                "%.0f MB > %.0f MB soft limit\n",
                kZoneInfo[zone].name,
                next         / (1024.0 * 1024.0),
                soft_limit_[zone] / (1024.0 * 1024.0));
    }

    return true;
}

void CUDAMemoryArbiter::release(SnnZone zone, size_t bytes) {
    // Clamp to zero — release() is always safe to over-call.
    size_t cur = used_[zone].load(std::memory_order_relaxed);
    size_t sub = (bytes > cur) ? cur : bytes;
    used_[zone].fetch_sub(sub, std::memory_order_acq_rel);
}

// ---------------------------------------------------------------------------
// cuMemPool controls
// ---------------------------------------------------------------------------
void CUDAMemoryArbiter::set_release_threshold(size_t bytes) {
    if (!pool_available_) return;
    cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &bytes);
}

size_t CUDAMemoryArbiter::get_release_threshold() const {
    if (!pool_available_) return 0;
    size_t threshold = 0;
    cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &threshold);
    return threshold;
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
SnnArbiterStats CUDAMemoryArbiter::stats() const {
    SnnArbiterStats s{};
    for (int i = 0; i < SNN_ZONE_COUNT; ++i) {
        s.used_bytes[i]       = used_[i].load(std::memory_order_relaxed);
        s.soft_limit_bytes[i] = soft_limit_[i];
        s.hard_limit_bytes[i] = hard_limit_[i];
    }
    if (pool_available_) {
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent,
                                &s.pool_used_bytes);
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent,
                                &s.pool_reserved_bytes);
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemHigh,
                                &s.pool_used_high_bytes);
    }
    return s;
}

void CUDAMemoryArbiter::print_status() const {
    auto s = stats();
    printf("[MemoryArbiter] VRAM zones (device:%d)\n", device_idx_);
    for (int i = 0; i < SNN_ZONE_COUNT; ++i) {
        double used_mb = s.used_bytes[i]       / (1024.0 * 1024.0);
        double soft_mb = s.soft_limit_bytes[i] / (1024.0 * 1024.0);
        double hard_mb = s.hard_limit_bytes[i] / (1024.0 * 1024.0);
        double pct     = soft_mb > 0.0 ? used_mb / soft_mb * 100.0 : 0.0;
        int    bar_w   = (int)(pct / 5.0);
        if (bar_w > 20) bar_w = 20;
        char bar[21];
        for (int b = 0; b < 20; ++b) bar[b] = (b < bar_w) ? '#' : '.';
        bar[20] = '\0';
        printf("  %-20s  %6.0f / %6.0f MB  hard=%6.0f MB  %5.1f%%  [%s]\n",
               kZoneInfo[i].name, used_mb, soft_mb, hard_mb, pct, bar);
    }
    if (pool_available_) {
        printf("  --- pool (live) ---\n");
        printf("  used:      %.0f MB\n", s.pool_used_bytes      / (1024.0*1024.0));
        printf("  reserved:  %.0f MB\n", s.pool_reserved_bytes  / (1024.0*1024.0));
        printf("  high mark: %.0f MB\n", s.pool_used_high_bytes / (1024.0*1024.0));
    }
}
