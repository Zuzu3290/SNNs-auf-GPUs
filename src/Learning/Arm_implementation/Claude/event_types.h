// =============================================================================
// include/event_types.h  —  Shared event camera data structures
// =============================================================================
#ifndef EVENT_TYPES_H
#define EVENT_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Raw event (4 fields, 10 bytes unpadded — aligned to 12 in practice)
// ---------------------------------------------------------------------------
typedef struct __attribute__((packed)) {
    uint16_t x;         // pixel column
    uint16_t y;         // pixel row
    int64_t  t;         // timestamp in microseconds
    uint8_t  polarity;  // 1 = ON (brightness increase), 0 = OFF
} Event;

// ---------------------------------------------------------------------------
// Batch of raw events (contiguous SOA layout for SIMD friendliness)
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t* xs;        // shape (N,)
    uint16_t* ys;        // shape (N,)
    int64_t*  ts;        // shape (N,) — microseconds
    uint8_t*  polarity;  // shape (N,) — 0 or 1
    size_t    count;     // N
    uint16_t  sensor_w;  // sensor width
    uint16_t  sensor_h;  // sensor height
} EventBatch;

// ---------------------------------------------------------------------------
// Voxel grid (2, T, H, W) float32 — ON and OFF channels
// Stored in channel-first contiguous layout.
// ---------------------------------------------------------------------------
typedef struct {
    float*   data;       // total size = 2 * T * H * W floats
    int      T;          // temporal bins
    int      H;          // sensor height
    int      W;          // sensor width
    size_t   stride_t;   // H * W
    size_t   stride_h;   // W
} VoxelGrid;

// ---------------------------------------------------------------------------
// Sensor configuration
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t width;
    uint16_t height;
    int      sensitivity;       // 1–10
    int      noise_filter_us;   // Background Activity Filter window
    int      hot_pixel_thresh;  // events/s above which pixel is masked
} SensorConfig;

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------
typedef struct {
    SensorConfig sensor;
    int          n_bins;       // temporal bins per voxel
    int          window_us;    // accumulation window in µs
    int          n_classes;    // output classes
    float        lif_thresh;   // LIF spike threshold
    float        lif_leak;     // LIF decay factor
    float        lif_reset;    // LIF reset potential
    bool         use_neon;     // use ARM NEON kernels
    const char*  backend;      // "mock" | "pyaer" | "metavision"
} PipelineConfig;

// ---------------------------------------------------------------------------
// Batch allocator / free
// ---------------------------------------------------------------------------
static inline EventBatch* event_batch_alloc(size_t capacity,
                                             uint16_t W, uint16_t H) {
    EventBatch* b = (EventBatch*)calloc(1, sizeof(EventBatch));
    if (!b) return NULL;
    b->xs       = (uint16_t*)aligned_alloc(16, capacity * sizeof(uint16_t));
    b->ys       = (uint16_t*)aligned_alloc(16, capacity * sizeof(uint16_t));
    b->ts       = (int64_t* )aligned_alloc(16, capacity * sizeof(int64_t));
    b->polarity = (uint8_t* )aligned_alloc(16, capacity * sizeof(uint8_t));
    b->count    = 0;
    b->sensor_w = W;
    b->sensor_h = H;
    return b;
}

static inline void event_batch_free(EventBatch* b) {
    if (!b) return;
    free(b->xs);
    free(b->ys);
    free(b->ts);
    free(b->polarity);
    free(b);
}

// ---------------------------------------------------------------------------
// Voxel grid allocator / free
// ---------------------------------------------------------------------------
static inline VoxelGrid* voxel_grid_alloc(int T, int H, int W) {
    VoxelGrid* v = (VoxelGrid*)calloc(1, sizeof(VoxelGrid));
    if (!v) return NULL;
    v->data     = (float*)aligned_alloc(16, 2 * T * H * W * sizeof(float));
    v->T        = T;
    v->H        = H;
    v->W        = W;
    v->stride_t = (size_t)H * W;
    v->stride_h = (size_t)W;
    if (v->data) memset(v->data, 0, 2 * T * H * W * sizeof(float));
    return v;
}

static inline void voxel_grid_free(VoxelGrid* v) {
    if (!v) return;
    free(v->data);
    free(v);
}

// Accessor: voxel[pol][t][y][x]
static inline float* voxel_at(VoxelGrid* v, int pol, int t, int y, int x) {
    return &v->data[pol * v->T * v->stride_t +
                    t * v->stride_t +
                    y * v->stride_h + x];
}

#ifdef __cplusplus
}
#endif

#endif // EVENT_TYPES_H
