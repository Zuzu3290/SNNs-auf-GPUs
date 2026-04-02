// =============================================================================
// include/arm_kernels.h  —  C interface to ARM64 assembly kernels
// =============================================================================
#ifndef ARM_KERNELS_H
#define ARM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// event_threshold_neon
// Apply hard threshold: output[i] = (input[i] >= threshold) ? 1.0f : 0.0f
// input/output must be 16-byte aligned for best NEON performance.
// ---------------------------------------------------------------------------
void event_threshold_neon(
    const float* input,
    float*       output,
    int          count,
    float        threshold
);

// ---------------------------------------------------------------------------
// lif_membrane_neon
// Leaky-Integrate-and-Fire update for N pixels (SIMD over pixels).
//   V[i] = leak * V[i] + input[i]
//   spike[i] = (V[i] >= v_thresh) ? 1.0f : 0.0f
//   V[i] reset to v_reset on spike
// membrane[] is updated in-place (carries state between calls).
// ---------------------------------------------------------------------------
void lif_membrane_neon(
    const float* input,
    float*       membrane,
    float*       spikes,
    int          N,
    float        leak,
    float        v_thresh,
    float        v_reset
);

// ---------------------------------------------------------------------------
// voxel_accumulate
// Scatter events into a (T, H, W) float32 voxel grid.
// voxel[t_bins[i] * HW + ys[i] * W + xs[i]] += weights[i]
// ---------------------------------------------------------------------------
void voxel_accumulate(
    const uint16_t* xs,
    const uint16_t* ys,
    const uint8_t*  t_bins,
    const float*    weights,
    float*          voxel,
    int             N,
    int             W,
    int             HW
);

// ---------------------------------------------------------------------------
// polarity_split
// Partition events into ON (polarity==1) and OFF (polarity==0) arrays.
// Returns counts via on_count and off_count pointers.
// ---------------------------------------------------------------------------
void polarity_split(
    const float*   input,
    const uint8_t* polarities,
    float*         on_out,
    float*         off_out,
    int            N,
    int*           on_count,
    int*           off_count
);

// ---------------------------------------------------------------------------
// event_count_xy
// Build a 2D event-count histogram. hist[y*W + x] += 1 per event.
// hist[] must be zeroed before calling.
// ---------------------------------------------------------------------------
void event_count_xy(
    const uint16_t* xs,
    const uint16_t* ys,
    uint32_t*       hist,
    int             N,
    int             W
);

// ---------------------------------------------------------------------------
// dot_product_f32_neon
// Fused dot product: returns sum(a[i] * b[i]) for i in [0, N).
// Used by the FC readout layer.
// ---------------------------------------------------------------------------
float dot_product_f32_neon(
    const float* a,
    const float* b,
    int          N
);

// ---------------------------------------------------------------------------
// relu_inplace_neon
// In-place ReLU: x[i] = max(x[i], 0.0f)
// ---------------------------------------------------------------------------
void relu_inplace_neon(float* x, int N);

#ifdef __cplusplus
}
#endif

#endif // ARM_KERNELS_H
