// =============================================================================
// src/kernel/spike_kernel.c  —  Event → Spike Kernel (C + ARM NEON)
// =============================================================================
// This module bridges the ARM assembly kernels with the Python layer.
// When NEON is available it calls the .s routines; otherwise falls back
// to portable C scalar code that compiles on any architecture.
//
// Build:
//   gcc -O2 -march=armv8-a+simd -I include -c \
//       src/kernel/spike_kernel.c -o build/obj/kernel/spike_kernel.o
// =============================================================================

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "event_types.h"
#include "arm_kernels.h"

// ---------------------------------------------------------------------------
// CPU feature detection
// ---------------------------------------------------------------------------
static bool has_neon(void) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#elif defined(__linux__)
    // Runtime detection via /proc/cpuinfo
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return false;
    char line[256];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "neon") || strstr(line, "asimd")) {
            found = true;
            break;
        }
    }
    fclose(f);
    return found;
#else
    return false;
#endif
}


// ---------------------------------------------------------------------------
// Scalar fallback: threshold
// ---------------------------------------------------------------------------
static void threshold_scalar(const float* in, float* out,
                              int N, float thresh) {
    for (int i = 0; i < N; i++)
        out[i] = (in[i] >= thresh) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Scalar fallback: LIF
// ---------------------------------------------------------------------------
static void lif_scalar(const float* input, float* membrane,
                       float* spikes, int N,
                       float leak, float v_thresh, float v_reset) {
    for (int i = 0; i < N; i++) {
        membrane[i] = leak * membrane[i] + input[i];
        if (membrane[i] >= v_thresh) {
            spikes[i]   = 1.0f;
            membrane[i] = v_reset;
        } else {
            spikes[i] = 0.0f;
        }
    }
}


// ---------------------------------------------------------------------------
// Public API: spike_kernel_run
// ---------------------------------------------------------------------------
// Applies LIF integration across a full voxel grid timestep.
// Dispatches to NEON or scalar based on runtime CPU detection.
//
// Arguments:
//   voxel    — (2, T, H, W) input voxel grid
//   membrane — (2, H, W)    membrane state (in/out, preserved between calls)
//   spikes   — (2, T, H, W) output spike array (pre-zeroed)
//   T, H, W  — dimensions
//   leak, v_thresh, v_reset — LIF parameters
//   use_neon — 0 = scalar, 1 = auto-detect NEON
// ---------------------------------------------------------------------------
void spike_kernel_run(
    const float* voxel,
    float*       membrane,
    float*       spikes,
    int          T,
    int          H,
    int          W,
    float        leak,
    float        v_thresh,
    float        v_reset,
    int          use_neon
) {
    bool neon = use_neon && has_neon();
    int  HW   = H * W;

    for (int pol = 0; pol < 2; pol++) {
        float* mem_p = membrane + pol * HW;     // (H,W) membrane slice

        for (int t = 0; t < T; t++) {
            const float* in_p  = voxel  + pol * T * HW + t * HW;
            float*       spk_p = spikes + pol * T * HW + t * HW;

            if (neon) {
                lif_membrane_neon(in_p, mem_p, spk_p, HW,
                                  leak, v_thresh, v_reset);
            } else {
                lif_scalar(in_p, mem_p, spk_p, HW,
                           leak, v_thresh, v_reset);
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Public API: threshold_kernel_run
// ---------------------------------------------------------------------------
void threshold_kernel_run(
    const float* voxel,
    float*       spikes,
    int          count,
    float        threshold,
    int          use_neon
) {
    bool neon = use_neon && has_neon();
    if (neon) {
        event_threshold_neon(voxel, spikes, count, threshold);
    } else {
        threshold_scalar(voxel, spikes, count, threshold);
    }
}


// ---------------------------------------------------------------------------
// Public API: voxel_build
// ---------------------------------------------------------------------------
// Convert an EventBatch into a VoxelGrid using bilinear temporal interpolation.
// Calls voxel_accumulate (ARM asm) for the scatter-add step.
//
// Interpolation: each event gets split across two adjacent time bins
// proportional to its normalised position within the window.
// ---------------------------------------------------------------------------
void voxel_build(
    const EventBatch* batch,
    VoxelGrid*        grid,
    int64_t           window_start_us,
    int64_t           window_us
) {
    if (!batch || !grid || batch->count == 0) return;

    int T  = grid->T;
    int H  = grid->H;
    int W  = grid->W;
    int HW = H * W;

    // Temporary arrays for scatter-add calls (one polarity at a time)
    uint16_t* px    = (uint16_t*)malloc(batch->count * sizeof(uint16_t));
    uint16_t* py    = (uint16_t*)malloc(batch->count * sizeof(uint16_t));
    uint8_t*  tbins = (uint8_t* )malloc(batch->count * sizeof(uint8_t));
    float*    wlow  = (float*   )malloc(batch->count * sizeof(float));
    float*    whigh = (float*   )malloc(batch->count * sizeof(float));
    uint8_t*  thigh = (uint8_t* )malloc(batch->count * sizeof(uint8_t));

    if (!px || !py || !tbins || !wlow || !whigh || !thigh) {
        fprintf(stderr, "[voxel_build] allocation failure\n");
        goto cleanup;
    }

    for (int pol = 0; pol < 2; pol++) {
        int n_pol = 0;

        for (size_t i = 0; i < batch->count; i++) {
            if ((int)batch->polarity[i] != (pol == 0 ? 1 : 0)) continue;

            // Normalised time in [0, T-1]
            double t_norm = (double)(batch->ts[i] - window_start_us)
                          / (double)window_us * (T - 1);
            if (t_norm < 0) t_norm = 0;
            if (t_norm > T - 1) t_norm = T - 1;

            int   tl = (int)t_norm;
            int   th = tl + 1;
            float wh = (float)(t_norm - tl);
            float wl = 1.0f - wh;

            px[n_pol]    = batch->xs[i];
            py[n_pol]    = batch->ys[i];
            tbins[n_pol] = (uint8_t)tl;
            wlow[n_pol]  = wl;
            whigh[n_pol] = wh;
            thigh[n_pol] = (uint8_t)(th < T ? th : tl);
            n_pol++;
        }

        if (n_pol == 0) continue;

        float* voxel_pol = grid->data + pol * T * HW;

        // Scatter low-bin contributions
        voxel_accumulate(px, py, tbins, wlow,
                         voxel_pol, n_pol, W, HW);

        // Scatter high-bin contributions
        voxel_accumulate(px, py, thigh, whigh,
                         voxel_pol, n_pol, W, HW);
    }

cleanup:
    free(px); free(py); free(tbins);
    free(wlow); free(whigh); free(thigh);
}


// ---------------------------------------------------------------------------
// Self-test (runs when compiled standalone)
// ---------------------------------------------------------------------------
#ifdef SPIKE_KERNEL_MAIN

#include <time.h>

static double clock_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    printf("=== spike_kernel self-test ===\n");
    printf("NEON available: %s\n\n", has_neon() ? "yes" : "no");

    int N = 346 * 260;   // one frame of pixels
    int T = 5;

    float* voxel    = aligned_alloc(16, 2 * T * N * sizeof(float));
    float* membrane = aligned_alloc(16, 2 * N * sizeof(float));
    float* spikes   = aligned_alloc(16, 2 * T * N * sizeof(float));

    // Fill with random data
    srand(42);
    for (int i = 0; i < 2 * T * N; i++)
        voxel[i] = (float)rand() / RAND_MAX;
    memset(membrane, 0, 2 * N * sizeof(float));
    memset(spikes,   0, 2 * T * N * sizeof(float));

    // Benchmark scalar vs NEON
    double t0, t1;

    t0 = clock_ms();
    spike_kernel_run(voxel, membrane, spikes,
                     T, 260, 346, 0.9f, 1.0f, 0.0f, 0);
    t1 = clock_ms();
    printf("Scalar LIF:  %.3f ms\n", t1 - t0);

    memset(membrane, 0, 2 * N * sizeof(float));
    memset(spikes,   0, 2 * T * N * sizeof(float));

    t0 = clock_ms();
    spike_kernel_run(voxel, membrane, spikes,
                     T, 260, 346, 0.9f, 1.0f, 0.0f, 1);
    t1 = clock_ms();
    printf("NEON  LIF:   %.3f ms\n", t1 - t0);

    // Count spikes
    int fired = 0;
    for (int i = 0; i < 2 * T * N; i++)
        if (spikes[i] > 0.5f) fired++;
    printf("Spike rate:  %.1f%%\n", 100.0f * fired / (2 * T * N));

    // Dot product benchmark
    float* a = aligned_alloc(16, N * sizeof(float));
    float* b = aligned_alloc(16, N * sizeof(float));
    for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    t0  = clock_ms();
    float dp = dot_product_f32_neon(a, b, N);
    t1  = clock_ms();
    printf("\ndot_product(1, 2, N=%d) = %.1f  (expect %.1f)  %.3f ms\n",
           N, dp, 2.0f * N, t1 - t0);

    free(voxel); free(membrane); free(spikes); free(a); free(b);
    printf("\nAll tests passed.\n");
    return 0;
}
#endif // SPIKE_KERNEL_MAIN
