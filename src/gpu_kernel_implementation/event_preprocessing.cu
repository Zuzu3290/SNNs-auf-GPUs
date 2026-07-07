// GPU-side event preprocessing — replaces the CPU tonic pipeline for the GPU-only config.
//
// Two kernels:
//   event_denoise_kernel  — mirrors tonic.transforms.Denoise(filter_time)
//                           Marks isolated events (no neighbour within filter_time us
//                           in a spatial +-1 window) for removal.
//   event_to_frame_kernel — mirrors tonic.transforms.ToFrame
//                           Scatters events into a [C, T, H, W] float32 frame tensor.
//
// C++ entry points:
//   torch::Tensor gpu_denoise_events(ev_x, ev_y, ev_t, filter_time_us, search_window)
//   torch::Tensor gpu_events_to_frame(ev_x, ev_y, ev_t, ev_p, keep_mask,
//                                     H, W, n_time_bins, t_start_us, t_end_us)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Denoise kernel
//
// For each event i, scan events in [i-search_window, i+search_window] that
// share an adjacent pixel (|dx|<=1, |dy|<=1). If any such neighbour exists
// within filter_time_us, the event is kept (keep_mask[i]=1), else discarded.
// Events must be sorted by timestamp ascending (tonic guarantees this).
// ---------------------------------------------------------------------------
__global__ void event_denoise_kernel(
    const int16_t* __restrict__ ev_x,
    const int16_t* __restrict__ ev_y,
    const int64_t* __restrict__ ev_t,
    int8_t*        __restrict__ keep_mask,
    int64_t n_events,
    int64_t filter_time_us,
    int     search_window
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_events) return;

    const int64_t t_i = ev_t[i];
    const int16_t x_i = ev_x[i];
    const int16_t y_i = ev_y[i];
    bool found = false;

    for (int64_t j = i - 1; j >= 0 && j >= i - search_window && !found; --j) {
        if (t_i - ev_t[j] > filter_time_us) break;
        int dx = (int)ev_x[j] - (int)x_i;
        int dy = (int)ev_y[j] - (int)y_i;
        if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) found = true;
    }

    for (int64_t j = i + 1; j < n_events && j <= i + search_window && !found; ++j) {
        if (ev_t[j] - t_i > filter_time_us) break;
        int dx = (int)ev_x[j] - (int)x_i;
        int dy = (int)ev_y[j] - (int)y_i;
        if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) found = true;
    }

    keep_mask[i] = found ? 1 : 0;
}

// ---------------------------------------------------------------------------
// ToFrame kernel
//
// Each surviving event is scattered into frame[polarity][time_bin][y][x].
// Polarity 0 = negative, 1 = positive. Accumulates event counts (float32).
// ---------------------------------------------------------------------------
__global__ void event_to_frame_kernel(
    const int16_t* __restrict__ ev_x,
    const int16_t* __restrict__ ev_y,
    const int64_t* __restrict__ ev_t,
    const int8_t*  __restrict__ ev_p,
    const int8_t*  __restrict__ keep_mask,
    float*         __restrict__ frame,    // [C=2, T, H, W]
    int64_t n_events,
    int64_t H, int64_t W,
    int64_t n_time_bins,
    int64_t t_start_us,
    int64_t t_end_us
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_events || !keep_mask[i]) return;

    const int64_t duration = t_end_us - t_start_us;
    if (duration <= 0) return;

    const int64_t t_rel = ev_t[i] - t_start_us;
    if (t_rel < 0) return;

    int64_t bin = t_rel * n_time_bins / duration;
    if (bin >= n_time_bins) bin = n_time_bins - 1;

    const int16_t x = ev_x[i];
    const int16_t y = ev_y[i];
    if (x < 0 || x >= (int16_t)W || y < 0 || y >= (int16_t)H) return;

    const int64_t c   = (ev_p[i] > 0) ? 1 : 0;
    const int64_t idx = c * (n_time_bins * H * W)
                      + bin * (H * W)
                      + (int64_t)y * W
                      + (int64_t)x;

    atomicAdd(&frame[idx], 1.0f);
}

// ---------------------------------------------------------------------------
// Entry point 1 — denoise
// Returns keep_mask [n_events] int8 on CUDA.
// ---------------------------------------------------------------------------
torch::Tensor gpu_denoise_events(
    torch::Tensor ev_x,           // [N] int16 CUDA
    torch::Tensor ev_y,           // [N] int16 CUDA
    torch::Tensor ev_t,           // [N] int64 CUDA
    int64_t filter_time_us = 10000,
    int     search_window  = 200
) {
    TORCH_CHECK(ev_x.is_cuda(), "events must be on CUDA");
    const int64_t n    = ev_x.size(0);
    auto keep_mask     = torch::zeros({n}, ev_x.options().dtype(torch::kInt8));
    if (n == 0) return keep_mask;

    const int block    = 256;
    const int grid     = static_cast<int>((n + block - 1) / block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    event_denoise_kernel<<<grid, block, 0, stream>>>(
        ev_x.data_ptr<int16_t>(), ev_y.data_ptr<int16_t>(),
        ev_t.data_ptr<int64_t>(), keep_mask.data_ptr<int8_t>(),
        n, filter_time_us, search_window
    );
    return keep_mask;
}

// ---------------------------------------------------------------------------
// Entry point 2 — events to frame
// Returns frame [2, n_time_bins, H, W] float32 on CUDA.
// ---------------------------------------------------------------------------
torch::Tensor gpu_events_to_frame(
    torch::Tensor ev_x,
    torch::Tensor ev_y,
    torch::Tensor ev_t,
    torch::Tensor ev_p,
    torch::Tensor keep_mask,
    int64_t H, int64_t W,
    int64_t n_time_bins,
    int64_t t_start_us,
    int64_t t_end_us
) {
    TORCH_CHECK(ev_x.is_cuda(), "events must be on CUDA");
    const int64_t n = ev_x.size(0);
    auto frame      = torch::zeros({2, n_time_bins, H, W},
                                    ev_x.options().dtype(torch::kFloat32));
    if (n == 0) return frame;

    const int block     = 256;
    const int grid      = static_cast<int>((n + block - 1) / block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    event_to_frame_kernel<<<grid, block, 0, stream>>>(
        ev_x.data_ptr<int16_t>(), ev_y.data_ptr<int16_t>(),
        ev_t.data_ptr<int64_t>(), ev_p.data_ptr<int8_t>(),
        keep_mask.data_ptr<int8_t>(), frame.data_ptr<float>(),
        n, H, W, n_time_bins, t_start_us, t_end_us
    );
    return frame;
}
