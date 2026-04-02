// =============================================================================
// src/pipeline.c  —  Native C/ARM Pipeline Entry Point
// =============================================================================
// This is the C-language equivalent of pipeline.py.
// It runs the full event camera → SNN pipeline without Python,
// using the ARM assembly kernels directly.
//
// Build:
//   make run-arm
//   # or manually:
//   gcc -O2 -march=armv8-a+simd -I include \
//       src/pipeline.c src/kernel/spike_kernel.c \
//       build/obj/asm_event_kernel.o \
//       -lpthread -lm -o build/bin/pipeline_c
//
// Usage:
//   ./build/bin/pipeline_c --backend mock --frames 20
//   ./build/bin/pipeline_c --backend pyaer --width 346 --height 260
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <pthread.h>

#include "event_types.h"
#include "arm_kernels.h"

// ---------------------------------------------------------------------------
// Forward declarations (implemented in other C modules)
// ---------------------------------------------------------------------------
extern void spike_kernel_run(const float*, float*, float*,
                              int, int, int,
                              float, float, float, int);
extern void voxel_build(const EventBatch*, VoxelGrid*,
                        int64_t, int64_t);


// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}


// ---------------------------------------------------------------------------
// Mock event generator
// ---------------------------------------------------------------------------
typedef struct {
    uint16_t sensor_w;
    uint16_t sensor_h;
    int      event_rate;     // events per second
    int64_t  t_current_us;  // current simulated time
    double   phase;
} MockCamera;

static EventBatch* mock_generate(MockCamera* cam, int64_t window_us) {
    int n = (int)((double)cam->event_rate * window_us / 1e6);
    if (n <= 0) return NULL;

    EventBatch* b = event_batch_alloc((size_t)n, cam->sensor_w, cam->sensor_h);
    if (!b) return NULL;

    // Moving edge: sinusoidal x position
    cam->phase += 2.0 * M_PI * window_us / 2e6;   // full cycle = 2s
    int edge_x = (int)(cam->sensor_w / 2 +
                       cam->sensor_w * 0.4 * sin(cam->phase));

    for (int i = 0; i < n; i++) {
        // Gaussian cluster around edge
        double gx = edge_x + ((double)rand() / RAND_MAX - 0.5) * 16.0;
        b->xs[i] = (uint16_t)fmax(0, fmin(cam->sensor_w - 1, gx));
        b->ys[i] = (uint16_t)(rand() % cam->sensor_h);
        b->ts[i] = cam->t_current_us +
                   (int64_t)((double)rand() / RAND_MAX * window_us);
        b->polarity[i] = (gx - edge_x > 0) ? 1 : 0;
    }
    b->count        = (size_t)n;
    b->sensor_w     = cam->sensor_w;
    b->sensor_h     = cam->sensor_h;
    cam->t_current_us += window_us;

    return b;
}


// ---------------------------------------------------------------------------
// Minimal SNN readout (linear classifier over spike rate)
// ---------------------------------------------------------------------------
typedef struct {
    float* weights;    // (n_classes, HW) fully-connected weights
    float* bias;       // (n_classes,)
    int    n_classes;
    int    input_size;
} LinearReadout;

static LinearReadout* readout_alloc(int n_classes, int input_size) {
    LinearReadout* r = calloc(1, sizeof(LinearReadout));
    r->weights    = calloc((size_t)(n_classes * input_size), sizeof(float));
    r->bias       = calloc((size_t)n_classes, sizeof(float));
    r->n_classes  = n_classes;
    r->input_size = input_size;
    // Random init: Xavier uniform
    float scale = sqrtf(6.0f / (input_size + n_classes));
    for (int i = 0; i < n_classes * input_size; i++)
        r->weights[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    return r;
}

static int readout_predict(LinearReadout* r, const float* spike_rates) {
    int   best_cls   = 0;
    float best_score = -1e9f;

    for (int c = 0; c < r->n_classes; c++) {
        const float* w = r->weights + c * r->input_size;
        // Use NEON dot product from ARM assembly
        float score = dot_product_f32_neon(w, spike_rates, r->input_size)
                    + r->bias[c];
        if (score > best_score) {
            best_score = score;
            best_cls   = c;
        }
    }
    return best_cls;
}

static void readout_free(LinearReadout* r) {
    if (!r) return;
    free(r->weights);
    free(r->bias);
    free(r);
}


// ---------------------------------------------------------------------------
// Pipeline statistics
// ---------------------------------------------------------------------------
typedef struct {
    double   total_ms;
    double   kernel_ms;
    double   readout_ms;
    long     n_frames;
    long     n_events;
} PipelineStats;

static void stats_print(const PipelineStats* s) {
    printf("\n=== Pipeline Stats ===\n");
    printf("  Frames:       %ld\n",        s->n_frames);
    printf("  Events:       %ld\n",        s->n_events);
    printf("  Wall time:    %.2f ms\n",    s->total_ms);
    printf("  Kernel time:  %.2f ms\n",    s->kernel_ms);
    printf("  Readout time: %.2f ms\n",    s->readout_ms);
    if (s->n_frames > 0) {
        double fps = s->n_frames / (s->total_ms / 1000.0);
        double lat = s->total_ms / s->n_frames;
        printf("  Throughput:   %.1f FPS\n",    fps);
        printf("  Avg latency:  %.2f ms/frame\n", lat);
    }
}


// ---------------------------------------------------------------------------
// Main pipeline loop
// ---------------------------------------------------------------------------
static int run_pipeline(PipelineConfig* cfg, int n_frames) {
    printf("\n=== Event Camera SNN Pipeline (C/ARM) ===\n\n");
    printf("  Sensor:     %d×%d\n",    cfg->sensor.width, cfg->sensor.height);
    printf("  Backend:    %s\n",       cfg->backend);
    printf("  Window:     %d µs\n",    cfg->window_us);
    printf("  Bins:       %d\n",       cfg->n_bins);
    printf("  Classes:    %d\n",       cfg->n_classes);
    printf("  NEON:       %s\n\n",     cfg->use_neon ? "yes" : "no");

    int W  = cfg->sensor.width;
    int H  = cfg->sensor.height;
    int T  = cfg->n_bins;
    int HW = H * W;

    // Allocate voxel grid and membrane state
    VoxelGrid* voxel = voxel_grid_alloc(T, H, W);
    float* membrane  = calloc((size_t)(2 * HW), sizeof(float));
    float* spikes    = calloc((size_t)(2 * T * HW), sizeof(float));
    float* rates     = calloc((size_t)(2 * HW), sizeof(float));

    if (!voxel || !membrane || !spikes || !rates) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    // Linear readout
    LinearReadout* readout = readout_alloc(cfg->n_classes, 2 * T * HW);

    // Mock camera
    MockCamera cam = {
        .sensor_w    = (uint16_t)W,
        .sensor_h    = (uint16_t)H,
        .event_rate  = 500000,
        .t_current_us = 0,
        .phase       = 0.0,
    };

    const char* class_names[] = {
        "background","car","pedestrian","cyclist",
        "truck","bus","sign","light","object","other"
    };

    PipelineStats stats = {0};
    double t_wall_start = now_ms();

    printf("%-6s  %-8s  %-14s  %-8s  %-8s\n",
           "Frame", "Events", "Class", "Rate%", "ms");
    printf("%-6s  %-8s  %-14s  %-8s  %-8s\n",
           "------","--------","----------","------","------");

    for (int frame = 0; frame < n_frames; frame++) {
        double t0 = now_ms();

        // ---- Module 1+2: generate / receive events ----
        EventBatch* batch = mock_generate(&cam, cfg->window_us);
        if (!batch) continue;
        stats.n_events += (long)batch->count;

        // ---- Module 3: build voxel grid ----
        memset(voxel->data, 0, (size_t)(2 * T * HW) * sizeof(float));
        voxel_build(batch, voxel, cam.t_current_us - cfg->window_us,
                    cfg->window_us);

        // ---- Module 5: ARM NEON spike kernel ----
        double t_kernel = now_ms();
        memset(spikes, 0, (size_t)(2 * T * HW) * sizeof(float));
        spike_kernel_run(voxel->data, membrane, spikes,
                         T, H, W,
                         cfg->lif_leak, cfg->lif_thresh, cfg->lif_reset,
                         cfg->use_neon);
        stats.kernel_ms += now_ms() - t_kernel;

        // ---- Module 6 (lite): compute spike rates per pixel ----
        for (int i = 0; i < 2 * HW; i++) {
            float sum = 0;
            for (int t = 0; t < T; t++)
                sum += spikes[i / HW * T * HW + t * HW + i % HW];
            rates[i] = sum / T;
        }
        relu_inplace_neon(rates, 2 * HW);

        // ---- Module 7: classify ----
        double t_readout = now_ms();
        int pred = readout_predict(readout, rates);
        stats.readout_ms += now_ms() - t_readout;

        double frame_ms = now_ms() - t0;
        stats.n_frames++;

        const char* name = (pred < cfg->n_classes) ? class_names[pred] : "?";
        float spike_pct  = 0;
        for (int i = 0; i < 2 * T * HW; i++)
            spike_pct += spikes[i];
        spike_pct = 100.0f * spike_pct / (2 * T * HW);

        printf("%-6d  %-8zu  %-14s  %-7.1f  %-8.2f\n",
               frame + 1, batch->count, name, spike_pct, frame_ms);

        event_batch_free(batch);
    }

    stats.total_ms = now_ms() - t_wall_start;
    stats_print(&stats);

    // Cleanup
    voxel_grid_free(voxel);
    readout_free(readout);
    free(membrane);
    free(spikes);
    free(rates);

    return 0;
}


// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("  --backend  mock|pyaer|metavision  (default: mock)\n");
    printf("  --width    sensor width            (default: 346)\n");
    printf("  --height   sensor height           (default: 260)\n");
    printf("  --bins     temporal bins           (default: 5)\n");
    printf("  --window   window µs               (default: 10000)\n");
    printf("  --classes  output classes          (default: 10)\n");
    printf("  --frames   frames to process       (default: 20)\n");
    printf("  --no-neon  disable NEON kernels\n");
    printf("  --help     show this message\n\n");
}

int main(int argc, char* argv[]) {
    PipelineConfig cfg = {
        .sensor = {
            .width           = 346,
            .height          = 260,
            .sensitivity     = 5,
            .noise_filter_us = 1500,
            .hot_pixel_thresh = 0,
        },
        .n_bins    = 5,
        .window_us = 10000,
        .n_classes = 10,
        .lif_thresh = 1.0f,
        .lif_leak   = 0.9f,
        .lif_reset  = 0.0f,
        .use_neon   = true,
        .backend    = "mock",
    };
    int n_frames = 20;

    static struct option long_opts[] = {
        {"backend",  required_argument, 0, 'b'},
        {"width",    required_argument, 0, 'W'},
        {"height",   required_argument, 0, 'H'},
        {"bins",     required_argument, 0, 't'},
        {"window",   required_argument, 0, 'w'},
        {"classes",  required_argument, 0, 'c'},
        {"frames",   required_argument, 0, 'f'},
        {"no-neon",  no_argument,       0, 'n'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "b:W:H:t:w:c:f:nh",
                              long_opts, &idx)) != -1) {
        switch (opt) {
            case 'b': cfg.backend          = optarg;          break;
            case 'W': cfg.sensor.width     = atoi(optarg);    break;
            case 'H': cfg.sensor.height    = atoi(optarg);    break;
            case 't': cfg.n_bins           = atoi(optarg);    break;
            case 'w': cfg.window_us        = atoi(optarg);    break;
            case 'c': cfg.n_classes        = atoi(optarg);    break;
            case 'f': n_frames             = atoi(optarg);    break;
            case 'n': cfg.use_neon         = false;           break;
            case 'h': print_usage(argv[0]); return 0;
            default:  print_usage(argv[0]); return 1;
        }
    }

    srand(42);
    return run_pipeline(&cfg, n_frames);
}
