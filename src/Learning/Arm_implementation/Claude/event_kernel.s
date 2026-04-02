// =============================================================================
// asm/event_kernel.s  —  ARM64 (AArch64) Event Processing Kernels
// =============================================================================
// Implements the hot-path event processing routines in ARM assembly:
//
//   event_threshold_neon   — SIMD batch threshold: events[] → spikes[]
//   lif_membrane_neon      — SIMD LIF membrane update for N pixels
//   voxel_accumulate       — scatter events into (T,H,W) voxel grid
//   polarity_split         — separate ON/OFF events into two arrays
//   event_count_xy         — count events per pixel (histogram)
//
// Calling convention: ARM64 AAPCS
//   x0–x7   integer / pointer arguments
//   v0–v7   SIMD float arguments
//   x0      integer return value
//   v0      float return value
//   x9–x15  caller-saved scratch
//   x19–x28 callee-saved (must preserve)
//
// NEON SIMD registers used:
//   v0–v15  working registers (16 × float32 lanes each = v_4s type)
//   Each v register = 128 bits = 4 × float32 (suffix .4s)
//                              = 8 × float16 (suffix .8h)
//                              = 16 × uint8  (suffix .16b)
//
// Build:
//   as --warn -o build/obj/asm_event_kernel.o asm/event_kernel.s
// =============================================================================

    .arch   armv8-a+simd
    .text
    .align  4

// =============================================================================
// Macro: function prologue / epilogue
// =============================================================================
.macro FUNC_BEGIN name
    .global \name
    .type   \name, %function
\name:
.endm

.macro FUNC_END name
    .size \name, . - \name
.endm


// =============================================================================
// 1. event_threshold_neon
// =============================================================================
// Purpose: Apply a hard threshold to a voxel float array.
//          output[i] = (input[i] >= threshold) ? 1.0f : 0.0f
//
// C signature:
//   void event_threshold_neon(
//       const float* input,    // x0 — source array (aligned to 16 bytes)
//       float*       output,   // x1 — destination array
//       int          count,    // x2 — number of elements
//       float        threshold // s0 — threshold value (float scalar)
//   );
//
// Strategy: Process 16 floats per iteration (4 NEON vectors × 4 lanes).
//           Remainder handled scalar.
// =============================================================================
FUNC_BEGIN event_threshold_neon

    // Broadcast threshold scalar → all 4 lanes of v16
    dup     v16.4s, v0.s[0]

    // Preload 1.0f into v17 for masking
    fmov    v17.4s, #1.0

    // x3 = count / 16  (main loop iterations)
    lsr     x3, x2, #4
    cbz     x3, .Lthresh_tail

.Lthresh_main:
    // Load 4 vectors (16 floats)
    ldp     q0, q1, [x0], #32
    ldp     q2, q3, [x0], #32

    // Compare ≥ threshold  → sets all-ones mask in result lanes
    fcmge   v4.4s, v0.4s, v16.4s
    fcmge   v5.4s, v1.4s, v16.4s
    fcmge   v6.4s, v2.4s, v16.4s
    fcmge   v7.4s, v3.4s, v16.4s

    // AND with 1.0f mask: result = mask ? 1.0f : 0.0f
    and     v4.16b, v4.16b, v17.16b
    and     v5.16b, v5.16b, v17.16b
    and     v6.16b, v6.16b, v17.16b
    and     v7.16b, v7.16b, v17.16b

    // Store 16 results
    stp     q4, q5, [x1], #32
    stp     q6, q7, [x1], #32

    subs    x3, x3, #1
    b.ne    .Lthresh_main

.Lthresh_tail:
    // Handle remaining elements (count % 16) scalar
    and     x3, x2, #15
    cbz     x3, .Lthresh_done

.Lthresh_scalar:
    ldr     s0, [x0], #4
    fcmge   s1, s0, s16
    and     v1.8b, v1.8b, v17.8b
    str     s1, [x1], #4
    subs    x3, x3, #1
    b.ne    .Lthresh_scalar

.Lthresh_done:
    ret

FUNC_END event_threshold_neon


// =============================================================================
// 2. lif_membrane_neon
// =============================================================================
// Purpose: Leaky-Integrate-and-Fire update for a 2D pixel array.
//          For each pixel:
//            V[i] = leak * V[i] + input[i]
//            if V[i] >= v_thresh:
//              spike[i] = 1.0f
//              V[i]     = v_reset
//            else:
//              spike[i] = 0.0f
//
// C signature:
//   void lif_membrane_neon(
//       const float* input,     // x0 — input current  (N floats)
//       float*       membrane,  // x1 — membrane state (N floats, in/out)
//       float*       spikes,    // x2 — spike output   (N floats)
//       int          N,         // x3 — number of pixels
//       float        leak,      // s0 — decay factor (e.g. 0.9)
//       float        v_thresh,  // s1 — spike threshold
//       float        v_reset    // s2 — post-spike reset
//   );
//
// Registers:
//   v16 = leak (broadcast)
//   v17 = v_thresh (broadcast)
//   v18 = v_reset (broadcast)
//   v19 = 1.0f (broadcast, for spike output)
// =============================================================================
FUNC_BEGIN lif_membrane_neon

    // Save callee-saved SIMD registers we'll use (v8–v15 are callee-saved)
    stp     d8,  d9,  [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    // Broadcast scalar params to SIMD vectors
    dup     v16.4s, v0.s[0]    // leak
    dup     v17.4s, v1.s[0]    // v_thresh
    dup     v18.4s, v2.s[0]    // v_reset
    fmov    v19.4s, #1.0       // 1.0f for spike output

    // Main loop: 4 pixels at a time
    lsr     x4, x3, #2
    cbz     x4, .Llif_tail

.Llif_main:
    // Load input current and membrane state (4 floats each)
    ldr     q0, [x0], #16      // input[i..i+3]
    ldr     q1, [x1]           // membrane[i..i+3]

    // V = leak * V + input
    fmla    v0.4s, v1.4s, v16.4s  // v0 = input + leak*membrane
                                    // (fmla: a += b*c)

    // fired = V >= v_thresh  (all-ones mask in fired lanes)
    fcmge   v2.4s, v0.4s, v17.4s

    // spike = fired ? 1.0f : 0.0f
    and     v3.16b, v2.16b, v19.16b

    // membrane = fired ? v_reset : V
    bif     v0.16b, v18.16b, v2.16b   // v0 = fired ? v18 : v0
                                        // bif: bit-insert-if-false

    // Store updated membrane and spikes
    str     q0, [x1], #16
    str     q3, [x2], #16

    subs    x4, x4, #1
    b.ne    .Llif_main

.Llif_tail:
    // Handle N % 4 scalar remainder
    and     x4, x3, #3
    cbz     x4, .Llif_done

.Llif_scalar:
    ldr     s0, [x0], #4       // input
    ldr     s1, [x1]           // membrane

    // V = leak*V + input
    fmul    s1, s1, s16
    fadd    s0, s0, s1

    // Check threshold
    fcmp    s0, s17
    b.lt    .Llif_no_spike

    // Spike: output=1, reset membrane
    fmov    s3, #1.0
    str     s18, [x1], #4      // membrane = v_reset
    str     s3,  [x2], #4      // spike = 1.0
    b       .Llif_scalar_next

.Llif_no_spike:
    str     s0,  [x1], #4      // membrane = V
    fmov    s3, wzr             // spike = 0.0
    str     s3,  [x2], #4

.Llif_scalar_next:
    subs    x4, x4, #1
    b.ne    .Llif_scalar

.Llif_done:
    // Restore callee-saved SIMD registers
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8,  d9,  [sp], #64
    ret

FUNC_END lif_membrane_neon


// =============================================================================
// 3. voxel_accumulate
// =============================================================================
// Purpose: Scatter-add events into a (T, H, W) voxel grid.
//          For each event (x, y, t_bin):
//            voxel[t_bin * H * W + y * W + x] += weight
//
// C signature:
//   void voxel_accumulate(
//       const uint16_t* xs,      // x0 — event x coords
//       const uint16_t* ys,      // x1 — event y coords
//       const uint8_t*  t_bins,  // x2 — temporal bin index per event
//       const float*    weights, // x3 — bilinear interpolation weight
//       float*          voxel,   // x4 — (T,H,W) output grid (float32)
//       int             N,       // x5 — number of events
//       int             W,       // x6 — sensor width
//       int             HW       // x7 — H*W (precomputed stride)
//   );
//
// Note: scatter operations can't be vectorized trivially (data-dependent
//       memory addresses). We use scalar loop but keep it pipeline-friendly.
// =============================================================================
FUNC_BEGIN voxel_accumulate

    cbz     x5, .Lvoxel_done

    // x8 = W as 64-bit for multiply
    uxtw    x8, w6
    // x9 = HW as 64-bit
    uxtw    x9, w7

.Lvoxel_loop:
    // Load event coordinates
    ldrh    w10, [x0], #2      // x coord
    ldrh    w11, [x1], #2      // y coord
    ldrb    w12, [x2], #1      // t_bin
    ldr     s0,  [x3], #4      // weight

    // Compute flat index: t_bin*HW + y*W + x
    uxtw    x10, w10
    uxtw    x11, w11
    uxtw    x12, w12

    mul     x12, x12, x9       // t_bin * HW
    madd    x11, x11, x8, x12  // + y * W
    add     x11, x11, x10      // + x

    // Load current voxel value, add weight, store back
    ldr     s1, [x4, x11, lsl #2]   // voxel[idx] (×4 for float32 byte offset)
    fadd    s1, s1, s0
    str     s1, [x4, x11, lsl #2]

    subs    x5, x5, #1
    b.ne    .Lvoxel_loop

.Lvoxel_done:
    ret

FUNC_END voxel_accumulate


// =============================================================================
// 4. polarity_split
// =============================================================================
// Purpose: Partition an array of events by polarity into two output arrays.
//          ON events  (polarity == 1) → on_out[]
//          OFF events (polarity == 0) → off_out[]
//          Returns counts in x0 (on_count) and x1 (off_count) via pointer.
//
// C signature:
//   void polarity_split(
//       const float*   input,     // x0 — interleaved event values
//       const uint8_t* polarities,// x1 — polarity per event (0 or 1)
//       float*         on_out,    // x2 — ON output buffer
//       float*         off_out,   // x3 — OFF output buffer
//       int            N,         // x4 — input length
//       int*           on_count,  // x5 — written: number of ON events
//       int*           off_count  // x6 — written: number of OFF events
//   );
// =============================================================================
FUNC_BEGIN polarity_split

    // x7 = ON counter, x8 = OFF counter
    mov     x7, #0
    mov     x8, #0

    cbz     x4, .Lsplit_done

.Lsplit_loop:
    ldrb    w9,  [x1], #1      // polarity byte
    ldr     s0,  [x0], #4      // input value

    cbnz    w9, .Lsplit_on

.Lsplit_off:
    str     s0, [x3], #4       // off_out[off_count] = val
    add     x8, x8, #1
    b       .Lsplit_next

.Lsplit_on:
    str     s0, [x2], #4       // on_out[on_count] = val
    add     x7, x7, #1

.Lsplit_next:
    subs    x4, x4, #1
    b.ne    .Lsplit_loop

.Lsplit_done:
    str     w7, [x5]           // *on_count  = x7
    str     w8, [x6]           // *off_count = x8
    ret

FUNC_END polarity_split


// =============================================================================
// 5. event_count_xy
// =============================================================================
// Purpose: Build a 2D event-count histogram for a frame (H×W).
//          histogram[y * W + x] += 1  for each event
//
// C signature:
//   void event_count_xy(
//       const uint16_t* xs,      // x0 — x coords
//       const uint16_t* ys,      // x1 — y coords
//       uint32_t*       hist,    // x2 — H*W count array (zero-initialised)
//       int             N,       // x3 — number of events
//       int             W        // x4 — sensor width
//   );
// =============================================================================
FUNC_BEGIN event_count_xy

    cbz     x3, .Lhist_done
    uxtw    x4, w4              // W as 64-bit

.Lhist_loop:
    ldrh    w5, [x0], #2       // x
    ldrh    w6, [x1], #2       // y

    uxtw    x5, w5
    uxtw    x6, w6

    // flat_idx = y * W + x
    madd    x5, x6, x4, x5

    // Atomic-style increment (single-threaded: plain load-add-store)
    ldr     w6, [x2, x5, lsl #2]
    add     w6, w6, #1
    str     w6, [x2, x5, lsl #2]

    subs    x3, x3, #1
    b.ne    .Lhist_loop

.Lhist_done:
    ret

FUNC_END event_count_xy


// =============================================================================
// 6. dot_product_f32_neon
// =============================================================================
// Purpose: Fused dot product — used by the FC readout layer in the SNN.
//          result = sum(a[i] * b[i]) for i in [0, N)
//
// C signature:
//   float dot_product_f32_neon(
//       const float* a,   // x0
//       const float* b,   // x1
//       int          N    // x2
//   );
//
// Uses NEON horizontal add for reduction.
// =============================================================================
FUNC_BEGIN dot_product_f32_neon

    // Accumulator: v_acc = 0
    movi    v4.4s, #0
    movi    v5.4s, #0

    // Main loop: 8 floats at a time (2 NEON vectors)
    lsr     x3, x2, #3
    cbz     x3, .Ldot_tail

.Ldot_main:
    ldp     q0, q1, [x0], #32     // a[0..7]
    ldp     q2, q3, [x1], #32     // b[0..7]

    fmla    v4.4s, v0.4s, v2.4s   // acc0 += a[0..3] * b[0..3]
    fmla    v5.4s, v1.4s, v3.4s   // acc1 += a[4..7] * b[4..7]

    subs    x3, x3, #1
    b.ne    .Ldot_main

    // Merge accumulators
    fadd    v4.4s, v4.4s, v5.4s

.Ldot_tail:
    // Scalar remainder
    and     x3, x2, #7
    cbz     x3, .Ldot_reduce

.Ldot_scalar:
    ldr     s0, [x0], #4
    ldr     s1, [x1], #4
    fmla    v4.s[0], s0, s1        // accumulate into lane 0
    subs    x3, x3, #1
    b.ne    .Ldot_scalar

.Ldot_reduce:
    // Horizontal add: sum all 4 lanes → s0
    faddp   v4.4s, v4.4s, v4.4s   // [a+b, c+d, a+b, c+d]
    faddp   v4.4s, v4.4s, v4.4s   // [a+b+c+d, ...]
    fmov    s0, v4.s[0]

    ret

FUNC_END dot_product_f32_neon


// =============================================================================
// 7. relu_inplace_neon
// =============================================================================
// Purpose: In-place ReLU activation — clamp negatives to zero.
//          x[i] = max(x[i], 0.0f)
//
// C signature:
//   void relu_inplace_neon(float* x, int N);
//
// =============================================================================
FUNC_BEGIN relu_inplace_neon

    movi    v16.4s, #0          // zero vector

    lsr     x2, x1, #4         // N / 16
    cbz     x2, .Lrelu_tail

.Lrelu_main:
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]

    fmax    v0.4s, v0.4s, v16.4s
    fmax    v1.4s, v1.4s, v16.4s
    fmax    v2.4s, v2.4s, v16.4s
    fmax    v3.4s, v3.4s, v16.4s

    stp     q0, q1, [x0], #32
    stp     q2, q3, [x0], #32

    subs    x2, x2, #1
    b.ne    .Lrelu_main

.Lrelu_tail:
    and     x2, x1, #15
    cbz     x2, .Lrelu_done

.Lrelu_scalar:
    ldr     s0, [x0]
    fmax    s0, s0, s16
    str     s0, [x0], #4
    subs    x2, x2, #1
    b.ne    .Lrelu_scalar

.Lrelu_done:
    ret

FUNC_END relu_inplace_neon


// =============================================================================
// End of file
// =============================================================================
    .end
