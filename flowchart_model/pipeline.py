"""
Event Camera SNN Pipeline — Master Integration
================================================
Wires all 7 modules into a single real-time inference loop.

Run with mock hardware (no camera needed):
    python pipeline.py --backend mock --n_classes 10 --n_frames 20

Run with real iniVation hardware:
    python pipeline.py --backend pyaer

Run with Prophesee hardware:
    python pipeline.py --backend metavision
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Module imports
sys.path.insert(0, str(Path(__file__).parent / "module1_camera"))
sys.path.insert(0, str(Path(__file__).parent / "module2_driver"))
sys.path.insert(0, str(Path(__file__).parent / "module3_buffer"))
sys.path.insert(0, str(Path(__file__).parent / "module4_gpu"))
sys.path.insert(0, str(Path(__file__).parent / "module5_cuda"))
sys.path.insert(0, str(Path(__file__).parent / "module6_snn"))
sys.path.insert(0, str(Path(__file__).parent / "module7_output"))

from event_camera  import EventCamera
from camera_driver import CameraDriver
from event_buffer  import TimeWindowBuffer
from gpu_memory    import GPUMemoryManager, get_best_device
from spike_kernel  import SpikeKernel
from snn_model     import SpikingClassifier
from output_decoder import RateDecoder, plot_spike_raster

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Default class names (replace with your dataset's labels)
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "background", "car", "pedestrian", "cyclist",
    "truck", "bus", "sign", "light", "object", "other"
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    print("\n" + "="*60)
    print(" Event Camera → SNN Real-Time Pipeline")
    print("="*60 + "\n")

    device = get_best_device()
    print(f"Device:        {device}")
    print(f"Camera:        {args.backend}")
    print(f"Sensor:        {args.width}×{args.height}")
    print(f"Window:        {args.window_us} µs")
    print(f"Temporal bins: {args.n_bins}")
    print(f"Classes:       {args.n_classes}\n")

    # ------------------------------------------------------------------
    # Module 2: Driver + filters
    # ------------------------------------------------------------------
    driver = CameraDriver(
        backend      = args.backend,
        sensor_size  = (args.width, args.height),
        event_rate_hz = args.event_rate,
    )
    driver.configure(
        sensitivity         = args.sensitivity,
        noise_filter_us     = args.noise_filter_us,
        hot_pixel_threshold = args.hot_pixel_threshold,
    )

    # ------------------------------------------------------------------
    # Module 3: Time-window buffer
    # ------------------------------------------------------------------
    buf = TimeWindowBuffer(
        sensor_size = (args.width, args.height),
        window_us   = args.window_us,
        n_bins      = args.n_bins,
    )

    # ------------------------------------------------------------------
    # Module 4: GPU memory
    # ------------------------------------------------------------------
    gm = GPUMemoryManager(
        device       = device,
        sensor_size  = (args.width, args.height),
        n_bins       = args.n_bins,
        double_buffer = True,
    )
    gm.allocate()

    # ------------------------------------------------------------------
    # Module 5: CUDA spike kernel
    # ------------------------------------------------------------------
    kernel = SpikeKernel(backend="auto", device=device)

    # ------------------------------------------------------------------
    # Module 6: SNN model
    # ------------------------------------------------------------------
    if HAS_TORCH:
        model = SpikingClassifier(
            sensor_size = (args.width, args.height),
            n_bins      = args.n_bins,
            n_classes   = args.n_classes,
            hidden_ch   = 64,
        ).to(torch.device(device))
        model.eval()

        # Optionally load checkpoint
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = None
        print("WARNING: PyTorch not available — SNN step will be skipped")

    # ------------------------------------------------------------------
    # Module 7: Decoder
    # ------------------------------------------------------------------
    n_names   = min(args.n_classes, len(CLASS_NAMES))
    names     = CLASS_NAMES[:n_names] + [f"class_{i}" for i in range(n_names, args.n_classes)]
    decoder   = RateDecoder(n_classes=args.n_classes, class_names=names)

    # ------------------------------------------------------------------
    # Streaming loop
    # ------------------------------------------------------------------
    print("Starting pipeline... (Ctrl+C to stop)\n")
    driver.start()

    frame_count  = 0
    total_events = 0
    t_pipeline   = time.perf_counter()
    latencies    = []

    print(f"{'Frame':>6}  {'Events':>8}  {'Class':>14}  {'Conf':>6}  {'ms':>8}  {'FPS':>7}")
    print("-" * 60)

    try:
        for batch in driver.stream(duration_ms=args.window_us / 1000.0):
            total_events += batch.count

            # Module 3: accumulate into window
            if not buf.push(batch):
                continue   # window not complete yet

            t0 = time.perf_counter()

            # Module 3 → voxel grid (numpy, CPU)
            voxel_np = buf.get_voxel_grid()   # (2, T, H, W)

            # Module 4: H2D transfer
            voxel_gpu = gm.push(voxel_np)

            prediction = None
            if HAS_TORCH and model is not None:
                # Module 5: event → spikes via CUDA kernel
                voxel_in = voxel_gpu.unsqueeze(0)   # add batch dim
                spikes   = kernel.lif_spikes(
                    voxel_in, v_thresh=0.5, leak=0.9)

                # Module 6: SNN forward
                with torch.no_grad():
                    spk_rec, _ = model(voxel_in)

                # Module 7: decode
                wall_ms    = (time.perf_counter() - t0) * 1000
                prediction = decoder.decode(spk_rec, wall_time_ms=wall_ms)
                latencies.append(wall_ms)

            frame_count += 1

            if prediction:
                fps = frame_count / (time.perf_counter() - t_pipeline)
                print(f"{frame_count:>6}  {total_events:>8,}  "
                      f"{prediction.class_name:>14}  "
                      f"{prediction.confidence*100:>5.1f}%  "
                      f"{prediction.wall_time_ms:>8.2f}  "
                      f"{fps:>7.1f}")
            else:
                print(f"{frame_count:>6}  {total_events:>8,}  "
                      f"{'–':>14}  {'–':>6}  {'–':>8}  {'–':>7}")

            if args.n_frames and frame_count >= args.n_frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        driver.stop()
        gm.free()

    # ------------------------------------------------------------------
    # Final stats
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_pipeline
    print("\n" + "="*60)
    print(" Pipeline Stats")
    print("="*60)
    print(f"  Frames processed: {frame_count:,}")
    print(f"  Total events:     {total_events:,}")
    print(f"  Wall time:        {elapsed:.2f} s")
    print(f"  Throughput:       {frame_count/elapsed:.1f} windows/s")
    if latencies:
        a = np.array(latencies)
        print(f"  Latency mean:     {a.mean():.2f} ms")
        print(f"  Latency P95:      {np.percentile(a,95):.2f} ms")
        print(f"  Effective FPS:    {1000/a.mean():.1f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Event Camera → SNN Real-Time Pipeline")

    parser.add_argument("--backend",    default="mock",
                        choices=["mock","pyaer","metavision"],
                        help="Camera backend")
    parser.add_argument("--width",      type=int,   default=346)
    parser.add_argument("--height",     type=int,   default=260)
    parser.add_argument("--event-rate", type=int,   default=500_000,
                        help="Simulated event rate (mock only)")
    parser.add_argument("--window-us",  type=int,   default=10_000,
                        help="Time window in microseconds")
    parser.add_argument("--n-bins",     type=int,   default=5,
                        help="Temporal bins for voxel grid")
    parser.add_argument("--n-classes",  type=int,   default=10)
    parser.add_argument("--n-frames",   type=int,   default=20,
                        help="Stop after N windows (0 = run forever)")
    parser.add_argument("--sensitivity",type=int,   default=5)
    parser.add_argument("--noise-filter-us", type=int, default=1500)
    parser.add_argument("--hot-pixel-threshold", type=int, default=0)
    parser.add_argument("--checkpoint", type=str,   default="",
                        help="Path to model .pt checkpoint")

    args = parser.parse_args()
    run_pipeline(args)
