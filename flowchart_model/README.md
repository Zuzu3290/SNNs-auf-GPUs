# Event Camera → SNN Real-Time Pipeline

A complete, modular pipeline from raw event camera sensor data to
spiking neural network classification, implemented in Python/CUDA.

```
Event Camera
    ↓  raw (x, y, t, polarity) events
Driver / SDK          [Module 2]  filters, bias tuning
    ↓  filtered EventBatch stream
Event Buffer          [Module 3]  time-window accumulation → voxel grid
    ↓  (2, T, H, W) numpy tensor
GPU Memory            [Module 4]  pinned memory + async H2D DMA
    ↓  (2, T, H, W) CUDA tensor
Custom CUDA Kernel    [Module 5]  LIF integration, threshold spike maps
    ↓  (2, T, H, W) spike tensor
SNN Simulation        [Module 6]  conv-LIF encoder + readout (snnTorch)
    ↓  (T, B, C) output spike trains
Output Decoder        [Module 7]  rate decode → class + confidence
```

---

## Quick Start (no hardware required)

```bash
# 1. Install dependencies
pip install torch numpy snntorch matplotlib

# 2. Run the full pipeline with a mock camera
python pipeline.py --backend mock --n_frames 20

# 3. Run individual module demos
python module1_camera/event_camera.py
python module2_driver/camera_driver.py
python module3_buffer/event_buffer.py
python module4_gpu/gpu_memory.py
python module5_cuda/spike_kernel.py
python module6_snn/snn_model.py
python module7_output/output_decoder.py
```

---

## Hardware Setup

### iniVation (DAVIS / DVS cameras)
```bash
pip install pyaer
python pipeline.py --backend pyaer --width 346 --height 260
```

### Prophesee (EVK3, EVK4, SilkyEvCam)
```bash
# Install from https://docs.prophesee.ai/stable/installation/
python pipeline.py --backend metavision --width 640 --height 480
```

---

## Module Reference

| Module | File | Key class |
|--------|------|-----------|
| 1 — Event Camera   | `module1_camera/event_camera.py`   | `EventCamera(backend)` |
| 2 — Driver/SDK     | `module2_driver/camera_driver.py`  | `CameraDriver` |
| 3 — Event Buffer   | `module3_buffer/event_buffer.py`   | `TimeWindowBuffer` |
| 4 — GPU Memory     | `module4_gpu/gpu_memory.py`        | `GPUMemoryManager` |
| 5 — CUDA Kernel    | `module5_cuda/spike_kernel.py`     | `SpikeKernel` |
| 6 — SNN Simulation | `module6_snn/snn_model.py`         | `SpikingClassifier` |
| 7 — Output         | `module7_output/output_decoder.py` | `RateDecoder` |
| — Master pipeline  | `pipeline.py`                      | `run_pipeline(args)` |

---

## Pipeline CLI Options

```
python pipeline.py [options]

  --backend          mock | pyaer | metavision  (default: mock)
  --width            sensor width  (default: 346)
  --height           sensor height (default: 260)
  --event-rate       simulated events/s for mock (default: 500000)
  --window-us        accumulation window µs (default: 10000)
  --n-bins           temporal bins T (default: 5)
  --n-classes        output classes (default: 10)
  --n-frames         stop after N windows, 0=forever (default: 20)
  --sensitivity      camera bias 1–10 (default: 5)
  --noise-filter-us  BAF filter window µs (default: 1500)
  --checkpoint       path to .pt model checkpoint
```

---

## Event Representations (Module 3)

| Method | Shape | Description |
|--------|-------|-------------|
| `get_voxel_grid()` | `(2, T, H, W)` | Bilinear-interpolated spatiotemporal bins |
| `get_event_frame()` | `(2, H, W)` | ON/OFF event count per pixel |
| `get_time_surface()` | `(2, H, W)` | Exponential decay timestamp surface |

---

## CUDA Kernels (Module 5)

| Kernel | Function | Description |
|--------|----------|-------------|
| `event_to_spikes_kernel` | `kernel.event_to_spikes(voxel, threshold)` | Hard threshold |
| `lif_membrane_kernel`    | `kernel.lif_spikes(voxel, v_thresh, leak)` | Stateful LIF |
| `polarity_merge_kernel`  | `kernel.merge_polarities(spikes)`           | ON/OFF → ±1 |

---

## SNN Architectures (Module 6)

| Model | Requirements | Best for |
|-------|-------------|----------|
| `SpikingClassifier` | snnTorch | Single-window classification |
| `SpikingEncoder`    | snnTorch | Feature extraction backbone |
| `SpikingRecurrent`  | Norse    | Temporal sequence tasks |

---

## Output Decoders (Module 7)

| Decoder | Method | Best for |
|---------|--------|----------|
| `RateDecoder`    | Spike count argmax | Standard classification |
| `LatencyDecoder` | Time-to-first-spike | Low-latency tasks |
