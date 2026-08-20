EVENT-BASED CAMERA PIPELINE — IMPLEMENTATION NOTES
====================================================

OVERVIEW
--------
The pipeline ingests raw events from a DVS (Dynamic Vision Sensor) event camera
and prepares them for SNN training. Unlike frame-based cameras, a DVS outputs
asynchronous (x, y, polarity, timestamp) tuples whenever a pixel detects a change
in log-luminance. This means recordings have variable length and variable event
density rather than a fixed frame rate.

The entry point is NeuromorphicEncoder in event_data_workflow/data_pipeline.py.
It enforces a strict four-layer order to avoid correctness bugs (e.g. caching
slices instead of recordings, applying transforms before caching):

    Layer 1 — Raw dataset load        (no transforms, no caching)
    Layer 2 — Adaptive cache          (raw recordings only)
    Layer 3 — Temporal slicing        (stateless, via tonic)
    Layer 4 — DataLoader construction (coordinator-driven worker/prefetch config)


SUPPORTED DATASETS & DOWNLOAD SIZES
------------------------------------
Registered in DATASET_REGISTRY (event_data_workflow/data_pipeline.py). Sizes
below are the COMPRESSED download size (what tonic pulls over the network on
first run into tmp/data/), not the size after caching/framing.

    #  Dataset          Classes  Samples                  Download size                                   Status (this machine)
    1  N-MNIST          10       70,000 (60k train/10k test) 1.18 GB  (train.zip 965 MB + test.zip 162 MB)  Downloaded — pipeline verified (diagnostics/verify_cache_fix.py)
    2  N-Caltech101     101      8,709                     3.72 GB  (single zip, Mendeley-hosted)           Not yet downloaded
    3  ASL-DVS          26       100,800 (4,200/letter)     not published (see note below)                  Not yet downloaded
    4  DVS128 Gesture   11       1,464 (1,176 train/288 test) ~3 GB tar / ~5 GB extracted, train+test combined  Not yet downloaded
    5  DSEC             —        —                          (out of scope — already known)                  —

Notes on how these numbers were obtained:
    N-Caltech101 — measured directly: the Mendeley download link 302-redirects
        to an S3 object with an exact Content-Length of 3,989,375,649 bytes.
        One monolithic zip (all 101 classes); tonic has no partial-download path.

    ASL-DVS — no official size is published anywhere (dataset repo, paper, or
        tonic's source). The download is a single Dropbox shared-folder zip
        containing all 26 letters as one archive (100,800 .mat files total,
        ~3,900 samples/letter); Dropbox doesn't expose a Content-Length for
        folder-zip links, and there's no per-class/per-file download offered,
        so the only way to get a real number is to let the download run.
        Given N-Caltech101's ~460 KB/sample average and ASL-DVS having ~12x
        as many samples, expect this to be the largest of the three by a wide
        margin — plan for it separately, not as a quick smoke test.

    DVS128 Gesture — the ~3 GB tar / ~5 GB extracted figure is the published
        aggregate (train+test combined) from a community-maintained mirror
        README, not measured directly here: figshare's ndownloader links sit
        behind an AWS WAF bot challenge that blocks HEAD/range probing.
        train_url and test_url are separate archives (ibmGestureTrain.tar.gz,
        ibmGestureTest.tar.gz); test has 288/1,464 samples (~20%), so the
        test-only download is an ESTIMATE of roughly 0.6 GB tar — useful as
        the smaller of the two splits for a partial smoke test, not a
        confirmed figure.


DATA FORMAT
-----------
Raw events from tonic datasets are structured NumPy arrays with dtype fields:
    t  — timestamp in microseconds (uint32/int64)
    x  — pixel column
    y  — pixel row
    p  — polarity: 0 = OFF (decreasing luminance), 1 = ON (increasing luminance)

Sensor size for N-MNIST: (34, 34, 2) — width × height × polarities.
For the DVS dataset from the professor, sensor size is read from the dataset
itself via full_raw.sensor_size.

The ToFrame transform from tonic converts the raw event stream into a dense
tensor of shape [T, C, H, W] where T is the number of time windows, C=2
(ON/OFF polarity channels), H and W are the sensor height and width.


TEMPORAL SLICING — TONIC SlicedDataset
---------------------------------------
Previously the project had a hand-written TemporalSlicedDataset class that
replicated what tonic already provides natively. This has been replaced entirely
with tonic's built-in machinery:

    from tonic import SlicedDataset
    from tonic.slicers import SliceByTime, SliceByEventCount

SliceByTime(time_window=N_microseconds)
    Cuts each recording into windows of fixed duration. Good when recordings
    have consistent length and temporal consistency matters (e.g. RNNs).

SliceByEventCount(event_count=N)
    Cuts each recording into windows of fixed event count regardless of time.
    Good when activity density varies across recordings and spatial consistency
    matters more than temporal consistency (e.g. CNNs).

SlicedDataset builds a slice index (start/end positions per recording) once at
construction and stores it as HDF5 at metadata_path. Subsequent runs load the
index from disk instead of rebuilding it, which is significant for large datasets.

The pipeline passes:
    metadata/train/slice_metadata.h5  — for the training split
    metadata/test/slice_metadata.h5   — for the test split

AdaptiveTemporalSlicer (event_data_workflow/temporal_slicer.py) has since been
removed entirely — nothing in the codebase references it any more. Its
successor is calibrate_events_per_slice() (event_data_workflow/data_pipeline.py,
Case A of Case_Study_Evaluation_Report.pdf): it samples recordings, measures
their event-count distribution, and derives an events_per_slice value for
SliceByEventCount instead of a guessed constant. It's the project's only
remaining custom slicing code, and is opt-in via
configuration/data_workflow.yaml's temporal_slicing.calibrate_events_per_slice
(false by default — timing-window slicing, SliceByTime, remains the default
method either way).


CACHING SYSTEM — AdaptiveCacheController
-----------------------------------------
Raw recordings are cached before slicing so that all slices derived from one
recording share a single cache entry. Caching after slicing would multiply
RAM usage by the slice expansion factor (e.g. 6x recordings = 6x cache entries
for data that is identical up to the slice boundaries).

Hardware topology is fixed and singular: CPU always loads and caches
recordings, GPU always trains. VRAM is never a cache storage target — the
GPU is only ever the training device. The controller chooses where one
dataset's cache lives among RAM and disk, not between hardware
configurations.

Strategy selection (event_data_workflow/cache_engine.py):

    memory      — full dataset fits in RAM → tonic MemoryCachedDataset
    disk        — limited RAM but disk available → tonic DiskCachedDataset
    hybrid      — large RAM (≥32 GB) with disk → DiskCachedDataset + hot RAM layer
    no_cache    — fallback, on-the-fly processing

The selection is driven by live system metrics from SystemResourceMonitor
(available RAM, disk space). GPU pressure (VRAM usage > 75%) forces the disk
strategy to prevent the cache and CUDA's pinned-memory allocator from
competing for the same physical RAM.

Cache eviction policy — plain FIFO:
    A single deque tracks insertion order; the oldest entry is evicted once
    the cache is full. This pipeline's access pattern is a shuffled
    DataLoader — every recording is equally likely to recur each epoch,
    with no popularity skew for a more elaborate policy to exploit, so
    plain FIFO gives the same practical hit rate with far less bookkeeping.


GPU REQUIREMENT
---------------
A CUDA-capable GPU is a hard requirement. The application raises RuntimeError
immediately on startup (learning/main.py) if no GPU is detected. All
torch.cuda.is_available() guards below the entry point have been removed — they
were dead code given this constraint.

NVML (pynvml) is used for real power readings during training via
GPUStats (event_data_workflow/gpu_stats.py). Power is sampled in a background
thread every 0.5 seconds alongside compute utilisation. Energy per sample is
estimated as: avg_power_W × elapsed_seconds.


TRANSFORMS
----------
The transform chain applied to each sample:

    tonic.transforms.Denoise(filter_time=10000)
        Removes noise events: discards any event that has no neighbour within
        10 ms in the (x, y, polarity) neighbourhood. Reduces spike noise from
        the sensor.

    tonic.transforms.ToFrame(sensor_size, time_window)
        Converts the raw event stream to a dense [T, C, H, W] NumPy array.
        Each bin accumulates all events within time_window microseconds.

    torch.from_numpy
        Converts the NumPy array to a PyTorch tensor (zero-copy, shared memory).

    torchvision.transforms.RandomRotation([-10, 10])   [train only]
        Light spatial augmentation. Applied only to the training split.

All transforms are named, picklable objects — compatible with PyTorch's
multiprocessing DataLoader workers without any special handling.


DATALOADER CONFIG — PipelineMemoryCoordinator
----------------------------------------------
Worker count, prefetch factor, pin_memory, and persistent_workers are set
by PipelineMemoryCoordinator (event_data_workflow/pipeline_coordinator.py)
based on available system RAM:

    CPU + GPU mode (worker_budget_gb ≥ 0.5 GB):
        num_workers      = min(settings.NUM_WORKERS, cpu_count)
        prefetch_factor  = 2
        pin_memory       = True   (page-locked H→D transfers via non_blocking=True)
        persistent_workers = True  (workers persist across epochs)

    GPU-only mode (worker_budget_gb < 0.5 GB):
        num_workers      = 0      (no forking, single-process loading)
        pin_memory       = False  (no H→D boundary to optimise)
        persistent_workers = False

When GPU VRAM usage exceeds 75%, the recording cache allocation is halved to
prevent the model and cache from competing for the remaining VRAM.
