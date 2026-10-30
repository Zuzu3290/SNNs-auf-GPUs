# The prefetch layer, removed 2026-09-23

Kept here as a record. The code below was live in `event_data_workflow/` until it was
measured and found to buy nothing, and it is preserved in full so a future attempt at
overlapping data loading with computation can start from it rather than from scratch.

## Why it was removed

It was built to stop the GPU paying the CPU's batch-preparation time as idle time: a
background thread pulled batches from the DataLoader into a queue, and a second stage
copied them to the GPU on a separate CUDA stream, several batches ahead.

The repository's own A/B (`diagnostics/pipeline_objective_gpu_overlap.py`) recorded
prefetch ON at 11.53 s against OFF at 12.20 s over 25 batches -- a 5.5% win. That was a
single unrepeated pass per condition with ON always running first.

Re-measured under alternation, six rounds of 24 batches each on cached DVS128 Gesture,
same samples in the same order, single-process loading in both conditions:

| round | prefetch ON | prefetch OFF |
|---|---|---|
| 1 | 2.81 s | 2.81 s |
| 2 | 3.16 s | 2.89 s |
| 3 | 3.31 s | 3.11 s |
| 4 | 3.33 s | 3.29 s |
| 5 | 3.45 s | 3.49 s |
| 6 | 3.10 s | 2.91 s |
| **mean** | **3.19 s +/- 0.23** | **3.08 s +/- 0.27** |

Prefetching was 3.5% SLOWER on average and lost four rounds of six, with the spreads
overlapping. The earlier 5.5% win did not reproduce.

The likely reason it never paid: the DataLoader already runs worker processes that
prepare the next batch while the GPU computes, and the frames come from a warm disk
cache rather than from live event decoding, so there was little CPU time left to hide.
The queue and the extra stream added their own synchronisation and allocator pressure.

What replaced it: a synchronous `.to(device)` per batch, in `DeviceLoader`, which keeps
the contract every call site depends on -- batches arrive device-resident -- in about
fifteen lines.

## What was also deleted with it

* `resource_policy` keys `calibrate_prefetch_depth`, `prefetch_depth_fallback`,
  `prefetch_vram_fraction`, `prefetch_depth_min`, `prefetch_depth_max`
* `NeuromorphicEncoder.compute_prefetch_depth()`, reproduced below
* the prefetcher section of `tests/unit_pipeline_integration.py`, including a
  regression test for a real depth=1 data-loss bug described in the code below

## `event_data_workflow/prefetch.py`

```python
"""
Gets the next batch of data ready before the GPU asks for it, so training
never has to stop and wait.

Same idea as these two projects:
  https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
  https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
"""
from __future__ import annotations
import collections
import queue
import threading
import torch


class AsyncGPUPrefetcher:
    """Fetches batches on a background thread, so preparing the next one
    never makes the training loop wait. Load-bearing when num_workers==0
    (no DataLoader worker process would otherwise overlap fetch with
    compute); a smaller, still-cheap extra buffering layer when
    num_workers>0, since the DataLoader's own workers already overlap."""

    def __init__(self, loader, queue_size: int = 2):
        self.loader = loader
        self.queue_size = max(1, queue_size)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __len__(self) -> int:
        return len(self.loader)

    def stop(self) -> None:
        """Stops the background thread. Call this before starting a new one
        on the same data source, so two threads never read from it at once."""
        if self.thread is not None and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join()

    def __iter__(self):
        buf: queue.Queue = queue.Queue(maxsize=self.queue_size)
        sentinel = object()
        errors: list[Exception] = []
        stop_event = self.stop_event

        def put_blocking(item) -> None:
            while not stop_event.is_set():
                try:
                    buf.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue

        def produce():
            try:
                for batch in self.loader:
                    if stop_event.is_set():
                        return
                    put_blocking(batch)
            except Exception as exc:
                errors.append(exc)
            finally:
                # put_nowait here could find the queue full and silently drop the sentinel, hanging buf.get() forever.
                put_blocking(sentinel)

        self.thread = threading.Thread(target=produce, daemon=True)
        self.thread.start()

        while True:
            item = buf.get()
            if item is sentinel:
                if errors:
                    raise errors[0]
                return
            yield item


class CudaPrefetcher:
    """Moves the next `depth` batches onto the GPU ahead of time, on a copy
    stream, so the GPU is never left waiting for that transfer.
    Batches it yields are already on the GPU — no need to move them again."""

    def __init__(self, loader, device: torch.device, depth: int = 1, stream=None):
        """`stream` is the copy stream to use. PASS ONE IN whenever this object is
        rebuilt for each pass over the data, which PrefetchedLoader does per epoch.

        WHY IT MATTERS. PyTorch's caching allocator keeps a SEPARATE pool of free
        blocks per CUDA stream: a block is permanently tied to the stream that
        allocated it, and freeing it returns it to that stream's pool only. So a
        fresh stream every epoch means every epoch's prefetch queue is allocated
        from CUDA anew, while the previous epoch's queue sits free but unreachable.

        MEASURED on a Colab T4 at T=20, batch 256, depth 32: reserved memory grew
        5.05 -> 6.53 -> 8.02 -> 9.50 -> 10.99 GB across five epochs, +1.48 GB each
        time, while memory actually in use stayed flat at 2.89 GB. 1.48 GB is one
        prefetch queue (32 x 45.2 MB). At that rate a 14.56 GB card runs out around
        epoch 8 -- and the batch size was calibrated against epoch 1's free VRAM, so
        the eventual OOM looks like a batch-size problem rather than an allocator one.

        Reusing one stream lets epoch N+1 allocate out of the pool epoch N filled.
        See pytorch/pytorch#16668 ("fragmentation ... scaled with the number of
        streams you use") and NVIDIA's own data_prefetcher, which likewise builds its
        stream once and keeps it.

        The fallback keeps this class usable on its own, where one instance covers
        the whole run and a private stream is the right thing.
        """
        self.loader = loader
        self.device = device
        self.depth = max(1, depth)
        if stream is not None:
            self.stream = stream
        else:
            self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __len__(self) -> int:
        return len(self.loader)

    def stop(self) -> None:
        """Stops the wrapped prefetcher, if it has a stop method."""
        stop = getattr(self.loader, "stop", None)
        if stop is not None:
            stop()

    def to_device(self, batch):
        data, targets = batch
        data = data.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return data, targets

    def __iter__(self):
        if self.stream is None:
            # CPU run: no stream to overlap on, just forward the transfer.
            for batch in self.loader:
                yield self.to_device(batch)
            return

        it = iter(self.loader)
        pending: collections.deque = collections.deque()

        def preload_one() -> bool:
            try:
                batch = next(it)
            except StopIteration:
                return False
            with torch.cuda.stream(self.stream):
                pending.append(self.to_device(batch))
            return True

        # Prime exactly one batch so training starts as soon as it's ready —
        # priming all `depth` batches here would stall the first yield until
        # `depth` fetches finish, which is invisible at depth=1 but a real
        # startup delay at higher depth. The buffer fills to `depth` below,
        # overlapped with training instead of blocking it.
        if not preload_one():
            return

        while pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            data, targets = pending.popleft()
            # Tell the caching allocator these tensors are still in use by the
            # side stream's copy until the default stream catches up, so it
            # can't reclaim/overwrite that memory early (required whenever a
            # tensor crosses streams like this — see PyTorch's CUDA stream docs).
            data.record_stream(torch.cuda.current_stream(self.device))
            targets.record_stream(torch.cuda.current_stream(self.device))
            while len(pending) < max(1, self.depth - 1):  # depth=1 -> depth-1=0, which would never refill and silently stall after one batch -- always fetch at least the next one
                if not preload_one():
                    break
            yield data, targets
```

## `PrefetchedLoader`, from `event_data_workflow/data_pipeline.py`

```python
class PrefetchedLoader:
    """The class training and testing actually use. Underneath, it just
    combines the two prefetchers in prefetch.py: one to fetch data on the
    CPU, one to move it onto the GPU ahead of time."""

    def __init__(self, loader, device: torch.device, depth: int = 1, queue_size: int | None = None):
        self.loader = loader
        self.device = device
        self.depth = max(1, depth)
        # Unset queue_size defaults to depth: the raw CPU-side buffer feeding
        # the CUDA-stream copies must be at least as deep as the device-
        # resident buffer it feeds, or it becomes the tighter bottleneck and
        # throttles the GPU below what `depth` was chosen to sustain.
        self.queue_size = max(1, queue_size if queue_size is not None else self.depth)
        self.current: CudaPrefetcher | None = None
        # ONE copy stream for this loader's whole life, not one per epoch.
        #
        # __iter__ below builds a fresh CudaPrefetcher every pass, and that used to
        # build a fresh torch.cuda.Stream with it. PyTorch's caching allocator pools
        # free blocks PER STREAM, so each epoch allocated its prefetch queue from CUDA
        # again while the previous epoch's queue stayed free-but-unreachable: reserved
        # memory climbed by one queue per epoch (measured: +1.48 GB, T=20/batch 256/
        # depth 32) with memory actually in use flat. See CudaPrefetcher.__init__.
        #
        # A stream is a long-lived handle, not per-iteration state -- nothing about
        # re-iterating invalidates it, and stop() only ends the CPU-side feeder thread.
        # Only one iterator is ever live per loader (see __iter__), so nothing else can
        # be issuing copies on it at the same time.
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self.current is not None:
            self.current.stop()
        async_stage = AsyncGPUPrefetcher(self.loader, queue_size=self.queue_size)
        self.current = CudaPrefetcher(async_stage, self.device, depth=self.depth,
                                      stream=self.stream)
        try:
            yield from self.current
        finally:
            self.current.stop()  # closes any abandoned iteration immediately, not just on the next __iter__() call — else the leaked thread races the global RNG (num_workers=0) against whatever runs next
```

## `NeuromorphicEncoder.compute_prefetch_depth()`

```python
    def compute_prefetch_depth(self, batch_size: int) -> int:
        """How many batches to keep queued ahead of the GPU, sized from live
        VRAM and this dataset's real per-sample size — not a fixed constant,
        so a large-sensor dataset (bigger batches) or a smaller card (less
        headroom) both get a depth that actually fits, instead of one number
        tuned for whichever dataset/GPU it happened to be set on.

        This runs after calibrate_batch_size() has already freed its own
        probe allocations but before real training has claimed anything —
        a live VRAM snapshot at this point looks more available than it's
        about to be. Subtracting batch_vram_fraction of total VRAM (the
        share calibrate_batch_size already earmarked for the real training
        step) before sizing the queue keeps the two from double-booking the
        same memory. Fraction/min/max all read from resource_policy in
        data_workflow.yaml, not fixed here."""
        if not self.wf.CALIBRATE_PREFETCH_DEPTH:
            return self.wf.PREFETCH_DEPTH_FALLBACK
        batch_bytes = self.batch_sample_bytes * batch_size
        metrics = monitor.snapshot()
        reserved_for_training_gb = metrics.gpu_memory_gb * self.wf.BATCH_VRAM_FRACTION
        true_available_gb = max(0.0, metrics.gpu_available_gb - reserved_for_training_gb)
        if batch_bytes <= 0 or true_available_gb <= 0:
            return self.wf.PREFETCH_DEPTH_MIN
        budget_bytes = true_available_gb * self.wf.PREFETCH_VRAM_FRACTION * (1024 ** 3)
        return max(self.wf.PREFETCH_DEPTH_MIN, min(int(budget_bytes / batch_bytes), self.wf.PREFETCH_DEPTH_MAX))
```
