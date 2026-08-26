# Async CPU→GPU Pipeline: How It Actually Works

This documents what exists today in `event_data_workflow/prefetch.py` and
`event_data_workflow/data_pipeline.py`. It is not a proposal — every claim
below is cited to a file and line.

## Data flow

```
DataLoader (num_workers>0, its own worker processes)
        │  yields CPU batches
        ▼
AsyncGPUPrefetcher            (prefetch.py:16-74)
  - background thread pulls from the loader
  - pushes into queue.Queue(maxsize=queue_size)
  - main thread's __iter__ pulls from that queue
        │  yields CPU batches, one step ahead
        ▼
CudaPrefetcher                (prefetch.py:77-142)
  - owns a dedicated torch.cuda.Stream
  - .to(device, non_blocking=True) issued on that side stream
  - keeps up to `depth` batches resident on the GPU ahead of use
        │  yields GPU-resident batches
        ▼
PrefetchedLoader               (data_pipeline.py:183-210)
  - wires the two together; this is the object training/testing iterate
        │
        ▼
Training loop (learning/training.py)
```

## Stage 1 — `AsyncGPUPrefetcher` (CPU-side queue)

A daemon thread (`produce()`, prefetch.py:53-63) reads batches from the
wrapped `DataLoader` and pushes them into a `queue.Queue(maxsize=queue_size)`.
The consumer (`__iter__`, prefetch.py:68-74) pulls from that same queue on
the main thread.

- It's **bounded and blocking**, not a lock-free ring buffer: `put_blocking`
  retries on `queue.Full` (prefetch.py:45-51), and `buf.get()` blocks until
  an item is ready. This is deliberate backpressure — an unbounded queue
  would let the producer run arbitrarily far ahead and blow up host memory.
- It matters most when `num_workers==0`: in that case nothing else overlaps
  fetching with compute, so this thread is load-bearing. With
  `num_workers>0` the DataLoader's own worker processes already overlap
  fetch and compute; this is a smaller extra buffering layer on top
  (prefetch.py:18-21).
- A sentinel object (not `None`) signals the end of iteration
  (prefetch.py:41, 63, 70-73), and any exception raised inside the producer
  thread is captured and re-raised on the consumer side (prefetch.py:59-60,
  71-72) instead of being silently swallowed.

## Stage 2 — `CudaPrefetcher` (GPU-side double buffering)

Wraps the CPU-side prefetcher and moves the *next* `depth` batches onto the
GPU ahead of when the training loop asks for them, on their own CUDA stream
(prefetch.py:86).

- `to_device()` (prefetch.py:97-101) issues `non_blocking=True` transfers.
- Those transfers happen `with torch.cuda.stream(self.stream)`
  (prefetch.py:118-119) — a side stream, not the default compute stream — so
  the copy can overlap with whatever the GPU is currently computing.
- Before handing a cross-stream tensor to the training loop,
  `record_stream()` is called on it (prefetch.py:137-138) so PyTorch's
  caching allocator knows the side stream still owns that memory and won't
  reclaim it early — this is the documented requirement for any tensor that
  crosses streams like this.
- Priming is intentionally shallow: only **one** batch is preloaded before
  the first `yield` (prefetch.py:122-128), not the full `depth` — priming
  `depth` batches up front would stall the first training step until all of
  them landed, which is invisible at `depth=1` but a real startup delay at
  higher depth. The buffer is topped back up to `depth` inside the loop
  (prefetch.py:139-141), overlapped with training instead of blocking it.
- `torch.cuda.current_stream().wait_stream(self.stream)` (prefetch.py:131)
  is the only synchronization point in this class — it makes the *default*
  stream wait for the *side* stream's copy to finish before compute reads
  the tensor. It does **not** call `torch.cuda.synchronize()` and does not
  stall the CPU thread; it's a stream-to-stream dependency, not a host
  barrier.

## Stage 3 — `PrefetchedLoader` (the object training code actually uses)

`data_pipeline.py:183-210` just wires the two stages together per
`__iter__()` call: build a fresh `AsyncGPUPrefetcher`, wrap it in a fresh
`CudaPrefetcher`, yield from that. It also stops any previous iteration
before starting a new one (data_pipeline.py:203-204, 209-210) so two
producer threads never race on the same underlying `DataLoader` at once —
this specifically matters when `num_workers==0`, where the global RNG state
would otherwise be raced.

`queue_size` defaults to `depth` if not given explicitly
(data_pipeline.py:192-196): the CPU-side queue has to be at least as deep as
the GPU-side buffer it feeds, or it becomes the tighter bottleneck and
throttles the GPU below what `depth` was chosen to sustain.

## Where `torch.cuda.synchronize()` still exists, and why

It is **not** called anywhere in the prefetch path above. It appears in two
places elsewhere, both intentional and both explained in code comments:

1. **`learning/training.py:392`** — once per epoch boundary, to safely read
   back a whole epoch's worth of `torch.cuda.Event` timing pairs
   (`elapsed_time()` raises if the paired events haven't completed yet, so a
   sync is required before reading them). This used to run once per
   *batch*; it was moved to once per *epoch* after a real run (5 epochs /
   1175 batches) exhausted CUDA event handles by holding all of them live
   for the whole run (training.py:383-391). Forward/backward timing itself
   is captured non-blockingly via `torch.cuda.Event(enable_timing=True)`
   inside `timed()` (training.py:135-153), which only queues on the stream
   and never stalls the loop — the sync is deferred to one read-back per
   epoch, not eliminated.
2. **`learning/utilities.py:348`** — inside a one-off VRAM-calibration
   probe, unrelated to the per-batch training/inference path.

So per-batch global barriers were deliberately designed out of the hot
loop; a coarse, once-per-epoch sync remains as the cost of reading back
timing data safely.
