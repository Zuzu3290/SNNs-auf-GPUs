"""
Keeps a DataLoader one batch ahead of the training loop in a background thread,
and overlaps the host->device copy itself with GPU compute via a CUDA stream,
optionally keeping several device-resident batches queued at once.
"""
from __future__ import annotations
import collections
import queue
import threading
import torch


class AsyncGPUPrefetcher:
    """
    Wraps a DataLoader and stays one batch ahead in a background thread, so
    training doesn't block on data prep while the GPU works. Uses a thread,
    not a process, so it's safe with CUDA-resident caches.
    """

    def __init__(self, loader, queue_size: int = 2):
        self.loader = loader
        self.queue_size = max(1, queue_size)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        buf: queue.Queue = queue.Queue(maxsize=self.queue_size)
        sentinel = object()
        errors: list[Exception] = []

        def produce():
            try:
                for batch in self.loader:
                    buf.put(batch)
            except Exception as exc:
                errors.append(exc)
            finally:
                buf.put(sentinel)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()

        while True:
            item = buf.get()
            if item is sentinel:
                if errors:
                    raise errors[0]
                return
            yield item


class CudaPrefetcher:
    """
    Wraps a batch iterator (typically AsyncGPUPrefetcher, so CPU-side batch
    prep and the H2D copy are both overlapped) and issues the host->device
    copy for the next `depth` batches on a dedicated CUDA stream while the
    current batch is still computing on the default stream — the standard
    "DataPrefetcher" pattern (NVIDIA's ImageNet examples, timm's
    PrefetchLoader), extended to a configurable lookahead instead of just
    one batch ahead. Closes the transfer gap that a plain `.to(device,
    non_blocking=True)` call inside the training loop still leaves on the
    GPU, since that copy is issued only once the previous batch's work is
    already done, not ahead of time.

    `depth` batches sit device-resident at once — cheap in VRAM (a 128-
    sample event-frame batch is typically a few MB, so depth=8-16 costs tens
    of MB, not gigabytes) but smooths over any per-batch fetch-latency
    variance beyond what depth=1 already covers, since a slow CPU-side batch
    doesn't stall the GPU as long as the queue ahead of it hasn't run dry.

    Yields tensors already resident on `device` — do not call `.to(device)`
    again on them in the training loop.
    """

    def __init__(self, loader, device: torch.device, depth: int = 1):
        self.loader = loader
        self.device = device
        self.depth = max(1, depth)
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __len__(self) -> int:
        return len(self.loader)

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

        for _ in range(self.depth):
            if not preload_one():
                break

        while pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            data, targets = pending.popleft()
            # Tell the caching allocator these tensors are still in use by the
            # side stream's copy until the default stream catches up, so it
            # can't reclaim/overwrite that memory early (required whenever a
            # tensor crosses streams like this — see PyTorch's CUDA stream docs).
            data.record_stream(torch.cuda.current_stream(self.device))
            targets.record_stream(torch.cuda.current_stream(self.device))
            preload_one()
            yield data, targets
