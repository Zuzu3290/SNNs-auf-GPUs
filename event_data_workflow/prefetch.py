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
    never makes the training loop wait."""

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

        def produce():
            try:
                for batch in self.loader:
                    if stop_event.is_set():
                        return
                    while not stop_event.is_set():
                        try:
                            buf.put(batch, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                errors.append(exc)
            finally:
                try:
                    buf.put_nowait(sentinel)
                except queue.Full:
                    pass

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
    """Moves the next `depth` batches onto the GPU ahead of time, on their
    own CUDA stream, so the GPU is never left waiting for that transfer.
    Batches it yields are already on the GPU — no need to move them again."""

    def __init__(self, loader, device: torch.device, depth: int = 1):
        self.loader = loader
        self.device = device
        self.depth = max(1, depth)
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
