"""
Keeps a DataLoader one batch ahead of the training loop in a background thread.
"""
from __future__ import annotations
import queue
import threading


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
