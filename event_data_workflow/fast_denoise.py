"""
Fast replacements for tonic.functional.denoise_numpy, which loops over every raw event in pure
Python and was found responsible for ~93% of N-Caltech101's real preprocessing cost.

Two candidates, both verified byte-identical to tonic's original on real data
(diagnostics/validate_fast_denoise.py), including exact same-timestamp tie-breaking (ties
resolve by original array position, not time value -- matching tonic's serial loop):
  - denoise_numpy_vectorized(): groups by pixel, vectorized searchsorted per neighbor. Correct
    but measured ~3x SLOWER (diagnostics/benchmark_fast_denoise.py) -- kept as a documented
    negative result, not deleted.
  - denoise_numpy_numba(): the same unchanged algorithm, JIT-compiled. ~72x faster, this is
    what FastDenoise (used by data_pipeline.py) actually calls.

Falls back to tonic's own denoise_numpy if numba is unavailable (it has a numpy version
ceiling) -- correct but slower, never a crash.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError as e:
    _NUMBA_AVAILABLE = False
    logger.warning(
        f"[FAST_DENOISE] numba unavailable ({e}) -- falling back to tonic's original "
        "denoise_numpy (correct, without the ~72x speedup). Run `pip install -r "
        "requirements.txt` (numpy<=2.4, numba>=0.66.0) to restore it."
    )

    def njit(*args, **kwargs):
        """No-op stand-in so the @njit decorator below doesn't raise when numba is missing."""
        def decorator(fn):
            return fn
        return decorator


def denoise_numpy_vectorized(events, filter_time: float = 10000):
    assert "x" in events.dtype.names and "y" in events.dtype.names and "t" in events.dtype.names
    n = len(events)
    if n == 0:
        return events.copy()

    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)

    width = int(x.max()) + 1
    height = int(y.max()) + 1
    pixel_id = x * height + y

    # Stable sort preserves original array order within each pixel group -- needed for
    # same-timestamp tie-breaking to match tonic's serial loop exactly.
    order = np.argsort(pixel_id, kind="stable")
    pixel_grouped = pixel_id[order]
    idx_grouped = order            # original array indices, ascending within each pixel group
    t_grouped = t[order]

    unique_pixels, group_start = np.unique(pixel_grouped, return_index=True)
    group_end = np.append(group_start[1:], n)
    pixel_indices = {int(p): idx_grouped[s:e] for p, s, e in zip(unique_pixels, group_start, group_end)}
    pixel_own_times = {int(p): t_grouped[s:e] for p, s, e in zip(unique_pixels, group_start, group_end)}
    empty_idx = np.empty(0, dtype=np.int64)
    empty_t = np.empty(0, dtype=np.float64)

    keep = np.zeros(n, dtype=bool)

    # Loop over unique active pixels, not raw events -- one vectorized searchsorted
    # call per neighbor direction covers every event at that pixel at once.
    for p, s, e in zip(unique_pixels, group_start, group_end):
        px, py = int(p) // height, int(p) % height
        my_idx = idx_grouped[s:e]
        my_t = t_grouped[s:e]
        satisfied = np.zeros(len(my_idx), dtype=bool)

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = px + dx, py + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue  # off-sensor neighbor -- original code's x>0/x<width-1/etc. guards skip these too
            n_idx = pixel_indices.get(nx * height + ny, empty_idx)
            n_t = pixel_own_times.get(nx * height + ny, empty_t)
            if len(n_idx) == 0:
                # A neighbor that never fires is equivalent to prev_time=0 (matches the
                # original's timestamp_memory initialization at filter_time everywhere).
                prev_time = np.zeros(len(my_idx))
            else:
                # Search by original array index, not time, for exact tie-breaking.
                pos = np.searchsorted(n_idx, my_idx, side="left") - 1
                has_prior = pos >= 0
                prev_time = np.where(has_prior, n_t[np.clip(pos, 0, len(n_t) - 1)], 0.0)
            satisfied |= (my_t - prev_time) < filter_time

        keep[my_idx] = satisfied

    kept_indices = np.sort(np.nonzero(keep)[0])
    return events[kept_indices]


@njit(cache=True)
def _denoise_core_numba(x, y, t, filter_time, width, height):
    """Same control flow as tonic.functional.denoise_numpy's Python loop, unchanged --
    only the execution engine differs (compiled, not interpreted)."""
    n = x.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    timestamp_memory = np.zeros((width, height), dtype=np.float64) + filter_time
    for i in range(n):
        xi = x[i]
        yi = y[i]
        ti = t[i]
        timestamp_memory[xi, yi] = ti + filter_time
        if (
            (xi > 0 and timestamp_memory[xi - 1, yi] > ti)
            or (xi < width - 1 and timestamp_memory[xi + 1, yi] > ti)
            or (yi > 0 and timestamp_memory[xi, yi - 1] > ti)
            or (yi < height - 1 and timestamp_memory[xi, yi + 1] > ti)
        ):
            keep[i] = True
    return keep


def denoise_numpy_numba(events, filter_time: float = 10000):
    if not _NUMBA_AVAILABLE:
        # Fall all the way back to tonic's own implementation -- correct by definition (it's
        # the reference), just without the JIT speedup. Never a crash, never wrong output.
        from tonic.functional import denoise_numpy
        return denoise_numpy(events, filter_time=filter_time)

    assert "x" in events.dtype.names and "y" in events.dtype.names and "t" in events.dtype.names
    if len(events) == 0:
        return events.copy()
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    width = int(x.max()) + 1
    height = int(y.max()) + 1
    keep = _denoise_core_numba(x, y, t, float(filter_time), width, height)
    return events[keep]


class FastDenoise:
    """Drop-in replacement for tonic.transforms.Denoise -- a picklable class, not a closure,
    for Windows' spawn-based multiprocessing (same reason FixedToFrame exists). ~72x faster,
    verified byte-identical output (diagnostics/validate_fast_denoise.py)."""

    def __init__(self, filter_time: float):
        self.filter_time = filter_time

    def __call__(self, events):
        return denoise_numpy_numba(events, filter_time=self.filter_time)
