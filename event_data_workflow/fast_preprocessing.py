"""Numba-JIT-compiled replacements for tonic's two slowest per-sample preprocessing steps:
denoising (tonic.functional.denoise_numpy) and framing (tonic.functional.to_frame_numpy,
n_time_bins mode). Both loop over every raw event in pure Python; denoising alone was found
responsible for ~93% of N-Caltech101's real preprocessing cost, and framing was measured at
40-66ms/sample against denoising's ~10ms (diagnostics/probe_preprocessing_gil_release.py,
since deleted -- see docs/results/ToFrame_Acceleration_Case_Study.pdf for the full writeup).

Both kernels here run the same unchanged algorithm as tonic's reference, verified
byte-identical on real data, only compiled instead of interpreted: denoise_numpy_numba
(~72x faster) and to_frame_numba (~6.5-12.4x faster; to_frame_numba covers n_time_bins
framing only -- time_window mode is a different algorithm, not implemented here).

numba is a pinned dependency (requirements.txt), so it is imported directly -- no
availability fallback.
"""
import numpy as np
from numba import njit


def denoise_numpy_vectorized(events, filter_time: float = 10000):
    """Groups events by pixel, vectorized searchsorted per neighbor. Correct but measured
    ~3x SLOWER than denoise_numpy_numba below -- kept as a documented negative result,
    not deleted."""
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
def denoise_core_numba(x, y, t, filter_time, width, height):
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
    assert "x" in events.dtype.names and "y" in events.dtype.names and "t" in events.dtype.names
    if len(events) == 0:
        return events.copy()
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    t = events["t"].astype(np.float64)
    width = int(x.max()) + 1
    height = int(y.max()) + 1
    keep = denoise_core_numba(x, y, t, float(filter_time), width, height)
    return events[keep]


class FastDenoise:
    """Drop-in replacement for tonic.transforms.Denoise -- a picklable class, not a closure,
    for Windows' spawn-based multiprocessing (same reason FixedToFrame exists). ~72x faster,
    verified byte-identical output on real data."""

    def __init__(self, filter_time: float):
        self.filter_time = filter_time

    def __call__(self, events):
        return denoise_numpy_numba(events, filter_time=self.filter_time)


@njit(cache=True)
def searchsorted_left(t, target):
    """First index where t[idx] >= target, matching np.searchsorted(t, target, side='left')
    on a sorted array. Written out rather than calling np.searchsorted directly so the
    exact 'left' semantics are guaranteed regardless of numba version support."""
    lo, hi = 0, t.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if t[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def bin_boundaries_numba(t, n_time_bins):
    """T+1 boundary indices into the (sorted-by-time) events array -- boundary[i] is where
    bin i starts, boundary[T] is where the last bin ends. Matches
    SliceByTimeBins.get_slice_metadata with overlap=0 exactly: time_window = (t[-1]-t[0])
    // n_time_bins (integer floor division), stride == time_window, so consecutive bins
    tile the array with no gap and no overlap -- boundary[i+1] IS indices_end[i] IS
    indices_start[i+1]. Floor-division truncation means boundary[T] can be < len(t): a
    few trailing events past the last bin's end are silently dropped, matching the
    reference's own behavior exactly rather than "helpfully" keeping them."""
    n = t.shape[0]
    t0 = t[0]
    time_window = (t[n - 1] - t0) // n_time_bins
    boundaries = np.empty(n_time_bins + 1, dtype=np.int64)
    for i in range(n_time_bins + 1):
        boundaries[i] = searchsorted_left(t, t0 + i * time_window)
    return boundaries


@njit(cache=True)
def accumulate_numba(x, y, p, boundaries, n_time_bins, height, width, n_polarities):
    """One linear pass over events (only up to boundaries[-1] -- see the tail-drop note
    above), advancing the bin pointer as boundaries are crossed, incrementing
    frames[bin, p, y, x] per event -- the numba-compiled equivalent of tonic's per-slice
    np.add.at(frames, (i, p, y, x), 1)."""
    frames = np.zeros((n_time_bins, n_polarities, height, width), dtype=np.int16)
    end = boundaries[n_time_bins]
    bin_idx = 0
    for j in range(end):
        while bin_idx < n_time_bins - 1 and j >= boundaries[bin_idx + 1]:
            bin_idx += 1
        frames[bin_idx, p[j], y[j], x[j]] += 1
    return frames


def to_frame_numba(events, sensor_size, n_time_bins: int):
    """Drop-in for tonic.functional.to_frame_numpy(..., n_time_bins=n_time_bins) -- same
    (T, P, H, W) int16 output. Does not replicate the empty-events zero-fill branch;
    callers (FastToFrame below) must guard len(events) == 0 the same way
    tonic.transforms.ToFrame.__call__ does, before this is ever invoked."""
    width, height, n_polarities = sensor_size
    x = events["x"].astype(np.int64)
    y = events["y"].astype(np.int64)
    p = events["p"].astype(np.int64)
    t = events["t"].astype(np.int64)

    boundaries = bin_boundaries_numba(t, n_time_bins)
    return accumulate_numba(x, y, p, boundaries, n_time_bins, height, width, n_polarities)


class FastToFrame:
    """Drop-in replacement for tonic.transforms.ToFrame (n_time_bins mode only) --
    a picklable class, not a closure, for Windows' spawn-based multiprocessing (same
    reason FastDenoise/FixedToFrame are classes, not functions)."""

    def __init__(self, sensor_size, n_time_bins: int):
        self.sensor_size = sensor_size
        self.n_time_bins = n_time_bins

    def __call__(self, events):
        if len(events) == 0:
            w, h, p = self.sensor_size
            return np.zeros((self.n_time_bins, p, h, w), dtype=np.int16)
        return to_frame_numba(events, sensor_size=self.sensor_size, n_time_bins=self.n_time_bins)
