"""
Phase B dataset loader — DSEC, the one dataset in event_data_workflow whose raw
structure does not fit tonic's plain (events, class_index) shape used by
data_pipeline.py's DATASET_REGISTRY.

Replaced MVSEC/TUM-VIE (removed — MVSEC had no real optical-flow ground truth
without a from-scratch loader, TUM-VIE's mocap only bracketed the start/end of
each recording, and both had external-hosting/size problems). DSEC has real
official optical-flow ground truth, disparity/depth targets. Scoped to optical
flow here — disparity has a different (instantaneous, not windowed) timestamp
structure and isn't wired.

tonic's own DSEC class hands back one whole recording (~134M events) as a
single sample — unusable directly, both as a cache/batch unit and because
there's no way to align that much raw event data with a single flow-frame
target. DSECRaw fixes this using DSEC's own optical_flow_forward_timestamps —
each flow frame already comes with an exact (start_us, stop_us) window in the
same absolute microsecond epoch as the event stream's own timestamps — so each
(event_sub_window, flow_frame) pair becomes one training sample, expanding one
recording into N samples instead of leaving it as one unusably large one.
"""
from __future__ import annotations

import numpy as np
import tonic
from torch.utils.data import Dataset


class DSECRaw(Dataset):
    """One sample = one DSEC optical-flow frame's exact time window, paired with
    the raw events falling inside it. data = windowed events_left structured
    array, target = the matching (H, W, 3) flow frame (channels: flow_x, flow_y,
    valid_mask — the model predicts the first two, the third is for loss masking,
    not something to predict)."""

    sensor_size = tonic.datasets.DSEC.sensor_size  # (640, 480, 2) — fixed, unlike MVSEC/N-Caltech101

    # h5py file handles (events) aren't safely shared across forked/spawned
    # DataLoader worker processes — force num_workers=0 for this dataset.
    requires_single_process_loading = True

    def __init__(self, save_to: str, split: str):
        self._dsec = tonic.datasets.DSEC(
            save_to=save_to,
            split=split,
            data_selection="events_left",
            target_selection=["optical_flow_forward_event", "optical_flow_forward_timestamps"],
        )

        # Flat (recording_idx, frame_idx) index across every recording, built once.
        # Reads each recording's target once during indexing (frame count comes
        # from the timestamps array) — a one-time O(num_recordings) cost, not
        # O(num_samples); acceptable since it happens once at startup, not per-batch.
        self._index: list[tuple[int, int]] = []
        for rec_idx in range(len(self._dsec)):
            _, target = self._dsec[rec_idx]
            n_frames = target[1].shape[0]
            self._index.extend((rec_idx, frame_idx) for frame_idx in range(n_frames))

        # Only one recording's raw events resident at a time — each is ~100M+
        # events, holding more than one in memory at once isn't reasonable.
        self._cached_rec_idx: int | None = None
        self._cached_events = None
        self._cached_flow = None
        self._cached_timestamps = None

    def __len__(self):
        return len(self._index)

    def _load_recording(self, rec_idx: int):
        if self._cached_rec_idx != rec_idx:
            data, target = self._dsec[rec_idx]
            self._cached_rec_idx = rec_idx
            self._cached_events = data[0]["events_left"]
            self._cached_flow = target[0]
            self._cached_timestamps = target[1]
        return self._cached_events, self._cached_flow, self._cached_timestamps

    def __getitem__(self, idx):
        rec_idx, frame_idx = self._index[idx]
        events, flow_frames, timestamps = self._load_recording(rec_idx)

        start_us, stop_us = timestamps[frame_idx]
        window = events[(events["t"] >= start_us) & (events["t"] < stop_us)]
        return window, flow_frames[frame_idx]
