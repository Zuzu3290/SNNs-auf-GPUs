"""
Phase B dataset loaders — datasets whose raw structure does not fit tonic's plain
(events, class_index) shape used by data_pipeline.py's DATASET_REGISTRY.

Each wrapper here is a minimal, faithful pass-through of what tonic's own dataset
class officially returns: the left-camera event stream as input, tonic's official
target structure as output, unmodified. No derived targets (net displacement,
windowing, resampling) are built yet — those come only after inspecting what the
official structure actually contains on real downloaded data.
"""
from __future__ import annotations

import tonic
from torch.utils.data import Dataset


class TUMVIERaw(Dataset):
    """Wraps tonic.datasets.TUMVIE. data = events_left, target = tonic's official
    {"images_left", "images_right", "mocap"} dict for that recording, unmodified."""

    sensor_size = tonic.datasets.TUMVIE.sensor_size

    # h5py file handles (events) aren't safely shared across forked/spawned
    # DataLoader worker processes — force num_workers=0 for this dataset.
    requires_single_process_loading = True

    def __init__(self, save_to: str, recording: str = "mocap-1d-trans"):
        self._tumvie = tonic.datasets.TUMVIE(save_to=save_to, recording=recording)

    def __len__(self):
        return len(self._tumvie)

    def __getitem__(self, idx):
        data, targets = self._tumvie[idx]
        return data["events_left"], targets


class MVSECRaw(Dataset):
    """Wraps tonic.datasets.MVSEC. data = events_left, target = tonic's official
    (depth_rect_left, depth_rect_right, pose) tuple for that recording, unmodified."""

    # DAVIS346 sensor nominal resolution — MVSEC's own class doesn't declare
    # sensor_size at all (unlike TUMVIE/NMNIST/etc). Worth confirming empirically
    # against a real downloaded recording before trusting this blindly.
    sensor_size = (346, 260, 2)

    # rosbag file handles (events) aren't safely shared across forked/spawned
    # DataLoader worker processes — force num_workers=0 for this dataset.
    requires_single_process_loading = True

    def __init__(self, save_to: str, scene: str = "indoor_flying"):
        self._mvsec = tonic.datasets.MVSEC(save_to=save_to, scene=scene)

    def __len__(self):
        return len(self._mvsec)

    def __getitem__(self, idx):
        data, targets = self._mvsec[idx]
        events_left, *_ = data
        return events_left, targets
