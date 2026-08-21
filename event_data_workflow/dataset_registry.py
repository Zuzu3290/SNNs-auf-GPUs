"""
Single source of truth for every dataset the pipeline can load: tonic class/
loader, sensor shape, class count, sample counts, and storage_size_gb
(compressed download size actually pulled by this entry's loader, not the
extracted/on-disk footprint — None where no figure has been measured or
documented; see docs/Event-Based_camera.md for sourcing).
"""
from __future__ import annotations
import logging
from pathlib import Path

import h5py
import numpy as np
import tonic
from tonic.download_utils import download_and_extract_archive
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# DVS128 Gesture's figshare.com/ndownloader URLs sit behind an AWS WAF bot
# challenge (confirmed: 202 Accepted, 0 bytes). ndownloader.figshare.com is
# the same files' canonical subdomain, no challenge.
tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"


class WindowedRecordingDataset(Dataset):
    """Flattens whole recordings (too large to use as one training sample)
    into (event_window, target) samples, one per target's (start_us, stop_us)
    window. Generic over any recording source via the three accessor args."""

    requires_single_process_loading = True

    def __init__(self, recordings: Dataset, get_events, get_targets, get_windows):
        self.recordings = recordings
        self.get_events = get_events
        self.get_targets = get_targets
        self.get_windows = get_windows

        self._index: list[tuple[int, int]] = []
        for rec_idx in range(len(recordings)):
            n_windows = len(get_windows(recordings[rec_idx]))
            self._index.extend((rec_idx, frame_idx) for frame_idx in range(n_windows))

        self._cached_rec_idx: int | None = None
        self._cached_events = None
        self._cached_targets = None
        self._cached_windows = None

    def __len__(self):
        return len(self._index)

    def _load(self, rec_idx: int):
        if self._cached_rec_idx != rec_idx:
            recording = self.recordings[rec_idx]
            self._cached_rec_idx = rec_idx
            self._cached_events = self.get_events(recording)
            self._cached_targets = self.get_targets(recording)
            self._cached_windows = self.get_windows(recording)
        return self._cached_events, self._cached_targets, self._cached_windows

    def __getitem__(self, idx: int):
        rec_idx, frame_idx = self._index[idx]
        events, targets, windows = self._load(rec_idx)
        start_us, stop_us = windows[frame_idx]
        window = events[(events["t"] >= start_us) & (events["t"] < stop_us)]
        return window, targets[frame_idx]


DSEC_RECORDINGS = [name for name, has_flow in tonic.datasets.DSEC.recordings["train"].items() if has_flow]  # full 18-recording optical-flow "train" set, ~134M events each


def load_dsec(save_to: str, split: str) -> WindowedRecordingDataset:
    """DSEC via tonic's own DSEC class, windowed by its optical-flow timestamps. `split` is unused (kept for call-site symmetry) -- always loads DSEC_RECORDINGS."""
    dsec = tonic.datasets.DSEC(
        save_to=save_to, split=DSEC_RECORDINGS, data_selection="events_left",
        target_selection=["optical_flow_forward_event", "optical_flow_forward_timestamps"],
    )
    return WindowedRecordingDataset(
        dsec,
        get_events=lambda rec: rec[0][0]["events_left"],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
    )


DAVIS_POSE_SENSOR_SIZE = (240, 180, 2)  # DAVIS240C
EVENT_XYTP_I64_DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])
DAVIS_POSE_SEQUENCES = [name for name in tonic.datasets.DAVISDATA.recordings if name != "calibration"]  # full Event-Camera-Dataset collection minus the groundtruth-less calibration recording


class DAVISPoseRecordings(Dataset):
    """One recording = one Event-Camera-Dataset sequence (Mueggler et al.,
    rpg.ifi.uzh.ch): events.txt (t_seconds x y p) + groundtruth.txt (t_seconds,
    position xyz, quaternion xyzw) from a DAVIS240C + mocap rig. Downloads/
    extracts each sequence on first use."""

    base_url = "https://rpg.ifi.uzh.ch/datasets/davis"
    sensor_size = DAVIS_POSE_SENSOR_SIZE

    def __init__(self, save_to: str, sequences: list[str]):
        self.root = Path(save_to) / "DAVISPose"
        self.sequences = sequences
        self.root.mkdir(parents=True, exist_ok=True)
        for name in sequences:
            seq_dir = self.root / name
            if not (seq_dir / "events.txt").exists():
                download_and_extract_archive(f"{self.base_url}/{name}.zip", str(seq_dir), filename=f"{name}.zip")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_dir = self.root / self.sequences[idx]

        raw_events = np.loadtxt(seq_dir / "events.txt")
        events = np.empty(len(raw_events), dtype=EVENT_XYTP_I64_DTYPE)
        events["t"] = (raw_events[:, 0] * 1e6).astype(np.int64)
        events["x"] = raw_events[:, 1].astype(np.int64)
        events["y"] = raw_events[:, 2].astype(np.int64)
        events["p"] = raw_events[:, 3].astype(np.int64)

        gt = np.loadtxt(seq_dir / "groundtruth.txt")
        gt_t_us = (gt[:, 0] * 1e6).astype(np.int64)
        poses = gt[:, 1:].astype(np.float32)  # (N, 7): x, y, z, qx, qy, qz, qw

        windows = np.stack([gt_t_us[:-1], gt_t_us[1:]], axis=1)
        return events, (poses[:-1], windows)


def load_davis_pose(save_to: str, split: str) -> WindowedRecordingDataset:
    """Camera 6-DOF pose (Mueggler et al., Event-Camera Dataset) via DAVISPoseRecordings."""
    recordings = DAVISPoseRecordings(save_to, sequences=DAVIS_POSE_SEQUENCES)
    return WindowedRecordingDataset(
        recordings,
        get_events=lambda rec: rec[0],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
    )


EYETRACKING_SENSOR_SIZE = (240, 180, 2)  # DAVIS240C, same sensor family as DAVIS Camera Pose
EYETRACKING_RECORDINGS = 1  # curated subset -- tonic's full "train" split is 16 recordings, millions of events each


class EyeTrackingRecordings(Dataset):
    """One recording = one 3ET-Eyetracking video: DVS events + per-frame (x, y) gaze position, windowed by that recording's own frame timestamps."""

    sensor_size = EYETRACKING_SENSOR_SIZE

    def __init__(self, save_to: str, split: str = "train"):
        base = tonic.datasets.ThreeET_Eyetracking(save_to=save_to, split=split)
        self.data, self.targets = base.data[:EYETRACKING_RECORDINGS], base.targets[:EYETRACKING_RECORDINGS]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        with h5py.File(self.data[idx], "r") as f:
            raw = f["events"][:]
        events = np.empty(len(raw), dtype=EVENT_XYTP_I64_DTYPE)
        events["t"], events["x"], events["y"], events["p"] = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

        gaze = np.loadtxt(self.targets[idx]).astype(np.float32)  # (N, 2): x, y gaze position per frame
        times_us = (np.loadtxt(self.data[idx].replace(".h5", "-frame_times.txt"), skiprows=2, usecols=1) * 1e6).astype(np.int64)

        n = min(len(gaze), len(times_us) - 1)
        windows = np.stack([times_us[:n], times_us[1:n + 1]], axis=1)
        return events, (gaze[:n], windows)


def load_eyetracking(save_to: str, split: str) -> WindowedRecordingDataset:
    """3ET-Eyetracking gaze-position regression via EyeTrackingRecordings."""
    recordings = EyeTrackingRecordings(save_to, split="train")
    return WindowedRecordingDataset(
        recordings,
        get_events=lambda rec: rec[0],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
    )


# Built-in dataset choices. Sample counts sourced from each dataset's own
# paper or tonic's _check_exists(). See dataset_registry_changes.md for the
# history behind entries that were added/removed/replaced.
DATASET_REGISTRY = {
    "1": {
        "name": "N-MNIST",
        "cls": tonic.datasets.NMNIST,
        "has_train_split": True,
        "sensor_size": tonic.datasets.NMNIST.sensor_size,
        "num_classes": 10,
        "num_train_samples": 60_000,
        "num_test_samples": 10_000,
        "storage_size_gb": 1.18,  # train.zip 965MB + test.zip 162MB
    },
    "2": {
        "name": "N-Caltech101",
        "cls": tonic.datasets.NCALTECH101,
        "has_train_split": False,
        "sensor_size": (240, 180, 2),
        "num_classes": 101,
        "num_train_samples": 6_967,
        "num_test_samples": 1_742,
        "storage_size_gb": 3.72,  # single zip, Mendeley-hosted
    },
    "3": {
        "name": "DAVIS Camera Pose",
        "kind": "regression",
        "loader": load_davis_pose,
        "sensor_size": DAVIS_POSE_SENSOR_SIZE,
        "num_classes": 1,
        "num_targets": 7,  # (x, y, z, qx, qy, qz, qw) -- real regression output width, for frameworks/regression/
        "num_train_samples": None,  # full DAVIS_POSE_SEQUENCES collection -- not measured until actually run
        "num_test_samples": None,
        "storage_size_gb": 7.7,  # full Event-Camera-Dataset collection (DAVIS_POSE_SEQUENCES, 24 sequences)
    },
    "4": {
        "name": "DVS128 Gesture",
        "cls": tonic.datasets.DVSGesture,
        "has_train_split": True,
        "sensor_size": tonic.datasets.DVSGesture.sensor_size,
        "num_classes": 11,
        "num_train_samples": 1_077,
        "num_test_samples": 264,
        "storage_size_gb": 3.0,  # compressed tar, train+test combined; ~5GB extracted
    },
    "5": {
        "name": "DSEC",
        "kind": "regression",
        "loader": load_dsec,
        "sensor_size": tonic.datasets.DSEC.sensor_size,
        "num_classes": 1,
        "num_targets": 2,  # (u, v) flow per pixel -- dense output, needs a decoder architecture, not a resized FC (frameworks/regression/)
        "num_train_samples": None,  # full DSEC_RECORDINGS optical-flow set -- not measured until actually run
        "num_test_samples": None,
        "storage_size_gb": None,  # full 18-recording optical-flow set, likely tens of GB -- not measured until actually run
    },
    "6": {
        "name": "Eye Tracking",
        "kind": "regression",
        "loader": load_eyetracking,
        "sensor_size": EYETRACKING_SENSOR_SIZE,
        "num_classes": 1,
        "num_targets": 2,  # (x, y) gaze position -- real regression output width, for frameworks/regression/
        "num_train_samples": 1_599,
        "num_test_samples": 400,
        "storage_size_gb": 3.87,  # whole-dataset zip (all subjects/videos); only EYETRACKING_RECORDINGS of them get used
    },
}


def resolve_dataset_entry(cfg) -> dict:
    """Match cfg.DATASET_NAME against DATASET_REGISTRY, else prompt interactively (works in a real terminal and in a live notebook kernel, both accept input() even though neither always reports a tty), else default to N-MNIST."""
    wanted = (cfg.DATASET_NAME or "").strip().upper()
    for entry in DATASET_REGISTRY.values():
        if entry["name"].upper() == wanted:
            return entry

    print("\n[PIPELINE] Select a dataset:")
    for key, entry in DATASET_REGISTRY.items():
        output = f"{entry['num_classes']} classes" if entry.get("kind", "classification") == "classification" else "regression target TBD"
        print(f"  {key}) {entry['name']}  [{output}]")
    try:
        choice = input("Enter number: ").strip()
    except EOFError:
        logger.warning("[PIPELINE] No interactive input source attached — defaulting to N-MNIST")
        return DATASET_REGISTRY["1"]

    if choice in DATASET_REGISTRY:
        return DATASET_REGISTRY[choice]
    logger.warning(f"[PIPELINE] Invalid selection '{choice}' — defaulting to N-MNIST")
    return DATASET_REGISTRY["1"]
