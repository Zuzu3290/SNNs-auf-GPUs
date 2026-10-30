"""Every dataset the pipeline can load: its tonic class (or loader), sensor shape and class count."""
from __future__ import annotations
import difflib
import logging
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import tonic
from tonic.download_utils import download_and_extract_archive
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

# figshare.com/ndownloader sits behind a bot challenge; the ndownloader subdomain serves the same files without one.
tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"


class WindowedRecordingDataset(Dataset):
    """Splits long recordings into (event_window, target) samples, one per target's (start_us, stop_us) window."""

    requires_single_process_loading = True

    def __init__(self, recordings: Dataset, get_events, get_targets, get_windows,
                  count_windows=None):
        """`count_windows(rec_idx)` counts a recording's windows without loading its events, which can take minutes."""
        self.recordings = recordings
        self.get_events = get_events
        self.get_targets = get_targets
        self.get_windows = get_windows

        self._index: list[tuple[int, int]] = []
        for rec_idx in range(len(recordings)):
            n_windows = (count_windows(rec_idx) if count_windows is not None
                         else len(get_windows(recordings[rec_idx])))
            self._index.extend((rec_idx, frame_idx) for frame_idx in range(n_windows))

        self._cached_rec_idx = None
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


DAVIS_POSE_SENSOR_SIZE = (240, 180, 2)  # DAVIS240C
EVENT_XYTP_I64_DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])
# One sequence per scene under full 6-DOF motion; the other sequences repeat the same scenes, and these give ~36,000 windows.
DAVIS_POSE_SEQUENCES = ["boxes_6dof", "shapes_6dof", "poster_6dof"]


class DAVISPoseRecordings(Dataset):
    """Event-Camera Dataset sequences (Mueggler et al.): events.txt plus mocap groundtruth.txt, downloaded on first use."""

    base_url = "https://rpg.ifi.uzh.ch/datasets/davis"
    sensor_size = DAVIS_POSE_SENSOR_SIZE

    def __init__(self, save_to: str, sequences: list[str]):
        self.root = Path(save_to) / "DAVISPose"
        self.sequences = sequences
        self.root.mkdir(parents=True, exist_ok=True)
        for name in sequences:
            if self.find_sequence_dir(name) is None:
                download_and_extract_archive(f"{self.base_url}/{name}.zip",
                                              str(self.root / name), filename=f"{name}.zip")

    def find_sequence_dir(self, name: str):
        """The folder holding this sequence's events.txt (archives nest one level deep), or None if not downloaded."""
        for candidate in (self.root / name / name, self.root / name):
            if (candidate / "events.txt").is_file():
                return candidate
        return None

    def sequence_dir(self, name: str) -> Path:
        found = self.find_sequence_dir(name)
        if found is None:
            raise FileNotFoundError(
                f"{name}: no events.txt under {self.root / name} or {self.root / name / name}. "
                "The download or extraction did not complete."
            )
        return found

    def window_count(self, idx: int) -> int:
        """Window count from the small groundtruth.txt alone, without parsing the multi-GB events.txt."""
        gt = np.loadtxt(self.sequence_dir(self.sequences[idx]) / "groundtruth.txt")
        return max(0, len(gt) - 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_dir = self.sequence_dir(self.sequences[idx])

        # pandas' C parser reads the multi-GB events.txt several times faster than np.loadtxt.
        raw_events = pd.read_csv(seq_dir / "events.txt", sep=r"\s+", header=None,
                                  dtype=np.float64).to_numpy()
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
    """Camera 6-DOF pose windows; `split` is unused."""
    recordings = DAVISPoseRecordings(save_to, sequences=DAVIS_POSE_SEQUENCES)
    return WindowedRecordingDataset(
        recordings,
        get_events=lambda rec: rec[0],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
        count_windows=recordings.window_count,
    )


EYETRACKING_SENSOR_SIZE = (240, 180, 2)  # DAVIS240C
EYETRACKING_RECORDINGS = 1  # tonic's full train split is 16 recordings of millions of events each


class EyeTrackingRecordings(Dataset):
    """3ET-Eyetracking videos: DVS events plus per-frame (x, y) gaze, windowed by the recording's frame timestamps."""

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
    """Gaze-position windows; `split` is unused."""
    recordings = EyeTrackingRecordings(save_to, split="train")
    return WindowedRecordingDataset(
        recordings,
        get_events=lambda rec: rec[0],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
    )


# 1-3 train end to end; 4-5 are loaders only, kept for anyone who wants to add a regression path (they have no num_classes).
DATASET_REGISTRY = {
    "1": {
        "name": "N-MNIST",
        "cls": tonic.datasets.NMNIST,
        "sensor_size": tonic.datasets.NMNIST.sensor_size,
        "num_classes": 10,
    },
    "2": {
        "name": "N-Caltech101",
        "cls": tonic.datasets.NCALTECH101,
        "sensor_size": (240, 180, 2),
        "num_classes": 101,
    },
    "3": {
        "name": "DVS128 Gesture",
        "cls": tonic.datasets.DVSGesture,
        "sensor_size": tonic.datasets.DVSGesture.sensor_size,
        "num_classes": 11,
    },
    "4": {
        "name": "DAVIS Camera Pose",
        "loader": load_davis_pose,
        "sensor_size": DAVIS_POSE_SENSOR_SIZE,
        "num_targets": 7,  # x, y, z, qx, qy, qz, qw
    },
    "5": {
        "name": "Eye Tracking",
        "loader": load_eyetracking,
        "sensor_size": EYETRACKING_SENSOR_SIZE,
        "num_targets": 2,  # x, y gaze position
    },
}


class UnknownDataset(Exception):
    """dataset.name names something that is not in DATASET_REGISTRY."""


def normalise_dataset_name(name: str) -> str:
    """Fold case, spaces, hyphens and underscores so 'DVS128 Gesture' and 'dvs128-gesture' match."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def dataset_menu() -> str:
    lines = []
    for key, entry in DATASET_REGISTRY.items():
        output = (f"{entry['num_classes']} classes" if "num_classes" in entry
                  else f"{entry['num_targets']} targets, loader only")
        lines.append(f"  {key}) {entry['name']}  [{output}]")
    return "\n".join(lines)


def lookup_dataset(wanted: str) -> dict:
    """One dataset by name or menu number; anything unrecognised raises with the closest matches."""
    text = str(wanted).strip()
    if text in DATASET_REGISTRY:
        return DATASET_REGISTRY[text]

    target = normalise_dataset_name(text)
    for entry in DATASET_REGISTRY.values():
        if normalise_dataset_name(entry["name"]) == target:
            return entry

    names = [entry["name"] for entry in DATASET_REGISTRY.values()]
    close = difflib.get_close_matches(text, names, n=3, cutoff=0.5)
    if not close:  # retry on the folded forms, so 'nmnist' still suggests N-MNIST
        folded = {normalise_dataset_name(n): n for n in names}
        close = [folded[m] for m in difflib.get_close_matches(target, list(folded), n=3, cutoff=0.5)]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise UnknownDataset(
        f"dataset.name = {text!r} is not a known dataset.{hint}\n"
        f"Available (name, or the number in brackets):\n{dataset_menu()}"
    )


def resolve_dataset_entry(cfg) -> dict:
    """The dataset named by dataset.name, or asked for at a prompt when it is null."""
    wanted = getattr(cfg, "DATASET_NAME", None)
    if wanted is not None and str(wanted).strip():
        entry = lookup_dataset(wanted)
        logger.info(f"[PIPELINE] Dataset from config: {entry['name']}")
        return entry

    print("\n[PIPELINE] Select a dataset:")
    print(dataset_menu())
    print("  (set dataset.name in the config to skip this prompt)")
    return lookup_dataset(input("Enter number: ").strip())
