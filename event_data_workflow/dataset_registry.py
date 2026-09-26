"""
Single source of truth for every dataset the pipeline can load: tonic class/
loader, sensor shape, class count, sample counts, and storage_size_gb
(compressed download size actually pulled by this entry's loader, not the
extracted/on-disk footprint — None where no figure has been measured or
documented; see docs/Event-Based_camera.md for sourcing).
"""
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

    def __init__(self, recordings: Dataset, get_events, get_targets, get_windows,
                  count_windows=None):
        """`count_windows(rec_idx)` returns a recording's window count WITHOUT
        materialising the recording.

        It matters more than it looks. Building the index needs only how many windows
        each recording has, but reaching that through `get_windows(recordings[i])`
        loads the whole recording -- and for DAVIS Camera Pose one recording is a 4 GB
        events.txt that takes about three minutes to parse. Over 24 sequences that is
        roughly 72 minutes spent before the first sample is ever read, to obtain 24
        integers that live in a 1.5 MB groundtruth file. Sources that can answer
        cheaply pass this; those that cannot leave it None and the slow path stands.
        """
        self.recordings = recordings
        self.get_events = get_events
        self.get_targets = get_targets
        self.get_windows = get_windows

        self._index: list[tuple[int, int]] = []
        for rec_idx in range(len(recordings)):
            n_windows = (count_windows(rec_idx) if count_windows is not None
                         else len(get_windows(recordings[rec_idx])))
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




DAVIS_POSE_SENSOR_SIZE = (240, 180, 2)  # DAVIS240C
EVENT_XYTP_I64_DTYPE = np.dtype([("x", np.int64), ("y", np.int64), ("t", np.int64), ("p", np.int64)])
# Sequences carrying NO groundtruth.txt. The Event-Camera Dataset was captured with a
# motion-capture rig for the indoor scenes only; these five were recorded handheld or
# outdoors where no mocap was available, so they ship events and images but no pose.
# VERIFIED against the extracted data: each contains calib.txt, events.txt, images.txt
# and nothing else. They cannot be used for pose regression and are excluded here rather
# than downloaded and discovered empty -- together they are about 1.1 GB of wasted
# download and several GB of wasted extraction.
DAVIS_POSE_NO_GROUNDTRUTH = ("calibration", "office_spiral", "office_zigzag",
                              "outdoors_running", "outdoors_walking", "urban")
# Three of the nineteen mocap sequences, not all of them. The nineteen are the same few
# scenes shot under different motions -- boxes, shapes, poster and slider, each repeated
# for rotation, translation and full 6-DOF. Taking all of them feeds the network the same
# scene content many times over, which trains it to those scenes rather than to pose.
# One of each SCENE under full 6-DOF motion is what there is to learn from: cluttered
# boxes, plain geometric shapes, and dense poster texture.
#
# It is also what makes this runnable: all nineteen window out to 168,516 samples, about
# 24 times N-Caltech101's iteration count. These three give roughly 36,000.
DAVIS_POSE_SEQUENCES = ["boxes_6dof", "shapes_6dof", "poster_6dof"]


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
            if self.find_sequence_dir(name) is None:
                download_and_extract_archive(f"{self.base_url}/{name}.zip",
                                              str(self.root / name), filename=f"{name}.zip")

    def find_sequence_dir(self, name: str) -> Path | None:
        """Where this sequence's files actually landed, or None if it is not downloaded.

        Each archive carries its own top-level folder, so extracting boxes_6dof.zip into
        .../DAVISPose/boxes_6dof produces .../DAVISPose/boxes_6dof/boxes_6dof/events.txt
        -- one level deeper than the path it was extracted to. Both layouts are accepted
        so an already-extracted copy is never re-downloaded because of the extra level.
        """
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
        """How many windows this sequence yields, read from groundtruth.txt alone.

        groundtruth.txt is about 1.5 MB; events.txt for the same sequence is about 4 GB
        and takes roughly three minutes to parse. Indexing needs only this count, so it
        must not touch the events -- see WindowedRecordingDataset's count_windows.
        """
        gt = np.loadtxt(self.sequence_dir(self.sequences[idx]) / "groundtruth.txt")
        return max(0, len(gt) - 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_dir = self.sequence_dir(self.sequences[idx])

        # pandas rather than np.loadtxt: these files reach 4 GB of plain text, where
        # numpy's pure-Python parser takes about three minutes. read_csv's C parser does
        # the same work several times faster, and this runs once per sequence per cache
        # build.
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
    """Camera 6-DOF pose (Mueggler et al., Event-Camera Dataset) via DAVISPoseRecordings. `split` is unused (kept for call-site symmetry)."""
    recordings = DAVISPoseRecordings(save_to, sequences=DAVIS_POSE_SEQUENCES)
    return WindowedRecordingDataset(
        recordings,
        get_events=lambda rec: rec[0],
        get_targets=lambda rec: rec[1][0],
        get_windows=lambda rec: rec[1][1],
        count_windows=recordings.window_count,
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
    """3ET-Eyetracking gaze-position regression via EyeTrackingRecordings. `split` is unused (kept for call-site symmetry)."""
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
        # MEASURED by HTTP HEAD on all 24 sequence zips, 2026-09-16. This is the
        # COMPRESSED download; the archives hold events.txt, which expands several
        # times over on extraction, so plan for considerably more free disk.
        "storage_size_gb": 8.0,
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


class UnknownDataset(Exception):
    """dataset.name names something that is not in DATASET_REGISTRY."""


def normalise_dataset_name(name: str) -> str:
    """Fold case, spaces, hyphens and underscores so 'DVS128 Gesture',
    'dvs128-gesture' and 'dvs128_gesture' all name the same dataset."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def dataset_menu() -> str:
    lines = []
    for key, entry in DATASET_REGISTRY.items():
        kind = entry.get("kind", "classification")
        output = (f"{entry['num_classes']} classes" if kind in ("classification", "detection")
                  else "regression target TBD")
        lines.append(f"  {key}) {entry['name']}  [{output}]")
    return "\n".join(lines)


def lookup_dataset(wanted: str) -> dict:
    """Find one dataset by name or by registry key ('1'..'8').

    Raises UnknownDataset on anything unrecognised, naming the closest matches. A typo
    must fail HERE, at startup, rather than being silently ignored -- a run that
    quietly used the wrong dataset, or fell back to N-MNIST while the config asked for
    DVS128 Gesture, is worse than one that refuses to start.
    """
    text = str(wanted).strip()
    if text in DATASET_REGISTRY:            # a registry key, as the menu prints them
        return DATASET_REGISTRY[text]

    target = normalise_dataset_name(text)
    for entry in DATASET_REGISTRY.values():
        if normalise_dataset_name(entry["name"]) == target:
            return entry

    names = [entry["name"] for entry in DATASET_REGISTRY.values()]
    close = difflib.get_close_matches(text, names, n=3, cutoff=0.5)
    if not close:  # try again on the normalised forms, so 'nmnist' still suggests N-MNIST
        folded = {normalise_dataset_name(n): n for n in names}
        close = [folded[m] for m in difflib.get_close_matches(target, list(folded), n=3, cutoff=0.5)]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise UnknownDataset(
        f"dataset.name = {text!r} is not a known dataset.{hint}\n"
        f"Available (name, or the number in brackets):\n{dataset_menu()}"
    )


def resolve_dataset_entry(cfg) -> dict:
    """Which dataset this run uses.

    Two paths, both supported:

      dataset.name set in config  -> used directly, no prompt. Required for anything
                                     non-interactive: a Colab cell, a scripted
                                     multi-seed sweep, or several runs launched at
                                     once cannot answer a prompt, and a run that
                                     needed one is not reproducible from its config.
      dataset.name absent/null    -> prompt, exactly as this pipeline always has.

    A name that is set but unrecognised RAISES rather than falling back -- see
    lookup_dataset.
    """
    wanted = getattr(cfg, "DATASET_NAME", None)
    if wanted is not None and str(wanted).strip():
        entry = lookup_dataset(wanted)
        logger.info(f"[PIPELINE] Dataset from config: {entry['name']}")
        return entry

    print("\n[PIPELINE] Select a dataset:")
    print(dataset_menu())
    print("  (set dataset.name in the config to skip this prompt)")
    try:
        choice = input("Enter number: ").strip()
    except EOFError:
        logger.warning("[PIPELINE] No interactive input source attached — defaulting to N-MNIST. "
                       "Set dataset.name in the config to choose deliberately.")
        return DATASET_REGISTRY["1"]

    try:
        return lookup_dataset(choice)
    except UnknownDataset:
        logger.warning(f"[PIPELINE] Invalid selection '{choice}' — defaulting to N-MNIST")
        return DATASET_REGISTRY["1"]
