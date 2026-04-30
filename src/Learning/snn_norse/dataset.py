# =============================================================
# dataset.py — N-MNIST event dataset loader
# =============================================================
# N-MNIST is the MNIST digit dataset recorded with a DVS
# (Dynamic Vision Sensor) event camera.
#
# Raw data format per sample:
#   stream of events: (x, y, timestamp, polarity)
#   x, y      → pixel location on 34×34 sensor
#   timestamp → microsecond precision
#   polarity  → +1 (pixel got brighter) or -1 (pixel got darker)
#
# What this file does:
#   1. Downloads N-MNIST automatically on first run
#   2. Bins all events into T equal time windows via ToFrame
#   3. Each window becomes a 2×34×34 frame (one channel per polarity)
#   4. Caches transformed tensors to disk so epoch 2+ loads instantly
#   5. Returns PyTorch DataLoaders ready for training

import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(config: dict):
    """
    Returns (train_loader, test_loader) for N-MNIST.

    Output tensor shape per sample: (T, 2, 34, 34)
      T     = config["timesteps"]
      2     = polarity channels (+ and -)
      34×34 = N-MNIST sensor resolution
    """
    T        = config["timesteps"]
    data_dir = config["data_dir"]

    # ── Transform: bin raw events into T time windows ─────────
    # ToFrame slices the full event recording into T equal windows.
    #
    # Example with T=16 on a 300ms recording:
    #   Window 0 : events from   0ms – 18.75ms → 2×34×34 frame
    #   Window 1 : events from  18.75ms – 37.5ms → 2×34×34 frame
    #   ...
    #   Window 15: last 18.75ms                 → 2×34×34 frame
    #
    # Each pixel that had at least one event in a window gets a
    # count value (how many times it fired). Pixels with no event
    # stay 0. Values > 1 are valid — they mean the pixel fired
    # multiple times within that window (more activity = larger input
    # to the LIF neuron, which builds voltage faster).
    frame_transform = transforms.ToFrame(
        sensor_size=tonic.datasets.NMNIST.sensor_size,  # (34, 34, 2)
        n_time_bins=T,
    )

    # ── Load raw N-MNIST ──────────────────────────────────────
    # Downloads ~1GB to data_dir on first run.
    train_raw = tonic.datasets.NMNIST(
        save_to=data_dir, train=True,  transform=frame_transform
    )
    test_raw = tonic.datasets.NMNIST(
        save_to=data_dir, train=False, transform=frame_transform
    )

    # ── Disk cache ────────────────────────────────────────────
    # DiskCachedDataset runs ToFrame on first access per sample
    # and saves the resulting tensor to disk.
    # From epoch 2 onwards, tensors are loaded directly — no
    # re-processing of raw events. Speeds up training significantly.
    #
    # Cache is stored in data_dir/cache_train/ and cache_test/
    # It is safe to delete these folders to reset the cache.
    train_dataset = tonic.DiskCachedDataset(
        train_raw,
        cache_path=f"{data_dir}/cache_nmnist_T{T}_train",
    )
    test_dataset = tonic.DiskCachedDataset(
        test_raw,
        cache_path=f"{data_dir}/cache_nmnist_T{T}_test",
    )

    # ── Collate: numpy arrays → float32 tensors ───────────────
    # Tonic returns numpy arrays per sample.
    # This converts them into a batched float32 tensor for PyTorch.
    def collate(batch):
        samples, labels = zip(*batch)
        x = torch.stack([torch.tensor(s, dtype=torch.float32) for s in samples])
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    print(f"Dataset      : N-MNIST (event camera, DVS recorded)")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")
    print(f"Input shape  : (T={T}, 2 polarities, 34×34 pixels)")
    print(f"Batch shape  : ({config['batch_size']}, {T}, 2, 34, 34)")
    print(f"Disk cache   : {data_dir}/cache_nmnist_T{T}_*/")

    return train_loader, test_loader
