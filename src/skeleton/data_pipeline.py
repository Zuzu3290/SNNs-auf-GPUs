"""
data_pipeline.py
----------------
Universal data processing pipeline for Spiking Neural Networks.
Two Encoder Classes:
  • DigitalDataEncoder   — For static image datasets (MNIST, FashionMNIST, CIFAR-10, etc.)
  • NeuromorphicEncoder  — For event-based datasets (N-MNIST, DVS, etc.)


"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as tonic_transforms
import snntorch.spikegen as spikegen
import numpy as np
from skeleton import Settings
from typing import Tuple
import matplotlib.pyplot as plt
from __future__ import annotations


class DigitalDatasetEncoder:
    """
    Encoder for static image datasets (MNIST, FashionMNIST, CIFAR-10, etc.).
 
    Handles:
      - DataLoader wrapping (no caching for digital data—already fast)
      - Returns raw batch DataLoaders: (data [B, C, H, W], targets [B])
    
    Note: Dataset construction is handled externally. This class only wraps
    pre-constructed datasets into DataLoaders.
    """
    def __init__(self, train_ds: Dataset, test_ds: Dataset, cfg: Settings):
        self.cfg = cfg
        self.train_loader = None
        self.test_loader = None
        self.build(train_ds, test_ds)

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)),
            transforms.Grayscale(num_output_channels=self.cfg.IMAGE_CHANNELS),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
    
    def build(self, train_ds: Dataset, test_ds: Dataset) -> None:
        """Wrap datasets into DataLoaders."""
        # Build dataloaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
 
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )

        return self.train_loader, self.test_loader

    def print_sample(self) -> None:
        """Fetch and display one sample from the training set."""
        data, target = next(iter(self.train_loader))
        sample = data[0]  # (C, H, W)
        img = sample.numpy()
 
        if img.shape[0] == 1:
            img = img[0]  # (H, W) for grayscale
        else:
            img = np.transpose(img, (1, 2, 0))  # (H, W, C)
 
        print(f"  Dataset: {self.cfg.DATASET_NAME}")
        print(f"  Type: Digital (static images)")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Target: {target[0].item()}")
 
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(f"Label: {target[0].item()}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    

class NeuromorphicEncoder:
    """
    Encoder for event-based neuromorphic datasets (N-MNIST, DVS, etc.).
 
    Handles:
      - DataLoader wrapping with optional disk caching
      - PadTensors collation for variable-length sequences
      - Returns raw batch DataLoaders: (data [T, B, C, H, W], targets [B])
        where T is variable (padded to longest in batch)
 
    Notes:
      - Data is already temporally structured (no encoding needed)
      - Dataset construction is handled externally
      - Caching significantly speeds up repeated dataloading
    """
 
    def __init__(self, train_ds: Dataset, test_ds: Dataset, cfg: Settings, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self.train_loader = None
        self.test_loader = None
        self.build(train_ds, test_ds)

    def _build(self, train_ds: Dataset, test_ds: Dataset) -> None:
        if self.use_cache:
            train_ds = DiskCachedDataset(
                train_ds,
                cache_path=f"{self.cfg.DATA_PATH}/cache/train"
            )
            test_ds = DiskCachedDataset(
                test_ds,
                cache_path=f"{self.cfg.DATA_PATH}/cache/test"
            )
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.BATCH_SIZE,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=True,
        )
 
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.cfg.BATCH_SIZE,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
        )

        return self.train_loader, self.test_loader

    def print_sample(self) -> None:
        """Fetch and display one sample from the training set."""
        data, targets = next(iter(self.train_loader))
        sample = data[:, 0]  # (T, C, H, W) — first sample in batch
 
        # Sum over time and polarity channels to visualize
        if sample.ndim == 4:
            frame = sample.numpy().sum(axis=0).sum(axis=0)  # (H, W)
        else:
            frame = sample.numpy().sum(axis=0)  # (H, W)
 
        print(f"  Dataset: {self.cfg.DATASET_NAME}")
        print(f"  Type: Neuromorphic (event-based)")
        print(f"  Sample shape (T, C, H, W): {sample.shape}")
        print(f"  Target: {targets[0].item()}")
        print(f"  Time steps (padded): {data.shape[0]}")
 
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frame, cmap="inferno")
        ax.set_title(f"Label: {targets[0].item()} (Event sum)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

from data_pipeline import DigitalDataEncoder, NeuromorphicEncoder

def build_dataloaders(config):
    if config.DATASET_TYPE.lower() == "digital":
        encoder = DigitalDatasetEncoder(config)

    elif config.DATASET_TYPE.lower() == "event":
        encoder = NeuromorphicEncoder(config)

    else:
        raise ValueError("DATASET_TYPE must be either 'digital' or 'event'")

    return encoder.load_dataset()

train_loader, test_loader = build_dataloaders(config)


# Receive raw events 
# (x,y,t,p)
# Group them into a short time window.
# Convert them into one of these:
# event voxel grid,
# event frame,
# time surface,
# direct spike input.
# Feed that into the SNN.
#Resume this study at https://www.perplexity.ai/search/76c72946-8644-487d-8c15-fd6d6bb00123

def main():
    cfg = Settings()

    if cfg.DATASET_TYPE.lower() == "digital":
        print("Initializing DigitalDataEncoder…")
        encoder = DigitalDataEncoder(cfg)
    elif cfg.DATASET_TYPE.lower() == "event":
        print("Initializing NeuromorphicEncoder…")
        encoder = NeuromorphicEncoder(cfg, use_cache=True)
    else:
        raise ValueError(f"Unknown DATASET_TYPE: {cfg.DATASET_TYPE}")
 
    # Get dataloaders
    train_loader, test_loader = encoder.get_dataloaders()
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = main()
    print("\n✓ Dataloaders ready for training/testing.")
    print("  - Encoding can be applied in training loop (snntorch, Norse, etc.)")
    print("  - Or use data directly if already encoded (neuromorphic).")