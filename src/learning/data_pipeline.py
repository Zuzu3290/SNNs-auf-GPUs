"""
Universal data processing pipeline for Spiking Neural Networks.
NeuromorphicEncoder  — For event-based datasets (N-MNIST, DVS, etc.)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torchvision
from skeleton import Settings


class NeuromorphicEncoder:

    def __init__(self, cfg: Settings, use_cache: bool = True, data_format: str = "TB"):
        """
        data_format controls the batch tensor dimension order:
          "TB" → [T, B, C, H, W]  — used by SNNTorch and Norse (default)
          "BT" → [B, T, C, H, W]  — used by SpikingJelly step_mode='m'
        """
        self.cfg         = cfg
        self.use_cache   = use_cache
        self.data_format = data_format
        self.train_loader: DataLoader
        self.test_loader:  DataLoader
        self.build()

    def build(self):
        data_path  = self.cfg.DATA_PATH
        batch_size = self.cfg.BATCH_SIZE
        timesteps  = self.cfg.TIMESTEPS

        if not data_path:
            sensor_size = tonic.datasets.NMNIST.sensor_size
            # n_time_bins produces a fixed T for every sample — no padding needed
            # and avoids vanishing gradients caused by T≈1300 with time_window=1000
            frame_tf = transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
            ])
            trainset = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=True)
            testset  = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=False)
            self.dataset_label = "N-MNIST (default)"
        else:
            full_raw    = tonic.datasets.FileDataset(save_to=data_path)
            sensor_size = full_raw.sensor_size
            frame_tf = transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, n_time_bins=timesteps),
            ])
            full_set = tonic.datasets.FileDataset(save_to=data_path, transform=frame_tf)
            n_train  = int(0.8 * len(full_set))
            n_test   = len(full_set) - n_train
            trainset, testset = torch.utils.data.random_split(full_set, [n_train, n_test])
            self.dataset_label = getattr(self.cfg, "DATASET_NAME", "Custom Neuromorphic Dataset")

        self.sensor_size = sensor_size
        H, W, C = self.sensor_size
        print(f"[INFO] Sensor size  : H={H}  W={W}  C={C}  (polarity channels)")
        print(f"[INFO] Dataset      : {self.dataset_label}")
        print(f"[INFO] Timesteps    : {timesteps}  (n_time_bins — fixed T, no padding)")
        print(f"[INFO] Data format  : {self.data_format}")

        self.validate_first_sample(trainset, "train")
        self.validate_first_sample(testset,  "test")

        aug_tf = transforms.Compose([
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10, 10]),
        ])

        if self.use_cache:
            train_data = DiskCachedDataset(trainset, transform=aug_tf,
                                           cache_path=f"./cache/train_T{timesteps}")
            test_data  = DiskCachedDataset(testset,
                                           cache_path=f"./cache/test_T{timesteps}")
        else:
            train_data, test_data = trainset, testset

        # Fixed T → all samples have identical shape → standard stack collate,
        # no PadTensors needed. Permute here based on data_format so each
        # framework receives the tensor in its expected dimension order.
        def collate(batch):
            samples, labels = zip(*batch)
            x = torch.stack([torch.tensor(s, dtype=torch.float32) for s in samples])
            y = torch.tensor(labels, dtype=torch.long)
            if self.data_format == "TB":
                x = x.permute(1, 0, 2, 3, 4).contiguous()  # [B,T,C,H,W] → [T,B,C,H,W]
            return x, y

        self.train_loader = DataLoader(train_data, batch_size=batch_size,
                                       collate_fn=collate, shuffle=True)
        self.test_loader  = DataLoader(test_data,  batch_size=batch_size,
                                       collate_fn=collate)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.test_loader

    def validate_first_sample(self, dataset, split: str) -> None:
        """Index dataset[0] directly — no DataLoader overhead. Exits on failure."""
        try:
            events, target = dataset[0]
            if events is None or (hasattr(events, "numel") and events.numel() == 0):
                raise ValueError("first sample is empty")
        except Exception as exc:
            print(f"[ERROR] {split} dataset validation failed — {exc}")
            sys.exit(1)


def main() -> tuple[DataLoader, DataLoader]:
    cfg = Settings()
    return NeuromorphicEncoder(cfg).get_dataloaders()
    
