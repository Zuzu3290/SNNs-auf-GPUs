"""
Universal data processing pipeline for Spiking Neural Networks.
NeuromorphicEncoder  — For event-based datasets (N-MNIST, DVS, etc.)
"""
from __future__ import annotations
 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, DistributedSampler
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torchvision
from skeleton import Settings

class NeuromorphicEncoder:

        def __init__(self, cfg: Settings, use_cache: bool = True):
            self.cfg = cfg
            self.use_cache = use_cache
            self.train_loader: DataLoader
            self.test_loader:  DataLoader
            self.build()
        
        def build(self):
            data_path = self.cfg.DATA_PATH
            batch_size = self.cfg.BATCH_SIZE
 
            if not data_path:
                sensor_size = tonic.datasets.NMNIST.sensor_size
                frame_tf    = transforms.Compose([transforms.Denoise(filter_time=10000), transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
                trainset = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=True)
                testset  = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=False)
                self.dataset_label = "N-MNIST (default)"
            else:
                full_raw    = tonic.datasets.FileDataset(save_to=data_path)
                sensor_size = full_raw.sensor_size
                frame_tf    = transforms.Compose(
                    [
                        transforms.Denoise(filter_time=10000), 
                        transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
                    ])
                full_set  = tonic.datasets.FileDataset(save_to=data_path, transform=frame_tf)
                # 80/20 split manually
                n_train   = int(0.8 * len(full_set))
                n_test    = len(full_set) - n_train
                trainset, testset = torch.utils.data.random_split(full_set, [n_train, n_test])
                self.dataset_label = getattr(self.cfg, "DATASET_NAME", "Custom Neuromorphic Dataset")
    
            self.sensor_size = sensor_size  
            H, W, C = self.sensor_size
            print(f"[INFO] Sensor size  : H={H}  W={W}  C={C}  (polarity channels)")
            print(f"[INFO] Dataset      : {self.dataset_label}")  
            # validate raw datasets before caching or DataLoader wrapping
            self.validate_first_sample(trainset, "train")
            self.validate_first_sample(testset,  "test")
    
            aug_tf = transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-30, 30])])
    
            if self.use_cache:
                train_data = DiskCachedDataset(trainset, transform=aug_tf, cache_path="./cache/train")
                test_data  = DiskCachedDataset(testset,  cache_path="./cache/test")
            else:
                train_data, test_data = trainset, testset
    
            num_workers = self.cfg.NUM_WORKERS
            pad = tonic.collation.PadTensors(batch_first=False)
            self.train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=pad,
                shuffle=True,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                drop_last=True,
            )
            self.test_loader = DataLoader(
                test_data,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=pad,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            #multi-threading using num_workers     

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



"""
First, sampler.set_epoch(epoch) must be called at the start of every epoch. The sampler uses the epoch number as a random seed for shuffling.
 If you forget this, every epoch will iterate over data in the same order, which degrades generalisation.

Second, pin_memory=True in the DataLoader pre-allocates page-locked host memory, enabling asynchronous CPU-to-GPU transfers 
when you call tensor.to(device, non_blocking=True). This overlap is where real throughput gains come from.

Third, persistent_workers=True avoids respawning worker processes every epoch — a significant overhead reduction when num_workers > 0.
"""
# def create_distributed_dataloader(dataset, config, ctx):
#     sampler = DistributedSampler(
#         dataset,
#         num_replicas=ctx.world_size,
#         rank=ctx.rank,
#         shuffle=True,
#     )
#     loader = DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         sampler=sampler,
#         num_workers=config.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=config.num_workers > 0,
#     )
#     return loader, sampler



