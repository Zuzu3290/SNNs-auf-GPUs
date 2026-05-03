"""
Universal data processing pipeline for Spiking Neural Networks.
NeuromorphicEncoder  — For event-based datasets (N-MNIST, DVS, etc.)
"""
from __future__ import annotations
 
import sys
import torch
from torch.utils.data import DataLoader
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torchvision
from snn_config import Settings


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
                frame_tf    = transforms.Compose([transforms.Denoise(filter_time=10000), transforms.ToFrame(sensor_size=sensor_size, time_window=1000)])
                full_set  = tonic.datasets.FileDataset(save_to=data_path, transform=frame_tf)
                # 80/20 split manually
                n_train   = int(0.8 * len(full_set))
                n_test    = len(full_set) - n_train
                trainset, testset = torch.utils.data.random_split(full_set, [n_train, n_test])
                self.dataset_label = getattr(self.cfg, "DATASET_NAME", "Custom Neuromorphic Dataset")
    
            self.sensor_size = sensor_size    
            # validate raw datasets before caching or DataLoader wrapping
            validate_first_sample(trainset, "train")
            validate_first_sample(testset,  "test")
    
            aug_tf = transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-10, 10])])
    
            if self.use_cache:
                train_data = DiskCachedDataset(trainset, transform=aug_tf, cache_path="./cache/train")
                test_data  = DiskCachedDataset(testset,  cache_path="./cache/test")
            else:
                train_data, test_data = trainset, testset
    
            pad = tonic.collation.PadTensors(batch_first=False)
            self._train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=pad, shuffle=True)
            self._test_loader  = DataLoader(test_data,  batch_size=batch_size, collate_fn=pad)
    
        def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
            return self._train_loader, self._test_loader
    
    
def validate_first_sample(dataset, split: str) -> None:
    """Index dataset[0] directly — no DataLoader overhead. Exits on failure."""
    try:
        events, target = dataset[0]
        if events is None or (hasattr(events, "numel") and events.numel() == 0):
            raise ValueError("first sample is empty")
    except Exception as exc:
        print(f"[ERROR] {split} dataset validation failed — {exc}")
        sys.exit(1)
 
 
def main():
    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg, use_cache=True)
    train_loader, test_loader = encoder.get_dataloaders()
 
    # --- runtime sensor-size report ---
    H, W, C = encoder.sensor_size
    print(f"[INFO] Dataset      : {encoder.dataset_label}")
    print(f"[INFO] Sensor size  : H={H}  W={W}  C={C}  (polarity channels)")
 
    # if encoder.input_size is not None and encoder.input_size != H * W * C:
    #     print(f"[WARN] architecture.input_size={encoder.input_size} does not match "
    #           f"flattened sensor size {H * W * C} — input layer requires adjustment.")
 
    return train_loader, test_loader
 
 
if __name__ == "__main__":
    train_loader, test_loader = main()
    print("\n✓ Dataloaders ready for training/testing.")
    print("  - Encoding can be applied in training loop (snntorch, Norse, etc.)")
    print("  - Or use data directly if already encoded (neuromorphic).")