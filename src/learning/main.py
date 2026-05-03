
from skeleton import Settings, NeuromorphicEncoder


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