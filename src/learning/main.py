import torch
import os

from src.skeleton.snn_config import Settings
from src.skeleton.Encoding import build_dataloaders
from src.learning.snntorch_module.Craft import build_model
from src.skeleton.training import SNNTrainer
from src.skeleton.inference import SNNInference


def main():
    cfg = Settings()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.PLOT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.REPORT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SNN auf GPUs — Application Started")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Dataset type: {cfg.DATASET_TYPE}")
    print(f"Dataset name: {cfg.DATASET_NAME}")
    print(f"Model type: {cfg.MODEL_TYPE}")
    print(f"Network structure: {cfg.network_structure}")

    train_loader, test_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = SNNTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=cfg,
        device=device
    )

    results = trainer.train()

    inference = SNNInference(
        model=model,
        config=cfg,
        device=device
    )

    if cfg.SAVE_MODEL:
        model_path = os.path.join(cfg.MODEL_DIR, "snn_model.pth")
        inference.save_model(model_path)
        print(f"Model saved to: {model_path}")

    print("=" * 60)
    print("SNN auf GPUs — Application Finished")
    print("=" * 60)


if __name__ == "__main__":
    main()