import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
from old_pipeline.frameworks.snn_torch import SNN_TORCH
from old_pipeline.frameworks.snn_norse import SNN_NORSE
from old_pipeline.data_pipeline import main as load_data

FRAMEWORK = "torch"   # switch between "torch" and "norse"

def main():
    cfg = Settings()

    # Get dataloaders from data_pipeline
    train_loader, test_loader = load_data()

    # Select framework
    if FRAMEWORK == "norse":
        model = SNN_NORSE(cfg)
    else:
        model = SNN_TORCH(cfg)

    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print(f"\n✓ Model ready  [{FRAMEWORK}]")
    cfg.display()
    return model, trainer, inference

if __name__ == "__main__":
    cfg = Settings()
    model, trainer, inference = main()

    # Train the model
    results = trainer.train(checkpoint_dir="./checkpoints")

    print("\n✓ Training complete!")
    print(f"  Final loss: {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy: {results['accuracy_history'][-1]:.4f}")