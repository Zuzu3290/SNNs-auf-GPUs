import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.data_pipeline import main as load_data  # Import the main() function

def main():
    cfg = Settings()
    
    # Get dataloaders from data_pipeline
    train_loader, test_loader = load_data()
    
    # Create model and pass loaders
    model = SNN_TORCH(cfg)
    trainer = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)
    
    print("\n✓ Model ready.")
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