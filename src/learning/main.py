import sys
import os
from pathlib import Path

# Add the 'src' directory to the path so we can import from 'skeleton' and 'learning'
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
# Using SpikingJelly framework
from learning.frameworks.snn_spikingjelly import SNN
from learning.data_pipeline import NeuromorphicEncoder as load_data

def main():
    # 1. Initialize settings
    cfg = Settings()
    
    # 2. Create the data encoder and retrieve the dataloaders
    # We pass 'cfg' to the class, then call the method to get the actual loaders
    encoder = load_data(cfg)
    train_loader, test_loader = encoder.get_dataloaders()
    
    # 3. Create the SpikingJelly model using our settings
    model = SNN(cfg)
    
    # 4. Setup trainer and inference engines
    trainer = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)
    
    print("\n✓ SpikingJelly Model ready.")
    cfg.display()
    
    return model, trainer, inference

if __name__ == "__main__":
    # Ensure a directory for checkpoints exists
    os.makedirs("./checkpoints", exist_ok=True)

    # Initialize the system
    model, trainer, inference = main()
    
    # 5. Start the Training process
    print("\nStarting training...")
    results = trainer.train(checkpoint_dir="./checkpoints")
    
    print("\n✓ Training complete!")
    
    # Check if results contains history before printing
    if results and 'loss_history' in results and len(results['loss_history']) > 0:
        final_loss = results['loss_history'][-1]
        final_acc = results['accuracy_history'][-1]
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Final accuracy: {final_acc:.4f}")
    else:
        print("   Training finished, but no history was returned.")