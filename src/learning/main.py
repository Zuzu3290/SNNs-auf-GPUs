import sys
import os
from pathlib import Path

# Add the 'src' directory to the path so we can import from 'skeleton' and 'learning'
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.event_data_workflow import NeuromorphicEncoder
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN_SJ

def main():
    # 1. Initialize settings
    cfg = Settings()

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model = SNN_SJ(cfg)
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
    
    results = trainer.train(checkpoint_dir="./checkpoints")

    print("\n✓ Training complete!")
    print(f"  Final loss     : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    trainer.plot_training()
    trainer.plot_raster()

    test_results = inference.run()

    print("\n✓ Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
