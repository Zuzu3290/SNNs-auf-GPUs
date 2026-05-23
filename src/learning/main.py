import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.event_data_workflow import NeuromorphicEncoder

def main():
    cfg = Settings()

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model = SNN_TORCH(cfg)
    trainer = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)
    
    print("\n✓ Model ready.")
    cfg.display()
    return model, trainer, inference

if __name__ == "__main__":
    cfg = Settings()
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
