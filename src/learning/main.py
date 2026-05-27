import sys
import os
from pathlib import Path

# Suppress TensorFlow C++ backend logs before any library imports.
# tonic (neuromorphic data) pulls in TF as an optional dependency; TF then
# warns about CUDA DLL mismatches that are irrelevant to PyTorch training.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add the 'src' directory to the path so we can import from 'skeleton' and 'learning'
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.event_data_workflow import NeuromorphicEncoder

torch.backends.cudnn.benchmark = True

def main():
    # 1. Initialize settings
    cfg = Settings()

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model = SNN_TORCH(cfg)

    if cfg.COMPILER_ENABLED:
        from compiler import compile_model
        model = compile_model(model, cfg)

    trainer              = model.get_trainer(train_loader)
    inference            = model.get_inference(test_loader)
    adversarial_evaluator = model.get_adversarial_evaluator(test_loader)

    print("\n✓ Torch Model ready.")
    cfg.display()

    return model, trainer, inference, adversarial_evaluator

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    model, trainer, inference, adversarial_evaluator = main()

    results = trainer.train(checkpoint_dir="./checkpoints")

    print("\n✓ Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    trainer.plot_training()
    trainer.plot_raster()

    test_results = inference.run()

    print("\n✓ Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")

    adversarial_evaluator.evaluate()
