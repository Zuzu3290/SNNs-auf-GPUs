import sys
import os
import argparse
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root → skeleton, event_data_workflow
sys.path.insert(0, str(Path(__file__).parent.parent))          # src/ → learning, compiler

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA-capable GPU required. No GPU detected.")

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN_SJ
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder
from learning.adversarial_robustness import AdversarialEvaluator
from compiler import compile_model

torch.backends.cudnn.benchmark = True

_MODELS = {
    "norse": SNN_NORSE,
    "torch": SNN_TORCH,
    "sj":    SNN_SJ,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an SNN model")
    parser.add_argument(
        "--model",
        choices=_MODELS.keys(),
        default="norse",
        help="Model backend: norse | torch | sj  (default: norse)",
    )
    args = parser.parse_args()

    os.makedirs("./checkpoints", exist_ok=True)

    cfg    = Settings()
    device = torch.device(cfg.DEVICE)

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model = _MODELS[args.model](cfg)
    print(f"\n  Model backend  : {args.model.upper()}")

    if cfg.COMPILER_ENABLED:
        model = compile_model(model, cfg)

    cfg.display()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(checkpoint_dir="./checkpoints")
    print("\n Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    tester       = SNNTester(model, test_loader, cfg, device)
    test_results = tester.run()
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")

    evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
    evaluator.evaluate()
