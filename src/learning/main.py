import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root → skeleton, event_data_workflow
sys.path.insert(0, str(Path(__file__).parent.parent))          # src/ → learning, compiler

import torch
from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN_SJ
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder
from learning.adversarial_robustness import AdversarialEvaluator
from compiler import compile_model
torch.backends.cudnn.benchmark = True #cuDNN's autotuner


if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    cfg    = Settings()
    device = torch.device(cfg.DEVICE)

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()
    model = SNN_NORSE(cfg)

    if cfg.COMPILER_ENABLED:
        model = compile_model(model, cfg)

    cfg.display()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(checkpoint_dir="./checkpoints")
    print("\n Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")
    # trainer.plot_training()
    # trainer.plot_raster()

    tester       = SNNTester(model, test_loader, cfg, device)
    test_results = tester.run()
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")

    evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
    evaluator.evaluate()
