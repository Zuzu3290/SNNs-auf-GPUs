import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",   # clean output — no timestamp prefix cluttering the log
    force=True,
)

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN as SNN_SPIKINGJELLY
from learning.event_data_workflow.data_pipeline import NeuromorphicEncoder

_FRAMEWORKS = {
    "torch":        SNN_TORCH,
    "norse":        SNN_NORSE,
    "spikingjelly": SNN_SPIKINGJELLY,
}

def main():
    cfg = Settings()

    ModelClass = _FRAMEWORKS.get(cfg.FRAMEWORK)
    if ModelClass is None:
        raise ValueError(f"Unknown framework '{cfg.FRAMEWORK}'. Choose from: {list(_FRAMEWORKS)}")

    encoder = NeuromorphicEncoder(cfg, framework=cfg.FRAMEWORK)
    train_loader, test_loader = encoder.get_dataloaders()

    model     = ModelClass(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print(f"\n✓ Model ready  [{cfg.FRAMEWORK}]")
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
