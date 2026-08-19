import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton import Settings
from frameworks.snn_torch import SNN_TORCH
from frameworks.snn_norse import SNN_NORSE
from frameworks.snn_spikingjelly import SNN_SJ
from frameworks.snn_sinabs import SNN_SINABS
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.robustness import AdversarialEvaluator
from learning.utilities import calibrate_batch_size, select_inference_mode

MODELS = {
    "norse":    SNN_NORSE,
    "torch":    SNN_TORCH,
    "sj":       SNN_SJ,
    "sinabs":   SNN_SINABS,
}

if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    cfg = Settings()
    device = torch.device(cfg.DEVICE)

    dataset_entry = resolve_dataset_entry(cfg)
    cfg.DATASET_NAME = dataset_entry["name"]  # locks the choice
    task_type = dataset_entry.get("kind", "classification")
    if task_type != "classification":
        raise RuntimeError(
            f"[MAIN] '{dataset_entry['name']}' requires task_type='{task_type}', but this pipeline "
            "is classification-only. Pick a classification dataset (N-MNIST, N-Caltech101, "
            "DVS128 Gesture) instead."
        )

    ModelClass = MODELS[cfg.FRAMEWORK]

    # Batch-size calibration ordering rationale: docs/functions.md
    sensor_w, sensor_h, in_channels = dataset_entry["sensor_size"]
    cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                             num_classes=dataset_entry["num_classes"])
    cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device)

    cfg.ENABLE_PIPELINE_MONITOR = True  # background CPU/GPU utilization + power sampling; set False to disable

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model = ModelClass(cfg)
    print(f"\n  Model backend  : {cfg.FRAMEWORK.upper()}")
    cfg.display()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(checkpoint_dir="./checkpoints")
    print("\n Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    visualize = select_inference_mode()
    tester       = SNNTester(model, test_loader, cfg, device, visualize=visualize)
    test_results = tester.run()
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")

    RUN_ADVERSARIAL_EVAL = False

    if RUN_ADVERSARIAL_EVAL:
        evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
        evaluator.evaluate()
