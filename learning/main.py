import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton import Settings, WorkflowSettings
from skeleton.snn_logging import configure_logging
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.robustness import AdversarialEvaluator
from learning.utilities import calibrate_batch_size, select_inference_mode, safe_empty_cache

# module path, class name -- imported dynamically below, only for cfg.FRAMEWORK.
# DataLoader worker processes (Windows spawn re-imports this whole file) never
# touch a model class, only the dataset transform, so keeping these out of the
# module-level imports means workers don't pay for loading all four ML
# frameworks (each with its own CUDA extension / JIT backend) just to sit idle.
FRAMEWORK_MODULES = {
    "norse":    ("frameworks.snn_norse", "SNN_NORSE"),
    "torch":    ("frameworks.snn_torch", "SNN_TORCH"),
    "sj":       ("frameworks.snn_spikingjelly", "SNN_SJ"),
    "sinabs":   ("frameworks.snn_sinabs", "SNN_SINABS"),
}

if __name__ == "__main__":
    configure_logging()
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

    if cfg.FRAMEWORK not in FRAMEWORK_MODULES:
        raise ValueError(
            f"training.framework='{cfg.FRAMEWORK}' has no model implementation wired up. "
            f"Available: {sorted(FRAMEWORK_MODULES)}. ('spyx'/'bindsnet' exist as config "
            "scaffolding in network_architecture.yaml only — no frameworks/snn_*.py backs them yet.)"
        )
    module_name, class_name = FRAMEWORK_MODULES[cfg.FRAMEWORK]
    ModelClass = getattr(importlib.import_module(module_name), class_name)

    # Batch-size calibration ordering rationale: docs/functions.md
    sensor_w, sensor_h, in_channels = dataset_entry["sensor_size"]
    cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                             num_classes=dataset_entry["num_classes"])
    wf = WorkflowSettings()
    if cfg.CALIBRATE_BATCH_SIZE:
        cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device, timesteps=wf.N_TIME_BINS,
                                               data_vram_fraction=wf.BATCH_VRAM_FRACTION, max_batch_size=wf.MAX_BATCH_SIZE,
                                               band_min=wf.BATCH_VRAM_BAND_MIN, band_max=wf.BATCH_VRAM_BAND_MAX)

    cfg.ENABLE_PIPELINE_MONITOR = True  # background CPU/GPU utilization + power sampling; set False to disable

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    if cfg.CALIBRATE_BATCH_SIZE:
        # One epoch = one real pass over the actual training set, not a fixed
        # iteration count disconnected from batch size or dataset size.
        # len(DataLoader) rather than ceil(len(dataset)/batch_size): the
        # loader's own __len__ already accounts for drop_last=True (floor,
        # not ceil) -- using ceil() here previously overcounted by one
        # unreachable iteration, since the loader exhausts (drops the final
        # partial batch) one iteration before that ceil'd count. This
        # matches what actually executes, exactly, for any drop_last setting.
        cfg.ITERA = len(train_loader.loader)
        n_train_samples = len(train_loader.loader.dataset)
        covered = cfg.ITERA * cfg.BATCH_SIZE
        print(f"  [CALIBRATE] iterations/epoch = {cfg.ITERA}  "
              f"(covers {covered}/{n_train_samples} training samples per epoch, "
              f"{covered / n_train_samples * 100:.2f}%)")

    model = ModelClass(cfg)
    print(f"\n  Model backend  : {cfg.FRAMEWORK.upper()}")
    cfg.display()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train()
    print("\n Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    trainer.plot_training()
    trainer.plot_iteration_metrics()
    trainer.plot_raster()

    # train_loader has persistent_workers=True -- its worker processes stay alive
    # until this DataLoader is garbage-collected, so drop every reference (trainer
    # holds one too) before test_loader spawns its own workers on top of them.
    del trainer, train_loader
    safe_empty_cache()

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
