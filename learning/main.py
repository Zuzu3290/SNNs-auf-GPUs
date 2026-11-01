import gc
import importlib
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton.config import FW_TO_CFG_KEY, load_settings
from skeleton.seeding import param_report, seed_everything, seed_model_init
from skeleton.results_collection import make_run_id, record_run
from frameworks.snn_model import SNNModel
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.utilities import calibrate_batch_size, measure_activation_density, measure_latency

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = load_settings()
    seed_everything(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    inference_device = torch.device(cfg.INFERENCE_DEVICE)
    # The chosen framework's file, e.g. --framework sj -> frameworks/spikingjelly_lif.py; only that library is imported.
    neuron_class = importlib.import_module(f"frameworks.{FW_TO_CFG_KEY[cfg.FRAMEWORK]}_lif").Neuron

    dataset_entry = resolve_dataset_entry()
    cfg.DATASET_NAME = dataset_entry["name"]
    sensor_w, sensor_h, in_channels = dataset_entry["sensor_size"]
    cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                            num_classes=dataset_entry["num_classes"])
    if cfg.CALIBRATE_BATCH_SIZE: cfg.BATCH_SIZE = calibrate_batch_size(SNNModel, neuron_class, cfg, device, timesteps=cfg.N_TIME_BINS,
                                                                       data_vram_fraction=cfg.BATCH_VRAM_FRACTION, max_batch_size=cfg.MAX_BATCH_SIZE, band_min=cfg.BATCH_VRAM_BAND_MIN)

    train_loader, test_loader = NeuromorphicEncoder(cfg, dataset_entry).get_dataloaders()
    cfg.resolve_iterations(train_loader)

    # seeded right before construction, all four frameworks start from identical weights
    seed_model_init(cfg.SEED)
    model = SNNModel(cfg, neuron_class)
    params = param_report(model)
    logger.info(f"weight fingerprint : {params['shared_fingerprint']}   ({params['total_trainable']} trainable params)")
    run_id = make_run_id(cfg.FRAMEWORK, cfg.SEED)
    results_dir, plots_dir = Path(cfg.DATA_DIR), Path(cfg.PLOT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.display(net=model)

    # training 
    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(results_dir=str(results_dir))
    final = results["epoch_log"][-1]
    logger.info(f"Training complete!  epoch {final['epoch']}: loss {final['train_loss']:.4f}   "
                f"accuracy {final['train_accuracy']:.4f}   spike rate {final['spike_rate']:.4f}")
    trainer.plot_training(save_dir=str(plots_dir))

    # Saved before the trainer is freed, since it holds the optimizer; reset first, as sinabs keeps state in batch-shaped buffers.
    if cfg.SAVE_CHECKPOINT:
        checkpoint_path = results_dir / "model_checkpoint.pt"
        model.reset()
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "framework": cfg.FRAMEWORK, "neuron": model.describe_neuron()}, checkpoint_path)
        logger.info(f"Checkpoint : {checkpoint_path}")

    epoch_log = list(trainer.epoch_log)
    num_workers = train_loader.loader.num_workers
    del trainer, train_loader  # frees the persistent train workers before the test loader starts its own
    gc.collect()  # the dead trainer's objects first, so empty_cache can return their VRAM
    torch.cuda.empty_cache()

    # inference
    model.to(inference_device)
    tester = SNNTester(model, test_loader, cfg, inference_device)
    test_results = tester.run()
    tester.plot_raster(save_dir=str(plots_dir))
    logger.info(f"Testing complete!  accuracy {test_results['overall_accuracy'] * 100:.2f}%   "
                f"SynOps energy {test_results['synops_energy_per_sample_pj']:.2f} pJ per sample")

    # batch-size-1 latency and per-layer activation density, each its own untimed pass
    latency = None
    if cfg.LATENCY_SAMPLES > 0:
        latency = measure_latency(model, test_loader, inference_device, cfg.LATENCY_SAMPLES)
        logger.info(f"latency (bs=1) : median {latency['latency_ms']:.2f} ms   "
                    f"p90 {latency['latency_p90_ms']:.2f} ms   ({latency['latency_samples']} samples)")
    density = measure_activation_density(model, test_loader, inference_device)
    if density.get("hidden_mean") is not None:
        logger.info(f"activation density : {density['hidden_mean'] * 100:.2f}%  "
                    f"(hidden layers, {density['layers_measured']} measured)")

    # results: training_runs.csv, inference_runs.csv, epochs.csv and layers.csv are appended to, never overwritten
    record_run(results_dir, cfg, model, run_id, results, test_results, epoch_log, params,
               num_workers, latency, density)
