import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton.config import add_args, from_args
from skeleton.snn_logging import configure_logging
from skeleton.seeding import (
    param_report, seed_everything, seed_model_init, shared_weight_fingerprint,
)
from skeleton.results_collection import build_epoch_rows, build_layer_rows, build_run_row, make_run_id, write_results
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.utilities import (
    calibrate_batch_size, collect_single_samples, measure_activation_density,
    measure_latency, safe_empty_cache,
)

FRAMEWORK_MODULES = {
    "norse":    ("frameworks.snn_norse", "SNN_NORSE"),
    "torch":    ("frameworks.snn_torch", "SNN_TORCH"),
    "sj":       ("frameworks.snn_spikingjelly", "SNN_SJ"),
    "sinabs":   ("frameworks.snn_sinabs", "SNN_SINABS"),
}

def parse_args():
    """Every flag is optional; with none, the three files in configuration/ are the whole config."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and evaluate one SNN framework on one event dataset.",
        epilog="examples:\n"
               "  python learning/main.py\n"
               "  python learning/main.py --framework sinabs --seed 1\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_args(parser).parse_args()


if __name__ == "__main__":
    configure_logging()
    cfg = from_args(parse_args())
    seed_everything(cfg.SEED)  # data order, augmentation, random_split -- see skeleton/seeding.py
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
    if cfg.CALIBRATE_BATCH_SIZE:
        cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device, timesteps=cfg.N_TIME_BINS,
                                               data_vram_fraction=cfg.BATCH_VRAM_FRACTION, max_batch_size=cfg.MAX_BATCH_SIZE,
                                               band_min=cfg.BATCH_VRAM_BAND_MIN)

    # Hardcoded on purpose, NOT a config key: background CPU/GPU utilization and power
    # sampling is a diagnostic, always wanted on a real run. Set False here to disable.
    cfg.ENABLE_PIPELINE_MONITOR = True

    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    # Runs whether or not the batch size was probed: epoch length follows from dataset
    # size and batch size either way.
    requested_itera = cfg.ITERA
    full_pass = len(train_loader.loader)
    cfg.resolve_iterations(train_loader)
    n_train_samples = len(train_loader.loader.dataset)
    covered = cfg.ITERA * cfg.BATCH_SIZE
    if requested_itera is None:
        note = "auto (full pass)"
    elif cfg.ITERA < requested_itera:
        note = f"capped to the full pass; config asked for {requested_itera}"
    else:
        note = f"capped by config at {requested_itera} of {full_pass}"
    print(f"  [ITERATIONS] {cfg.ITERA} batches/epoch x {cfg.BATCH_SIZE} batch = "
          f"{covered}/{n_train_samples} samples ({covered / n_train_samples * 100:.1f}%)  "
          f"-- {note}")

    # Seed immediately before construction, so weight init depends only on the seed
    # regardless of what drew from the RNG first (dataset probing, batch-size
    # calibration). This is what makes all four frameworks start from identical weights.
    seed_model_init(cfg.SEED)
    model = ModelClass(cfg)
    params = param_report(model)
    print(f"  weight fingerprint : {params['shared_fingerprint']}   "
          f"({params['total_trainable']} trainable params)")
    print(f"\n  Model backend  : {cfg.FRAMEWORK.upper()}")
    results_dir, plots_dir = Path(cfg.DATA_DIR), Path(cfg.PLOT_DIR)
    run_id = make_run_id(cfg.FRAMEWORK, cfg.SEED)
    for directory in (results_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfg.display(net=model.net)
    model.activity.clear()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(csv_path=str(results_dir / "training_results.csv"))
    # The last EPOCH, not the last batch. loss_history/accuracy_history are per-BATCH
    # series, so [-1] was one batch of 256 samples -- noisy, and it disagreed with the
    # "Epoch 5/5" block printed directly above it (0.9414 against 93.23%) for no reason
    # a reader could see. These now restate the final epoch, which is also what
    # epochs.csv and runs.csv record.
    print("\n Training complete!")
    final_epoch = results["epoch_log"][-1] if results.get("epoch_log") else None
    if final_epoch:
        print(f"  Final loss      : {final_epoch['train_loss']:.4f}   (epoch "
              f"{final_epoch['epoch']} mean)")
        print(f"  Final accuracy  : {final_epoch['train_accuracy']:.4f}   (epoch "
              f"{final_epoch['epoch']} mean)")
        print(f"  Final spike rate: {final_epoch['spike_rate']:.4f}   (epoch "
              f"{final_epoch['epoch']} mean)")
    else:
        print(f"  Final loss      : {results['loss_history'][-1]:.4f}   (last batch)")
        print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}   (last batch)")
        print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}   (last batch)")

    trainer.plot_training(save_dir=str(plots_dir))
    trainer.plot_iteration_metrics(save_dir=str(plots_dir))
    trainer.plot_raster(save_dir=str(plots_dir))

    # Copied out BEFORE the trainer is dropped below: epoch_log is the source for
    # epochs.csv, and the last activity snapshot is the free per-layer spike record
    # (hooked layers only) that layers.csv falls back to when spike counting was off.
    epoch_log = list(trainer.epoch_log)
    activity_snapshot = getattr(trainer, "last_activity_snapshot", None) or {}
    num_workers = getattr(getattr(train_loader, "loader", None), "num_workers", None)

    # train_loader has persistent_workers=True -- its worker processes stay alive
    # until this DataLoader is garbage-collected, so drop every reference (trainer
    # holds one too) before test_loader spawns its own workers on top of them.
    del trainer, train_loader
    safe_empty_cache()

    # Right at the train -> inference boundary, on the just-trained model, before
    # inference starts touching it (eval_mode() etc.) -- output.save_checkpoint in
    # SNN_module.yaml, opt-in and false by default (real disk usage every run
    # doesn't need). Load back with model.load_state(torch.load(path)).
    if cfg.SAVE_CHECKPOINT:
        checkpoint_path = results_dir / "model_checkpoint.pt"
        torch.save(model.get_state(), checkpoint_path)
        print(f"  Checkpoint     : {checkpoint_path}")

    tester       = SNNTester(model, test_loader, cfg, device)
    test_results = tester.run(csv_path=str(results_dir / "test.csv"))
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Spikes/neuron   : {test_results['avg_spikes_per_neuron_per_inference']:.4f} per inference")

    # ------------------------------------------------------------------------------
    # Results, in the schema the SNNs_2 plotting layer reads (runs/epochs/layers.csv).
    # ADDITIVE: training_results.csv and test.csv above are untouched. runs.csv is
    # APPEND-ONLY, so running one framework per invocation -- one Colab cell each, then
    # again with a different seed -- accumulates into a single comparable table. The
    # number of seeds per framework does not have to match.
    # ------------------------------------------------------------------------------
    # ---- batch-size-1 latency, its own untimed pass -------------------------------
    # Separate from the batched test above because it answers a different question:
    # "one event arrives, how long until the answer is ready" (MLPerf Single-Stream),
    # not "how much wall-clock does each sample cost at this batch size". Dividing a
    # batch time by the batch size gives the second and is often mistaken for the first.
    latency = None
    if cfg.LATENCY_SAMPLES > 0:
        try:
            singles = collect_single_samples(test_loader, device, cfg.LATENCY_SAMPLES)
            latency = measure_latency(model, singles, device)
            print(f"\n  latency (bs=1)  : median {latency['latency_ms']:.2f} ms   "
                  f"p90 {latency['latency_p90_ms']:.2f} ms   "
                  f"({latency['latency_samples']} samples)")
        except Exception as exc:  # a diagnostic must not cost a finished run
            print(f"\n  !! latency (bs=1) not measured: {type(exc).__name__}: {exc}")

    # ---- activation density, its own untimed pass ---------------------------------
    density = {}
    try:
        density = measure_activation_density(model, test_loader, device)
        if density.get("hidden_mean") is not None:
            print(f"  activation density: {density['hidden_mean'] * 100:.2f}%  "
                  f"(hidden layers, {density['layers_measured']} measured)")
    except Exception as exc:  # a diagnostic must not cost a finished run
        print(f"\n  !! activation density not measured: {type(exc).__name__}: {exc}")

    try:
        run_row = build_run_row(
            cfg, model,
            train_results=results, test_results=test_results,
            epoch_log=epoch_log,
            params=params,
            timesteps=test_results.get("timesteps"), num_workers=num_workers,
            latency=latency,
            density=density,
            # The SAME id the run folder is named after, so a figure maps to its row.
            run_id=run_id,
            notes=" ".join(f"{k}={v}" for k, v in cfg.OVERRIDES.items()),
        )
        paths = write_results(
            results_dir,
            run_row=run_row,
            epoch_rows=build_epoch_rows(epoch_log),
            layer_rows=build_layer_rows(
                model, activity_snapshot,
                cv_isi=test_results.get("cv_isi_per_layer"),
                dense_macs=test_results.get("dense_macs_per_layer"),
            ),
            json_payload={"run": run_row, "config_path": cfg.CONFIG_PATH,
                          "neuron": model.describe_neuron(),
                          "test": {k: v for k, v in test_results.items()
                                   if k not in ("confusion_matrix", "class_metrics")}},
        )
        print()
        print(f"results  : {paths['runs']}  (run_id {run_row['run_id']})")
        print(f"  epochs   : {paths['epochs']}")
        print(f"  layers   : {paths['layers']}")
    except Exception as exc:  # never lose a finished run to a bookkeeping error
        print()
        print(f"!! results not written: {type(exc).__name__}: {exc}")

