import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton.config import settings_from_cli
from skeleton.snn_logging import configure_logging
from skeleton.seeding import param_report, seed_everything, seed_model_init
from skeleton.results_collection import build_epoch_rows, build_layer_rows, build_run_row, make_run_id, write_results
from frameworks.snn_model import SNNModel
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.utilities import (
    calibrate_batch_size, collect_single_samples, measure_activation_density,
    measure_latency, safe_empty_cache,
)

if __name__ == "__main__":
    configure_logging()
    cfg = settings_from_cli()
    seed_everything(cfg.SEED)  # data order, augmentation, random_split -- see skeleton/seeding.py
    device = torch.device(cfg.DEVICE)

    dataset_entry = resolve_dataset_entry(cfg)
    cfg.DATASET_NAME = dataset_entry["name"]  # locks the choice

    # Batch-size calibration ordering rationale: docs/functions.md
    sensor_w, sensor_h, in_channels = dataset_entry["sensor_size"]
    cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                             num_classes=dataset_entry["num_classes"])
    if cfg.CALIBRATE_BATCH_SIZE:
        cfg.BATCH_SIZE = calibrate_batch_size(SNNModel, cfg, device, timesteps=cfg.N_TIME_BINS,
                                               data_vram_fraction=cfg.BATCH_VRAM_FRACTION, max_batch_size=cfg.MAX_BATCH_SIZE,
                                               band_min=cfg.BATCH_VRAM_BAND_MIN)

    # No encoder reference is kept: it holds the train loader, whose persistent workers must exit before testing.
    train_loader, test_loader = NeuromorphicEncoder(cfg).get_dataloaders()

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
    model = SNNModel(cfg)
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
    results = trainer.train(results_dir=str(results_dir))
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
    print(f"  SynOps energy  : {test_results['synops_energy_per_sample_pj']:.2f} pJ per sample")

    # ------------------------------------------------------------------------------
    # Results, in the schema the SNNs_2 plotting layer reads (runs/epochs/layers.csv).
    # runs.csv is
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
        singles = collect_single_samples(test_loader, device, cfg.LATENCY_SAMPLES)
        latency = measure_latency(model, singles, device)
        print(f"\n  latency (bs=1)  : median {latency['latency_ms']:.2f} ms   "
              f"p90 {latency['latency_p90_ms']:.2f} ms   "
              f"({latency['latency_samples']} samples)")

    # ---- activation density, its own untimed pass ---------------------------------
    density = measure_activation_density(model, test_loader, device)
    if density.get("hidden_mean") is not None:
        print(f"  activation density: {density['hidden_mean'] * 100:.2f}%  "
              f"(hidden layers, {density['layers_measured']} measured)")

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
            dense_macs=test_results.get("dense_macs_per_layer"),
        ),
        json_payload={"run": run_row,
                      "neuron": model.describe_neuron(),
                      "test": {k: v for k, v in test_results.items()
                               if k not in ("confusion_matrix", "class_metrics")}},
    )
    print()
    print(f"results  : {paths['runs']}  (run_id {run_row['run_id']})")
    print(f"  epochs   : {paths['epochs']}")
    print(f"  layers   : {paths['layers']}")

