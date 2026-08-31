import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning, frameworks
import torch
from skeleton import Settings, WorkflowSettings
from skeleton.snn_logging import configure_logging
from skeleton.seeding import (
    param_report, seed_everything, seed_model_init, shared_weight_fingerprint,
)
from skeleton.results import make_run_id, write_results
from skeleton.results_collect import build_epoch_rows, build_layer_rows, build_run_row
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.robustness import AdversarialEvaluator
from learning.utilities import (
    calibrate_batch_size, collect_single_samples, measure_latency,
    safe_empty_cache, select_inference_mode,
)

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

def parse_args():
    """Every flag is optional and every default reproduces this pipeline's original
    behaviour, so `python learning/main.py` on its own runs exactly as it always has:
    the three base config files, a dataset prompt, and output to the `output:` paths.
    """
    import argparse

    from skeleton.cli import add_common_args

    parser = argparse.ArgumentParser(
        description="Train and evaluate one SNN framework on one event dataset.",
        epilog="examples:\n"
               "  python learning/main.py\n"
               "  python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 "
               "--framework sinabs --seed 1\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_common_args(parser).parse_args()


if __name__ == "__main__":
    from skeleton.cli import build, run_banner

    configure_logging()
    args = parse_args()
    cfg, wf_settings, run_info = build(args)
    print(run_banner("learning/main.py", cfg, run_info))
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
    wf = wf_settings  # same merged config; overlay and --cache-root already applied
    if cfg.CALIBRATE_BATCH_SIZE:
        cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device, timesteps=wf.N_TIME_BINS,
                                               data_vram_fraction=wf.BATCH_VRAM_FRACTION, max_batch_size=wf.MAX_BATCH_SIZE,
                                               band_min=wf.BATCH_VRAM_BAND_MIN, band_max=wf.BATCH_VRAM_BAND_MAX)

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
    results_dir, _, plots_dir = run_info["results_dir"], run_info["equivalence_dir"], run_info["plots_dir"]

    # ---- one folder per run, for everything with a fixed filename -----------------
    # training_results.csv, batch_metrics.csv, test.csv and the seven diagnostic PNGs
    # are all named without the framework or seed in them, so four frameworks writing
    # into one experiment folder would leave only the last one's files. Giving each run
    # its own subfolder keyed by run_id fixes that, and because the SAME run_id goes
    # into the runs.csv row, any figure can be traced back to the row that describes it.
    #
    # runs.csv / epochs.csv / layers.csv stay at the TOP of results_dir: they are
    # append-only across runs, which is the whole point of them.
    #
    # Only when --experiment routed the output. Without it nothing is nested and the
    # original ./outputs layout is untouched.
    run_id = make_run_id(cfg.FRAMEWORK, cfg.SEED)
    run_results_dir = (results_dir / run_id) if run_info["routed"] else results_dir
    run_plots_dir = (plots_dir / run_id) if run_info["routed"] else plots_dir
    for directory in (results_dir, run_results_dir, run_plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if run_info["routed"]:
        print(f"  run_id         : {run_id}   -> {run_results_dir}")

    # After the run directories exist, so the OUTPUT section can name the paths this
    # run will actually write to rather than the config's unrouted defaults.
    cfg.display(output_dirs={
        "Results dir": run_results_dir,
        "Plots dir":   run_plots_dir,
        "Shared CSVs": f"{results_dir}   (runs.csv, epochs.csv, layers.csv)",
    } if run_info["routed"] else None)

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(csv_path=str(run_results_dir / "training_results.csv"))
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

    trainer.plot_training(save_dir=str(run_plots_dir))
    trainer.plot_iteration_metrics(save_dir=str(run_plots_dir))
    trainer.plot_raster(save_dir=str(run_plots_dir))

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

    visualize = select_inference_mode()
    tester       = SNNTester(model, test_loader, cfg, device, visualize=visualize)
    test_results = tester.run(csv_path=str(run_results_dir / "test.csv"))
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Spikes/neuron   : {test_results['avg_spikes_per_neuron_per_inference']:.4f} per inference")
    if test_results["avg_firing_rate_hz"] is not None:
        print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")
    else:
        print("  Avg Firing Rate : n/a -- set framing.sample_duration_us for Hz")

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

    try:
        run_row = build_run_row(
            cfg, wf, model, run_info,
            train_results=results, test_results=test_results,
            epoch_log=epoch_log,
            params=params,
            timesteps=test_results.get("timesteps"), num_workers=num_workers,
            latency=latency,
            # The SAME id the run folder is named after, so a figure maps to its row.
            run_id=run_id,
            notes=" ".join(f"{k}={v}" for k, v in (run_info.get("overrides") or {}).items()),
        )
        paths = write_results(
            results_dir,
            run_row=run_row,
            epoch_rows=build_epoch_rows(epoch_log),
            layer_rows=build_layer_rows(model, activity_snapshot),
            json_payload={"run": run_row, "config_path": run_info.get("config_path"),
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

    RUN_ADVERSARIAL_EVAL = False

    if RUN_ADVERSARIAL_EVAL:
        evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
        # Routed like every other artefact. Called bare it would default to
        # ./outputs/data/ and be left behind on a Colab runtime.
        evaluator.evaluate(csv_path=str(run_results_dir / "adversarial_robustness.csv"))
