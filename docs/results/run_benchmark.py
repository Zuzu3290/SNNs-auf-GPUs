"""
Benchmark across SNN framework backends — Norse, snnTorch, SpikingJelly, Sinabs —
on real event-camera data.

Two run modes:
  Diagnostic (default) — EPOCHS/ITERA deliberately small, finishes in one sitting,
    good for confirming everything runs and comparing the *shape* of behavior.
  Full (--full) — EPOCHS=5, iterates the entire train/test set, matching the
    methodology used to cross-check this project's parameter fairness against
    an external reference implementation (see docs/Haseeb-open-items.md).

Usage:
    python docs/results/run_benchmark.py                          # diagnostic, N-MNIST
    python docs/results/run_benchmark.py --dataset "ASL-DVS"       # any DATASET_REGISTRY name
    python docs/results/run_benchmark.py --full --cooldown-minutes 20
    python docs/results/run_benchmark.py --full --trades           # TRADES on during training
"""
import sys
import os
import argparse
import hashlib
import json
import time
import itertools
import traceback
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# training.py's logging prints unicode arrows (→); when stdout is redirected to
# a file on Windows it defaults to the cp1252 locale encoding, which can't
# represent that character and crashes every single training run. Reconfigure
# once, here, rather than touching the shared training.py for an environment quirk.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
import numpy as np

from skeleton import Settings
from event_data_workflow import NeuromorphicEncoder, DATASET_REGISTRY
from event_data_workflow.gpu_stats import GPUStats
from learning.training import SNNTrainer
from learning.inference import SNNTester
from learning.frameworks.snn_torch import SNN_TORCH
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN_SJ
from learning.frameworks.snn_sinabs import SNN_SINABS

DIAGNOSTIC_CONFIG = dict(
    EPOCHS=1,
    ITERA=20,          # batches per "epoch" — small on purpose, see module docstring
    TEST_BATCHES=10,
)
FULL_CONFIG = dict(
    EPOCHS=5,          # matches the reference run this project's parameter-fairness
    ITERA=None,        # fixes were cross-checked against — see Haseeb-open-items.md.
    TEST_BATCHES=None, # None = iterate every batch in the loader, not a capped subset.
)

CLASSIFICATION_MODELS = {
    "norse":    SNN_NORSE,
    "torch":    SNN_TORCH,
    "sj":       SNN_SJ,
    "sinabs":   SNN_SINABS,
}

# Regression variants (Phase B: DSEC) live under frameworks/personal/,
# gitignored/local-only — degrade gracefully if this checkout doesn't have it.
try:
    from learning.frameworks.personal import (
        SNN_TORCH_REGRESSION, SNN_NORSE_REGRESSION, SNN_SJ_REGRESSION,
    )
    REGRESSION_MODELS = {
        "norse": SNN_NORSE_REGRESSION,
        "torch": SNN_TORCH_REGRESSION,
        "sj":    SNN_SJ_REGRESSION,
    }
except ImportError:
    REGRESSION_MODELS = {}


def dataset_slug(name: str) -> str:
    """Filesystem-safe folder name for a dataset — keeps each dataset's data/plots separate."""
    return name.lower().replace(" ", "_").replace("-", "_")


class LimitedLoader:
    """Wraps a DataLoader to cap how many batches SNNTester iterates. n=None
    means no cap — iterate the whole loader (full-scale run)."""
    def __init__(self, loader, n):
        self.loader = loader
        self.n = n

    def __iter__(self):
        if self.n is None:
            return iter(self.loader)
        return itertools.islice(iter(self.loader), self.n)

    @property
    def dataset(self):
        """SNNTester.run() reads .dataset (for the duck-typed set_phase() cache
        hook) directly off whatever it's given — forward to the real loader."""
        return self.loader.dataset


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def weight_fingerprint(model) -> str:
    """Hash of trainable parameters — records exactly what was trained, for
    reproducibility tracking (same idea as a git commit hash for weights)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(
        (n, t) for n, t in model.named_parameters() if t.requires_grad
    ):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def energy_warnings(total_j, dynamic_j, idle_w, load_w) -> list[str]:
    """Reasons not to trust an energy measurement. A negative dynamic energy
    means the GPU drew LESS power while "working" than while idle — i.e. the
    GPU wasn't what was actually doing the work, or the baseline was polluted.
    Silently writing such a number is worse than flagging it."""
    problems: list[str] = []
    if total_j is None:
        problems.append("no power samples collected -- energy unmeasurable")
        return problems
    if dynamic_j is not None and dynamic_j < 0:
        problems.append(
            f"dynamic energy is NEGATIVE ({dynamic_j:.2f} J): idle baseline "
            f"exceeds power drawn under load."
        )
    if idle_w is not None and load_w is not None and load_w <= idle_w:
        problems.append(
            f"mean power under load ({load_w:.2f} W) is not above idle "
            f"({idle_w:.2f} W) -- the measured region did not load the GPU."
        )
    return problems


def run_one(name, ModelClass, cfg, train_loader, test_loader, device, data_dir, test_batches):
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    cfg.FRAMEWORK = name

    device_idx = (device.index or 0) if device.type == "cuda" else 0
    idle_gpu_stats = GPUStats(device_idx=device_idx)
    idle_cold_w = idle_gpu_stats.measure_idle_baseline(duration_s=3.0)
    print(f"  cold idle baseline: {idle_cold_w:.2f} W" if idle_cold_w is not None else "  cold idle baseline: N/A")

    model = ModelClass(cfg)
    fingerprint = weight_fingerprint(model)
    print(f"  weight fingerprint (post-init): {fingerprint}")

    trainer = SNNTrainer(model, train_loader, cfg, device)

    t0 = time.perf_counter()
    train_results = trainer.train(
        checkpoint_dir=str(ROOT / "checkpoints" / name),
        csv_path=str(data_dir / f"{name}_train.csv"),
    )
    train_time_s = time.perf_counter() - t0

    idle_hot_w = idle_gpu_stats.measure_idle_baseline(duration_s=3.0)
    print(f"  hot idle baseline (post-train): {idle_hot_w:.2f} W" if idle_hot_w is not None else "  hot idle baseline: N/A")

    tester = SNNTester(model, LimitedLoader(test_loader, test_batches), cfg, device)
    t0 = time.perf_counter()
    test_results = tester.run(csv_path=str(data_dir / f"{name}_test.csv"))
    test_time_s = time.perf_counter() - t0

    train_energy_j = None  # pulled from the per-epoch CSV below, matching make_plots.py's approach
    train_avg_power_w = None
    train_csv = data_dir / f"{name}_train.csv"
    if train_csv.exists():
        import csv as csv_module
        with open(train_csv) as f:
            rows = list(csv_module.DictReader(f))
        if rows:
            train_energy_j = sum(float(r["energy_j"]) for r in rows)
            train_avg_power_w = sum(float(r["avg_power_w"]) for r in rows) / len(rows)

    dynamic_train_energy_j = (
        train_energy_j - idle_hot_w * train_time_s
        if train_energy_j is not None and idle_hot_w is not None else None
    )
    warnings = energy_warnings(train_energy_j, dynamic_train_energy_j, idle_hot_w, train_avg_power_w)
    if warnings:
        print(f"  {len(warnings)} ENERGY WARNING(S):")
        for w in warnings:
            print(f"    - {w}")

    summary = {
        "framework": name,
        "weight_fingerprint": fingerprint,
        "idle_power_cold_w": idle_cold_w,
        "idle_power_hot_w": idle_hot_w,
        "train_time_s": train_time_s,
        "test_time_s": test_time_s,
        "train_energy_j": train_energy_j,
        "train_energy_dynamic_j": dynamic_train_energy_j,
        "train_avg_power_w": train_avg_power_w,
        "energy_warnings": " | ".join(warnings) if warnings else None,
        "loss_history": train_results["loss_history"],
        "accuracy_history": train_results["accuracy_history"],
        "spike_rate_history": train_results["spike_rate_history"],
        "test_overall_accuracy": test_results["overall_accuracy"],
        "test_avg_spikes_per_sample": test_results["avg_spikes_per_sample"],
        "test_avg_input_spikes_per_sample": test_results.get("avg_input_spikes_per_sample"),
        "test_framework_ratio": test_results.get("framework_ratio"),
        "test_avg_firing_rate_hz": test_results["avg_firing_rate_hz"],
        "test_avg_latency_ms": test_results["avg_latency_ms"],
        "test_avg_latency_per_sample_ms": test_results["avg_latency_per_sample_ms"],
        "test_p90_latency_per_sample_ms": test_results.get("p90_latency_per_sample_ms"),
        "test_p99_latency_per_sample_ms": test_results.get("p99_latency_per_sample_ms"),
        "test_throughput_samples_per_s": test_results.get("throughput_samples_per_s"),
        "test_energy_per_sample_pj": test_results["energy_per_sample_pj"],
        "test_gpu_mem_peak_gb": test_results.get("gpu_mem_peak_gb"),
        "test_class_metrics": test_results["class_metrics"],
        "test_confusion_matrix": test_results["confusion_matrix"],
        "test_gt_distribution": test_results["gt_distribution"],
        "test_pred_distribution": test_results["pred_distribution"],
    }

    out_path = data_dir / f"{name}_summary.json"
    with open(out_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)
    print(f"  -> saved {out_path}")

    del model, trainer, tester
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# Columns for the tabular results/runs.csv, one row per framework run — mirrors
# the flat, append-only schema style used to cross-check this project's
# parameter fairness (see docs/Haseeb-open-items.md), so results from either
# side can be compared directly rather than needing to unpack nested JSON.
RUN_CSV_COLUMNS = [
    "run_id", "dataset", "framework", "task_type", "epochs", "trades_enabled",
    "weight_fingerprint", "idle_power_cold_w", "idle_power_hot_w",
    "train_time_s", "test_time_s", "train_energy_j", "train_energy_dynamic_j",
    "train_avg_power_w", "energy_warnings",
    "final_train_loss", "final_train_accuracy",
    "test_overall_accuracy", "test_avg_latency_per_sample_ms",
    "test_p90_latency_per_sample_ms", "test_p99_latency_per_sample_ms",
    "test_throughput_samples_per_s", "test_avg_spikes_per_sample",
    "test_avg_input_spikes_per_sample", "test_framework_ratio",
    "test_gpu_mem_peak_gb",
]


def write_runs_csv(rows: list[dict], path: Path):
    import csv as csv_module
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=RUN_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n[TABLE] Appended {len(rows)} row(s) to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="N-MNIST",
                         help=f"Any DATASET_REGISTRY name: {[e['name'] for e in DATASET_REGISTRY.values()]}")
    parser.add_argument("--all", action="store_true",
                         help="Run every dataset in DATASET_REGISTRY that has a working model "
                              "path today (classification datasets; regression ones are skipped "
                              "unless frameworks/personal/ is present — see docs/Haseeb-open-items.md). "
                              "One dataset's failure doesn't stop the others.")
    parser.add_argument("--full", action="store_true",
                         help="Full-scale run: 5 epochs, entire train/test set (vs the small "
                              "diagnostic default). Takes much longer.")
    parser.add_argument("--trades", action="store_true",
                         help="Enable TRADES adversarial training (off by default here — "
                              "diagnostic/comparison runs use plain training).")
    parser.add_argument("--cooldown-minutes", type=float, default=0.0,
                         help="Idle wait between each framework's run, so consecutive runs don't "
                              "share GPU thermal/power state. 0 = no wait (default).")
    args = parser.parse_args()

    config = FULL_CONFIG if args.full else DIAGNOSTIC_CONFIG

    if args.all:
        runnable = [e["name"] for e in DATASET_REGISTRY.values() if e.get("kind", "classification") == "classification"]
        skipped  = [e["name"] for e in DATASET_REGISTRY.values() if e.get("kind") == "regression"]
        if skipped:
            print(f"Skipping (regression, not end-to-end trainable yet): {skipped}")
        overall_errors = {}
        for name in runnable:
            print(f"\n{'#'*60}\n#  DATASET: {name}\n{'#'*60}")
            try:
                run_dataset(name, config, args.trades, args.cooldown_minutes)
            except Exception as e:
                print(f"  !! Dataset '{name}' FAILED entirely: {e}")
                traceback.print_exc()
                overall_errors[name] = str(e)
        print("\n" + "=" * 60)
        print("  ALL-DATASETS RUN COMPLETE")
        print("=" * 60)
        for name in runnable:
            print(f"  {name:20s} {'FAILED — ' + overall_errors[name] if name in overall_errors else 'done'}")
        with open(HERE / "data" / "all_datasets_errors.json", "w") as f:
            json.dump(overall_errors, f, indent=2)
        return

    run_dataset(args.dataset, config, args.trades, args.cooldown_minutes)


def run_dataset(dataset_name: str, config: dict = DIAGNOSTIC_CONFIG, trades_enabled: bool = False,
                 cooldown_minutes: float = 0.0):
    entry = next((e for e in DATASET_REGISTRY.values() if e["name"].upper() == dataset_name.upper()), None)
    if entry is None:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid: {[e['name'] for e in DATASET_REGISTRY.values()]}")

    task_type = entry.get("kind", "classification")
    MODELS = REGRESSION_MODELS if task_type == "regression" else CLASSIFICATION_MODELS
    if task_type == "regression" and not MODELS:
        raise RuntimeError(
            f"'{entry['name']}' needs regression model variants (frameworks/personal/), "
            "not present in this checkout — see docs/Haseeb-open-items.md."
        )
    if task_type == "regression" and trades_enabled:
        print("  Note: --trades requested but this dataset is regression — TRADES doesn't "
              "apply to a dense target and is skipped automatically (see training.py).")

    data_dir = HERE / "data" / dataset_slug(entry["name"])
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = Settings()
    cfg.DATASET_NAME = entry["name"]  # locks the pick; NeuromorphicEncoder won't re-prompt
    cfg.EPOCHS = config["EPOCHS"]
    if config["ITERA"] is not None:
        cfg.ITERA = config["ITERA"]
    else:
        cfg.ITERA = 10**9  # sentinel "no cap" — SNNTrainer.train() breaks on the loader's own StopIteration first
    cfg.TRADES_ENABLED = trades_enabled
    if cfg.DEVICE == "auto":
        # This is an automated/reproducible benchmark script — always autodetect,
        # never prompt (unlike main.py's select_hardware_config, which can).
        cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.DEVICE)

    print(f"Building {entry['name']} dataloaders (task_type={task_type})...")
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()
    test_batches = config["TEST_BATCHES"]
    print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)} "
          f"(using {'all' if test_batches is None else test_batches})")
    print(f"Mode: {'FULL' if config is FULL_CONFIG else 'diagnostic'}   "
          f"Epochs: {cfg.EPOCHS}   TRADES: {'ON' if trades_enabled else 'off'}")

    results = {}
    errors = {}
    run_id = time.strftime("%Y%m%d-%H%M%S")
    csv_rows = []
    model_names = list(MODELS.items())
    for i, (name, ModelClass) in enumerate(model_names):
        try:
            results[name] = run_one(name, ModelClass, cfg, train_loader, test_loader, device, data_dir, test_batches)
            r = results[name]
            csv_rows.append({
                "run_id": run_id, "dataset": entry["name"], "framework": name,
                "task_type": task_type, "epochs": cfg.EPOCHS, "trades_enabled": trades_enabled,
                "weight_fingerprint": r["weight_fingerprint"],
                "idle_power_cold_w": r["idle_power_cold_w"], "idle_power_hot_w": r["idle_power_hot_w"],
                "train_time_s": round(r["train_time_s"], 2), "test_time_s": round(r["test_time_s"], 2),
                "train_energy_j": r["train_energy_j"], "train_energy_dynamic_j": r["train_energy_dynamic_j"],
                "train_avg_power_w": r["train_avg_power_w"], "energy_warnings": r["energy_warnings"],
                "final_train_loss": r["loss_history"][-1] if r["loss_history"] else None,
                "final_train_accuracy": r["accuracy_history"][-1] if r["accuracy_history"] else None,
                "test_overall_accuracy": r.get("test_overall_accuracy"),
                "test_avg_latency_per_sample_ms": r["test_avg_latency_per_sample_ms"],
                "test_p90_latency_per_sample_ms": r["test_p90_latency_per_sample_ms"],
                "test_p99_latency_per_sample_ms": r["test_p99_latency_per_sample_ms"],
                "test_throughput_samples_per_s": r["test_throughput_samples_per_s"],
                "test_avg_spikes_per_sample": r["test_avg_spikes_per_sample"],
                "test_avg_input_spikes_per_sample": r["test_avg_input_spikes_per_sample"],
                "test_framework_ratio": r["test_framework_ratio"],
                "test_gpu_mem_peak_gb": r["test_gpu_mem_peak_gb"],
            })
        except Exception as e:
            print(f"  !! {name} FAILED: {e}")
            traceback.print_exc()
            errors[name] = str(e)

        is_last = (i == len(model_names) - 1)
        if cooldown_minutes > 0 and not is_last:
            print(f"\n[COOLDOWN] Waiting {cooldown_minutes:.0f} minute(s) before the next framework "
                  f"(lets GPU power/thermal state settle between runs)...")
            time.sleep(cooldown_minutes * 60)

    if csv_rows:
        write_runs_csv(csv_rows, data_dir / "runs.csv")

    print("\n" + "=" * 60)
    print(f"  BENCHMARK COMPLETE — {entry['name']}")
    print("=" * 60)
    print(f"  {'framework':<10} {'acc':>8} {'loss':>8} {'train_s':>9} {'ratio':>8} {'p90_ms':>8}")
    for name in MODELS:
        if name in results:
            r = results[name]
            if task_type == "classification":
                ratio = r["test_framework_ratio"]
                ratio_str = f"{ratio:>8.2f}" if ratio is not None else f"{'N/A':>8}"
                print(f"  {name:<10} {r['test_overall_accuracy']*100:>7.1f}% "
                      f"{r['loss_history'][-1]:>8.3f} {r['train_time_s']:>8.1f}s "
                      f"{ratio_str} {r['test_avg_latency_per_sample_ms']:>7.2f}ms")
            else:
                print(f"  {name:<10} {'n/a':>8} {'n/a':>8} {r['train_time_s']:>8.1f}s")
        else:
            print(f"  {name:<10} FAILED — {errors.get(name)}")

    with open(data_dir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
