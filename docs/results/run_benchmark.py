"""
Diagnostic benchmark across 3 SNN framework backends — Norse, snnTorch,
SpikingJelly — on real N-MNIST data.

Not a full training run: EPOCHS/ITERA are deliberately small (see CONFIG below)
so this finishes in one sitting and produces comparable loss/accuracy curves,
not converged models. TRADES and the custom CUDA kernel are both disabled —
TRADES because training.py's adversarial path doesn't yet check
is_differentiable() (see docs/frameworks/additional_frameworks.md), the kernel
because this is a framework comparison, not a kernel benchmark.

Usage:
    python docs/results/run_benchmark.py                    # N-MNIST (default)
    python docs/results/run_benchmark.py --dataset "ASL-DVS" # any DATASET_REGISTRY name
"""
import sys
import os
import argparse
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
from learning.training import SNNTrainer
from learning.inference import SNNTester
from learning.frameworks.snn_torch import SNN_TORCH
from learning.frameworks.snn_norse import SNN_NORSE
from learning.frameworks.snn_spikingjelly import SNN_SJ

CONFIG = dict(
    EPOCHS=1,
    ITERA=20,          # batches per "epoch" — small on purpose, see module docstring
    TEST_BATCHES=10,   # cap test set too — BindsNET especially is slow per-batch
)

CLASSIFICATION_MODELS = {
    "norse":    SNN_NORSE,
    "torch":    SNN_TORCH,
    "sj":       SNN_SJ,
}

# Regression variants (Phase B: MVSEC/TUM-VIE) live under frameworks/personal/,
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
    """Wraps a DataLoader to cap how many batches SNNTester iterates — full
    test set isn't needed for a diagnostic comparison run."""
    def __init__(self, loader, n):
        self.loader = loader
        self.n = n

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)


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


def run_one(name, ModelClass, cfg, train_loader, test_loader, device, data_dir):
    print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")
    cfg.FRAMEWORK = name

    model = ModelClass(cfg)
    trainer = SNNTrainer(model, train_loader, cfg, device)

    t0 = time.perf_counter()
    train_results = trainer.train(
        checkpoint_dir=str(ROOT / "checkpoints" / name),
        csv_path=str(data_dir / f"{name}_train.csv"),
    )
    train_time_s = time.perf_counter() - t0

    tester = SNNTester(model, LimitedLoader(test_loader, CONFIG["TEST_BATCHES"]), cfg, device)
    t0 = time.perf_counter()
    test_results = tester.run(csv_path=str(data_dir / f"{name}_test.csv"))
    test_time_s = time.perf_counter() - t0

    summary = {
        "framework": name,
        "train_time_s": train_time_s,
        "test_time_s": test_time_s,
        "loss_history": train_results["loss_history"],
        "accuracy_history": train_results["accuracy_history"],
        "spike_rate_history": train_results["spike_rate_history"],
        "test_overall_accuracy": test_results["overall_accuracy"],
        "test_avg_spikes_per_sample": test_results["avg_spikes_per_sample"],
        "test_avg_firing_rate_hz": test_results["avg_firing_rate_hz"],
        "test_avg_latency_ms": test_results["avg_latency_ms"],
        "test_avg_latency_per_sample_ms": test_results["avg_latency_per_sample_ms"],
        "test_energy_per_sample_pj": test_results["energy_per_sample_pj"],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="N-MNIST",
                         help=f"Any DATASET_REGISTRY name: {[e['name'] for e in DATASET_REGISTRY.values()]}")
    parser.add_argument("--all", action="store_true",
                         help="Run every dataset in DATASET_REGISTRY that has a working model "
                              "path today (classification datasets; regression ones are skipped "
                              "unless frameworks/personal/ is present — see docs/Haseeb-open-items.md). "
                              "One dataset's failure doesn't stop the others.")
    args = parser.parse_args()

    if args.all:
        runnable = [e["name"] for e in DATASET_REGISTRY.values() if e.get("kind", "classification") == "classification"]
        skipped  = [e["name"] for e in DATASET_REGISTRY.values() if e.get("kind") == "regression"]
        if skipped:
            print(f"Skipping (regression, not end-to-end trainable yet): {skipped}")
        overall_errors = {}
        for name in runnable:
            print(f"\n{'#'*60}\n#  DATASET: {name}\n{'#'*60}")
            try:
                run_dataset(name)
            except Exception as e:
                print(f"  !! Dataset '{name}' FAILED entirely: {e}")
                traceback.print_exc()
                overall_errors[name] = str(e)
        print("\n" + "=" * 60)
        print("  ALL-DATASETS RUN COMPLETE")
        print("=" * 60)
        for name in names:
            print(f"  {name:20s} {'FAILED — ' + overall_errors[name] if name in overall_errors else 'done'}")
        with open(HERE / "data" / "all_datasets_errors.json", "w") as f:
            json.dump(overall_errors, f, indent=2)
        return

    run_dataset(args.dataset)


def run_dataset(dataset_name: str):
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

    data_dir = HERE / "data" / dataset_slug(entry["name"])
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = Settings()
    cfg.DATASET_NAME = entry["name"]  # locks the pick; NeuromorphicEncoder won't re-prompt
    cfg.EPOCHS = CONFIG["EPOCHS"]
    cfg.ITERA = CONFIG["ITERA"]
    cfg.TRADES_ENABLED = False
    if cfg.DEVICE == "auto":
        # This is an automated/reproducible benchmark script — always autodetect,
        # never prompt (unlike main.py's select_hardware_config, which can).
        cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.DEVICE)

    print(f"Building {entry['name']} dataloaders (task_type={task_type})...")
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()
    print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)} (using {CONFIG['TEST_BATCHES']})")

    results = {}
    errors = {}
    for name, ModelClass in MODELS.items():
        try:
            results[name] = run_one(name, ModelClass, cfg, train_loader, test_loader, device, data_dir)
        except Exception as e:
            print(f"  !! {name} FAILED: {e}")
            traceback.print_exc()
            errors[name] = str(e)

    print("\n" + "=" * 60)
    print(f"  BENCHMARK COMPLETE — {entry['name']}")
    print("=" * 60)
    for name in MODELS:
        if name in results:
            r = results[name]
            if task_type == "classification":
                print(f"  {name:10s} acc={r['test_overall_accuracy']*100:5.1f}%  "
                      f"final_loss={r['loss_history'][-1]:.3f}  "
                      f"train_time={r['train_time_s']:.1f}s")
            else:
                print(f"  {name:10s} train_time={r['train_time_s']:.1f}s")
        else:
            print(f"  {name:10s} FAILED — {errors.get(name)}")

    with open(data_dir / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
