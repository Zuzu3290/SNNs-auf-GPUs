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
from learning.frameworks.snn_sinabs import SNN_SINABS
from learning.training import SNNTrainer
from learning.inference import SNNTester
from event_data_workflow import NeuromorphicEncoder, resolve_dataset_entry
from learning.adversarial_robustness import AdversarialEvaluator

torch.backends.cudnn.benchmark = True

_MODELS = {
    "norse":    SNN_NORSE,
    "torch":    SNN_TORCH,
    "sj":       SNN_SJ,
    "sinabs":   SNN_SINABS,
}

# Regression model variants (Phase B: MVSEC/TUM-VIE) live under frameworks/personal/,
# which is gitignored — a personal, local-only setup, not shared with collaborators.
# A fresh checkout won't have it, so this import must degrade gracefully rather than
# crash main.py for everyone else.
try:
    from learning.frameworks.personal import (
        SNN_TORCH_REGRESSION, SNN_NORSE_REGRESSION, SNN_SJ_REGRESSION, SNN_SINABS_REGRESSION,
    )
    _REGRESSION_MODELS = {
        "norse":    SNN_NORSE_REGRESSION,
        "torch":    SNN_TORCH_REGRESSION,
        "sj":       SNN_SJ_REGRESSION,
        "sinabs":   SNN_SINABS_REGRESSION,
    }
except ImportError:
    _REGRESSION_MODELS = {}

# Task types each framework backend's output head/loss actually implements today.
_SUPPORTED_TASK_TYPES = {
    fw: {"classification"} | ({"regression"} if fw in _REGRESSION_MODELS else set())
    for fw in _MODELS
}


def select_hardware_config(cfg: Settings) -> str | None:
    """Resolve cfg.DEVICE == "auto" into a concrete hardware configuration.
    Any other value (cpu/cuda) is left untouched — this only fires when the
    config explicitly asks to be prompted. Interactive terminals get a
    numbered picker (same convention as NeuromorphicEncoder.select_dataset);
    non-interactive runs (Colab, CI, batch) autodetect instead, so this never
    blocks a scripted run. Returns a cache force_mode override, or None to
    leave cache-tier selection adaptive."""
    if cfg.DEVICE != "auto":
        return None

    if sys.stdin.isatty():
        print("\n[MAIN] Select a hardware configuration:")
        print("  1) CPU only — data loading, caching, and training all run on CPU")
        print("  2) Hybrid   — CPU caches recordings (RAM/disk), GPU trains")
        print("  3) GPU only — recordings cached in GPU VRAM, GPU trains")
        try:
            choice = input("Enter number [2]: ").strip() or "2"
        except EOFError:
            # isatty() can report True with no real input behind it (some
            # CI runners, notebook cells) — fall back instead of crashing.
            choice = "2" if torch.cuda.is_available() else "1"
        if choice not in ("1", "2", "3"):
            print(f"[MAIN] Invalid selection '{choice}' — defaulting to hybrid")
            choice = "2"
    else:
        choice = "2" if torch.cuda.is_available() else "1"

    if choice == "1":
        cfg.DEVICE = "cpu"
        return None
    if choice == "3":
        cfg.DEVICE = "cuda"
        return "gpu_memory"
    cfg.DEVICE = "cuda"
    return None


def select_inference_mode() -> bool:
    """Ask whether inference should show a live visualization alongside the usual
    statistical output, or statistics only. Interactive terminals get a numbered
    picker; non-interactive runs (Colab, CI, batch) default to statistics-only,
    since a live matplotlib window has no meaning without a display attached.
    Returns True if visualization was requested."""
    if not sys.stdin.isatty():
        return False

    print("\n[MAIN] Inference output:")
    print("  1) Statistics only")
    print("  2) Statistics + live visualization (opens a window showing input frames vs. predictions)")
    try:
        choice = input("Enter number [1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    if choice not in ("1", "2"):
        print(f"[MAIN] Invalid selection '{choice}' — defaulting to statistics only")
        choice = "1"
    return choice == "2"


if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    cfg = Settings()
    cache_force_mode = select_hardware_config(cfg)
    if cfg.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"cfg.DEVICE is 'cuda' (training.device in SNN_module.yaml) but no "
            f"CUDA-capable GPU was detected. Set training.device: cpu for a CPU-only run."
        )
    device = torch.device(cfg.DEVICE)

    # Resolve + lock the dataset choice before building the encoder, so an
    # incompatible dataset/framework pairing fails immediately — before any
    # download or caching — instead of crashing deep inside training with a
    # confusing shape/target-type mismatch.
    dataset_entry = resolve_dataset_entry(cfg)
    cfg.DATASET_NAME = dataset_entry["name"]  # locks the choice; NeuromorphicEncoder matches it directly, no re-prompt

    task_type = dataset_entry.get("kind", "classification")
    supported = _SUPPORTED_TASK_TYPES.get(cfg.FRAMEWORK, set())
    if task_type not in supported:
        reason = (
            "no regression model variant exists for this framework"
            if not _REGRESSION_MODELS
            else f"framework '{cfg.FRAMEWORK}' only implements: {sorted(supported)}"
        )
        raise RuntimeError(
            f"[MAIN] '{dataset_entry['name']}' requires task_type='{task_type}', but {reason}. "
            "Pick a classification dataset (N-MNIST, N-Caltech101, ASL-DVS, DVS128 Gesture) instead, "
            "or a framework with a regression variant (see docs/Haseeb-open-items.md)."
        )

    encoder = NeuromorphicEncoder(cfg, cache_force_mode=cache_force_mode)
    train_loader, test_loader = encoder.get_dataloaders()

    model_registry = _REGRESSION_MODELS if task_type == "regression" else _MODELS
    ModelClass = model_registry.get(cfg.FRAMEWORK)
    if ModelClass is None:
        raise ValueError(
            f"Unknown framework '{cfg.FRAMEWORK}' for task_type='{task_type}'. Valid: {list(model_registry)}"
        )
    model = ModelClass(cfg)
    print(f"\n  Model backend  : {cfg.FRAMEWORK.upper()}  (task_type={task_type})")

    if task_type == "regression":
        # SNNTrainer/SNNTester/AdversarialEvaluator below still assume classification —
        # targets.to(device).long(), accuracy_history, overall_accuracy, confusion-matrix-
        # shaped results. The model itself is ready (verified: builds, forwards, computes
        # mse_regression loss, backprops); wiring the trainer to a regression target/metric
        # path is the next concrete step, not done here. See docs/Haseeb-open-items.md.
        print("  [MAIN] Regression model built and verified in isolation, but the training/testing/"
              "adversarial-eval loop below still assumes classification targets and metrics — "
              "expect it to fail past this point until that's wired. See docs/Haseeb-open-items.md.")

    cfg.display()

    trainer = SNNTrainer(model, train_loader, cfg, device)
    results = trainer.train(checkpoint_dir="./checkpoints")
    print("\n Training complete!")
    print(f"  Final loss      : {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy  : {results['accuracy_history'][-1]:.4f}")
    print(f"  Final spike rate: {results['spike_rate_history'][-1]:.4f}")

    visualize = select_inference_mode() if task_type == "classification" else False
    tester       = SNNTester(model, test_loader, cfg, device, visualize=visualize)
    test_results = tester.run()
    print("\n Testing complete!")
    print(f"  Test accuracy  : {test_results['overall_accuracy'] * 100:.2f}%")
    print(f"  Energy/sample  : {test_results['energy_per_sample_pj']:.2f} pJ")
    print(f"  Avg Firing Rate : {test_results['avg_firing_rate_hz']:.2f} Hz")

    evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
    evaluator.evaluate()
