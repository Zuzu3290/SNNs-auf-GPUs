import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton, event_data_workflow, learning

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

torch.backends.cudnn.benchmark = False

_MODELS = {
    "norse":    SNN_NORSE,
    "torch":    SNN_TORCH,
    "sj":       SNN_SJ,
    "sinabs":   SNN_SINABS,
}


def select_hardware_config(cfg: Settings) -> str | None:
    """Resolve cfg.DEVICE == "auto" into a concrete hardware configuration.
    Any other value (cpu/cuda) is left untouched — this only fires when the
    config explicitly asks to be prompted.

    GPU-centric project: CPU-only was never a practical configuration here
    (SNN per-timestep compute is overwhelmingly GPU-favorable — see
    diagnostics/gpu_idle_elimination_plan.md). On a machine with no CUDA
    device, this now surfaces the existing "cfg.DEVICE is 'cuda' but no
    CUDA-capable GPU was detected" error further down instead of silently
    falling back to an unsupported CPU-only run.

    Always resolves to hybrid (CPU caches recordings, GPU trains) with
    cache-tier selection left adaptive — determine_dataset_strategy() picks
    memory/disk/hybrid/gpu_memory per-dataset based on whether it actually
    fits each tier's budget (see cache_engine.py)."""
    if cfg.DEVICE != "auto":
        return None
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
    if task_type != "classification":
        raise RuntimeError(
            f"[MAIN] '{dataset_entry['name']}' requires task_type='{task_type}', but this pipeline "
            "is classification-only. Pick a classification dataset (N-MNIST, N-Caltech101, "
            "ASL-DVS, DVS128 Gesture) instead."
        )

    encoder = NeuromorphicEncoder(cfg, cache_force_mode=cache_force_mode)
    train_loader, test_loader = encoder.get_dataloaders()

    ModelClass = _MODELS.get(cfg.FRAMEWORK)
    if ModelClass is None:
        raise ValueError(f"Unknown framework '{cfg.FRAMEWORK}'. Valid: {list(_MODELS)}")
    model = ModelClass(cfg)

    # torch.compile: builds an optimized execution graph instead of running
    # eager per-op kernels, cutting Python/kernel-launch overhead — see
    # SNN_GPU_Evaluation_Metrics.md's "Runtime GPU diagnostics" section.
    # Default mode (NOT "reduce-overhead"/CUDA Graphs — tried and reverted,
    # see diagnostics/gpu_performance_investigation_report.md): SNNTorch's
    # Leaky/Alpha neuron internal state causes tensor-rank instability (4 vs
    # 2) that repeatedly blows dynamo's recompile_limit under CUDA-graph
    # capture, measured at ~5-6 SECONDS/batch forever (~40-50x slower than
    # eager, never stabilizing) — CUDA Graphs assume a fixed, replay-safe
    # kernel sequence, which this model's state handling doesn't provide.
    # Plain default mode has no such requirement and measured 80-130ms/batch
    # after warmup, actually faster than eager's ~130-170ms.
    # suppress_errors is the safety net for platforms where the inductor/
    # Triton backend is unreliable (e.g. Windows), or where a given subgraph
    # can't be captured as a CUDA graph (e.g. a data-dependent branch): a
    # compile failure at trace time degrades that graph to eager instead of
    # crashing the run. Returns an OptimizedModule that proxies attribute
    # access to the wrapped model, so model.activity/.optimizer/.loss_fn/
    # .credit_assignment()/.synops_layer_map()/.get_state() etc. keep working
    # unchanged below.
    #
    # torch.compile(model, ...) itself never raises — compilation is lazy,
    # so the actual failure only surfaces on the FIRST real forward call,
    # deep inside the training loop, which a try/except here can't catch.
    # On CUDA, the inductor backend needs Triton for kernel codegen; without
    # it, suppress_errors keeps the run alive but dynamo still re-attempts
    # and re-logs a full traceback for every new graph it hits (one per
    # distinct shape/call-site in the T-step loop) — noisy, not a crash, but
    # not "graceful" either. Check once, upfront, and skip straight to eager
    # instead of a wall of repeated failures when we already know it can't work.
    use_compile = getattr(cfg, "USE_TORCH_COMPILE", True)
    if use_compile and device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            use_compile = False
            print("[MAIN] torch.compile skipped: no Triton installation found "
                  "(needed by the inductor backend for CUDA kernel codegen — "
                  "common on Windows). Running in eager mode.")

    if use_compile:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        try:
            model = torch.compile(model, mode=getattr(cfg, "TORCH_COMPILE_MODE", "default"))
        except Exception as e:
            print(f"[MAIN] torch.compile unavailable ({e}) — continuing in eager mode.")

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

    evaluator = AdversarialEvaluator(model, test_loader, cfg, device)
    evaluator.evaluate()
