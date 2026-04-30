# =============================================================
# run.py — Single entry point for the Norse SNN system
# =============================================================
# Usage:
#   python run.py
#
# What it does:
#   1. Reads config.py
#   2. Loads N-MNIST dataset
#   3. Builds the SNN model
#   4. Trains for config["epochs"] epochs
#   5. Evaluates on test set (all metrics)
#   6. Saves results to results/<neuron_type>_<decoding>_T<timesteps>/
#
# To run a different experiment:
#   change neuron_type / any parameter in config.py → run again
#   a new results folder is created automatically

import os
import sys
import torch

# allow imports from this folder regardless of where run.py is called from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config  import CONFIG
from dataset import get_dataloaders
from model   import build_model
from train   import train
from evaluate import evaluate


def main():
    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Norse SNN Experiment Runner")
    print(f"{'='*55}")
    print(f"  Device       : {device}")
    print(f"  Neuron type  : {CONFIG['neuron_type']}")
    print(f"  Decoding     : {CONFIG['decoding']}")
    print(f"  Timesteps    : {CONFIG['timesteps']}")
    print(f"  Hidden size  : {CONFIG['hidden_size']}")
    print(f"  Num layers   : {CONFIG['num_layers']}")
    print(f"  Surrogate    : {CONFIG['surrogate']}")
    print(f"  Optimizer    : {CONFIG['optimizer']}")
    print(f"  Epochs       : {CONFIG['epochs']}")
    print(f"  Batch size   : {CONFIG['batch_size']}")
    print(f"  LR           : {CONFIG['lr']}")
    print(f"  tau_mem      : {CONFIG['tau_mem']}")
    print(f"  threshold    : {CONFIG['threshold']}")
    print(f"  dropout      : {CONFIG['dropout_p']}")
    print(f"  weight_decay : {CONFIG['weight_decay']}")
    print(f"{'='*55}\n")

    # ── Results folder ────────────────────────────────────────
    # Auto-named from key config values so each experiment
    # gets its own folder and never overwrites another.
    # Example: results/LIF_rate_T16/
    run_label  = (
        f"{CONFIG['neuron_type']}_"
        f"{CONFIG['decoding']}_"
        f"T{CONFIG['timesteps']}_"
        f"H{CONFIG['hidden_size']}_"
        f"L{CONFIG['num_layers']}"
    )
    results_dir = os.path.join(CONFIG["results_dir"], run_label)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}\n")

    # ── Step 1: Load dataset ──────────────────────────────────
    print("─" * 40)
    print("Step 1: Loading N-MNIST dataset")
    print("─" * 40)
    train_loader, test_loader = get_dataloaders(CONFIG)

    # ── Step 2: Build model ───────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 2: Building SNN model")
    print("─" * 40)
    model = build_model(CONFIG).to(device)

    # ── Step 3: Train ─────────────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 3: Training")
    print("─" * 40)
    log_rows = train(model, train_loader, CONFIG, results_dir, device)

    # ── Step 4: Evaluate ──────────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 4: Evaluating on test set")
    print("─" * 40)
    results = evaluate(model, test_loader, CONFIG, log_rows, results_dir, device)

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Run complete: {run_label}")
    print(f"  Test accuracy    : {results['test_acc']}%")
    print(f"  Spike sparsity   : {results['sparsity']}%")
    print(f"  Inference time   : {results['infer_time']} ms/sample")
    print(f"  Convergence epoch: {results['conv_epoch']}")
    print(f"  Results saved to : {results_dir}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
