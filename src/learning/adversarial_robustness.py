from __future__ import annotations

import os
import csv
import torch
import torch.nn.functional as F
from typing import Sequence

from learning.training import aggregate_spike_output
from skeleton import Settings


def generate_fgsm_input(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Single-step gradient sign perturbation (FGSM).

    Uses the surrogate gradient already present in the SNN forward pass,
    so no special handling is needed per framework.
    """
    data = data.clone().detach().requires_grad_(True)
    spk_rec = model(data)
    loss = F.cross_entropy(aggregate_spike_output(spk_rec.float()), targets)
    loss.backward()
    return (data + epsilon * data.grad.sign()).detach()


def generate_pgd_input(
    model: torch.nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    steps: int,
) -> torch.Tensor:
    """Iterative FGSM with epsilon-ball projection (PGD).

    Step size alpha = 2 * epsilon / steps so the total budget is epsilon
    regardless of the number of iterations.
    """
    original = data.clone().detach()
    adv = data.clone().detach()
    alpha = 2.0 * epsilon / steps

    for _ in range(steps):
        adv = adv.requires_grad_(True)
        spk_rec = model(adv)
        loss = F.cross_entropy(aggregate_spike_output(spk_rec.float()), targets)
        loss.backward()
        adv = torch.clamp(
            (adv + alpha * adv.grad.sign()).detach(),
            original - epsilon,
            original + epsilon,
        )
    return adv


class AdversarialEvaluator:

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        cfg: Settings,
        device: torch.device,
    ):
        self.model = model
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device

    def compute_attack_accuracy(
        self,
        attack: str,
        epsilon: float,
        pgd_steps: int = 20,
    ) -> float:
        """Run one full pass over the test set under a given attack and return accuracy.

        attack: "clean" | "fgsm" | "pgd"
        """
        self.model.eval()
        correct = total = 0

        for data, targets in self.test_loader:
            data    = data.to(self.device)
            targets = targets.to(self.device).long()

            differentiable = getattr(self.model, "is_differentiable", lambda: True)()
            if attack == "fgsm" and differentiable:
                data = generate_fgsm_input(self.model, data, targets, epsilon)
            elif attack == "pgd" and differentiable:
                data = generate_pgd_input(self.model, data, targets, epsilon, pgd_steps)

            with torch.no_grad():
                spk_rec = self.model(data)
                preds = aggregate_spike_output(spk_rec.float()).argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

        return correct / total

    def evaluate(
        self,
        epsilons: Sequence[float] = (0.01, 0.05, 0.1, 0.2),
        pgd_steps: int = 20,
        csv_path: str = "./outputs/data/adversarial_robustness.csv",
    ) -> list[dict]:
        """Benchmark clean accuracy vs FGSM and PGD across epsilon values."""
        results = []

        clean_acc = self.compute_attack_accuracy("clean", epsilon=0.0)
        results.append({"attack": "clean", "epsilon": 0.0, "accuracy": round(clean_acc, 4)})

        print("\n[ADVERSARIAL ROBUSTNESS BENCHMARK]")
        print(f"  Clean accuracy : {clean_acc * 100:.2f}%\n")
        print(f"  {'Attack':<14} {'ε':>6}  {'Accuracy':>10}  {'Drop':>8}")
        print(f"  {'-' * 44}")

        for epsilon in epsilons:
            acc = self.compute_attack_accuracy("fgsm", epsilon)
            results.append({"attack": "FGSM", "epsilon": epsilon, "accuracy": round(acc, 4)})
            print(f"  {'FGSM':<14} {epsilon:>6.3f}  {acc * 100:>9.2f}%  {(clean_acc - acc) * 100:>7.1f}%")

        print()

        for epsilon in epsilons:
            acc = self.compute_attack_accuracy("pgd", epsilon, pgd_steps)
            results.append({"attack": f"PGD-{pgd_steps}", "epsilon": epsilon, "accuracy": round(acc, 4)})
            print(f"  {f'PGD-{pgd_steps}':<14} {epsilon:>6.3f}  {acc * 100:>9.2f}%  {(clean_acc - acc) * 100:>7.1f}%")

        write_robustness_csv(results, csv_path)
        return results


def write_robustness_csv(results: list[dict], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Adversarial robustness results saved → {path}")
