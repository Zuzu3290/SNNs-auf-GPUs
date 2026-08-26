"""
Reproducibility: seeding and weight fingerprinting.

Ported from the SNNs_2 controlled-comparison pipeline. The design is deliberately
NARROW rather than a blanket "make everything deterministic" hammer, because the
property being protected is specific:

    with the same seed, every framework must start from byte-identical weights.

That holds only because the sole consumers of the RNG during model construction are
the two Conv2d layers and the Linear, in that order. Every spiking-neuron layer wired
into this project (snnTorch Leaky, Norse LIFCell, SpikingJelly LIFNode, Sinabs LIF)
draws no random numbers at construction, so seeding immediately before building the
model is enough to make the four backends comparable. `verify_cross_framework_init()`
is the check that this assumption still holds; run it rather than trusting it.

`torch.use_deterministic_algorithms()` is intentionally NOT set here. It would force
slower kernels and change the very latency numbers this project exists to measure.
Seeding fixes the starting point and the data order; it does not claim bitwise-identical
GPU arithmetic across runs.
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

# Parameter-name fragments that identify the dense layers shared by all four
# backends. Used to fingerprint only the weights every framework actually has in
# common -- Sinabs additionally registers one trainable `tau_mem` per spiking layer
# (3 extra tensors), so a fingerprint over *all* trainable params can never match
# across backends even when the conv/linear weights are byte-identical.
SHARED_PARAM_HINTS = ("conv", "fc", "linear", "weight", "bias")


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect a run's trajectory.

    numpy and the stdlib `random` matter here as well as torch: torchvision's
    RandomRotation (the train augmentation in data_pipeline) and
    torch.utils.data.random_split both draw from torch, while tonic's own
    transforms can reach for numpy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_generator(seed: int) -> torch.Generator:
    """Generator for `random_split`, so a dataset with no predefined train/test
    split (N-Caltech101, DSEC) is divided the same way every run. Without this the
    split silently changes between runs and no two runs are comparable."""
    return torch.Generator().manual_seed(seed)


# Loader order is drawn from its own stream, offset from the model seed, so batch
# order and weight init never share a sequence.
LOADER_SEED_OFFSET = 10_000


def loader_seed(seed: int) -> int:
    return seed + LOADER_SEED_OFFSET


def loader_generator(seed: int) -> torch.Generator:
    """Generator for a shuffling DataLoader, so batch order depends only on this
    seed and not on whatever else in the process has already drawn from the global
    RNG."""
    return torch.Generator().manual_seed(loader_seed(seed))


def reset_loader_order(loader, seed: int) -> bool:
    """Re-seed a DataLoader's generator so the NEXT full pass replays the identical
    batch sequence.

    Required whenever several frameworks are trained in one process (as
    run_comparison.py does). A DataLoader draws from its generator on
    every `iter()` call, advancing it, so the second framework over the same loader
    object sees a DIFFERENT batch order than the first -- measured: pass 1 gave
    labels [5,2,2,8...], pass 2 gave [7,7,4,1...]. Left unreset, a cross-framework
    comparison is partly measuring which batches each framework happened to get.

    Accepts either a raw DataLoader or a PrefetchedLoader wrapping one. Returns
    False if the loader has no generator (an unshuffled loader, already deterministic).
    """
    inner = getattr(loader, "loader", loader)
    generator = getattr(inner, "generator", None)
    if generator is None:
        return False
    generator.manual_seed(loader_seed(seed))
    return True


def seed_model_init(seed: int) -> None:
    """Call immediately before constructing a model, so weight init depends only on
    the seed regardless of what ran beforehand (dataset probing, batch-size
    calibration, another framework's model). This is the call that makes the
    cross-framework weight fingerprints match."""
    torch.manual_seed(seed)


def _shared_params(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    named = [(n, t) for n, t in model.named_parameters() if t.requires_grad]
    shared = [
        (n, t) for n, t in named
        if any(h in n.lower() for h in SHARED_PARAM_HINTS) and "tau" not in n.lower()
    ]
    return sorted(shared, key=lambda kv: kv[0])


def _digest(tensors: Iterable[tuple[str, torch.Tensor]], name_sensitive: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        if name_sensitive:
            digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()[:16]


def weight_fingerprint(model: nn.Module) -> str:
    """Hash of every trainable parameter, names included. Identifies exactly what was
    trained, but is NOT comparable across backends -- Sinabs' extra `tau_mem` tensors
    and each backend's own module naming both change it. Use for run provenance."""
    named = sorted(
        ((n, t) for n, t in model.named_parameters() if t.requires_grad),
        key=lambda kv: kv[0],
    )
    return _digest(named, name_sensitive=True)


def shared_weight_fingerprint(model: nn.Module) -> str:
    """Hash of the conv/linear weights only, with names EXCLUDED so that differing
    module naming across backends (`net.0.weight` vs `conv1.weight`) doesn't change
    the result. This is the cross-framework gate: same seed must give the same value
    for all four backends."""
    return _digest(_shared_params(model), name_sensitive=False)


def param_report(model: nn.Module) -> dict:
    """Everything needed for Gate A in one call."""
    trainable = [t for t in model.parameters() if t.requires_grad]
    shared = _shared_params(model)
    return {
        "total_trainable": sum(t.numel() for t in trainable),
        "n_tensors": len(trainable),
        "shared_trainable": sum(t.numel() for _, t in shared),
        "n_shared_tensors": len(shared),
        "fingerprint": weight_fingerprint(model),
        "shared_fingerprint": shared_weight_fingerprint(model),
    }


def verify_cross_framework_init(reports: dict[str, dict]) -> tuple[bool, list[str]]:
    """Gate A1 + A2. Given {framework: param_report(...)} built under one seed,
    report whether every backend started from the same weights.

    Returns (passed, list of human-readable problems).
    """
    problems: list[str] = []
    if not reports:
        return False, ["no reports given"]

    shared_counts = {fw: r["shared_trainable"] for fw, r in reports.items()}
    if len(set(shared_counts.values())) != 1:
        problems.append(f"shared trainable param COUNT differs: {shared_counts}")

    prints = {fw: r["shared_fingerprint"] for fw, r in reports.items()}
    if len(set(prints.values())) != 1:
        problems.append(f"shared weight FINGERPRINT differs: {prints}")

    totals = {fw: r["total_trainable"] for fw, r in reports.items()}
    if len(set(totals.values())) != 1:
        problems.append(
            f"total trainable param count differs: {totals} -- expected when a backend "
            "registers extra state as parameters (Sinabs' trainable tau_mem); a Gate A1 "
            "failure unless that is deliberately configured away"
        )

    return not problems, problems
