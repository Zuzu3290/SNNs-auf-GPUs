"""Unit tests for skeleton/seeding.py.

    python tests/unit_seeding.py

This pipeline had NO seeding of any kind before the merge, so any measured difference
between two frameworks was confounded with initialisation noise. These tests cover the
weight-init half, which is what the frameworks/network unit wired in.

The data-order half (loader_generator, split_generator, reset_loader_order) is tested
here at the generator level only. Wiring it into the real DataLoaders belongs to the
data unit, so the tests that need a live loader are not here yet.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg
from frameworks.adapters import lif_factory
from frameworks.spiking_net import build_network
from skeleton.seeding import (
    LOADER_SEED_OFFSET, loader_generator, loader_seed, param_report,
    reset_loader_order, seed_everything, seed_model_init, shared_weight_fingerprint,
    split_generator, verify_cross_framework_init, weight_fingerprint,
)

suite = Suite("unit_seeding")


def fingerprint_for(framework: str, seed: int) -> str:
    cfg = fresh_cfg()
    seed_model_init(seed)
    return shared_weight_fingerprint(build_network(lif_factory(framework, cfg), cfg))


# ---------------------------------------------------------------------------
# 1. weight init
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_within_a_framework() -> None:
    for framework in FRAMEWORKS:
        first, second = fingerprint_for(framework, 0), fingerprint_for(framework, 0)
        suite.check(f"same seed twice gives the same weights: {framework}", first == second,
                    f"{first} vs {second}")


def test_different_seed_changes_the_weights() -> None:
    for framework in FRAMEWORKS:
        suite.check(f"a different seed gives different weights: {framework}",
                    fingerprint_for(framework, 0) != fingerprint_for(framework, 1))


def test_all_frameworks_share_one_fingerprint_per_seed() -> None:
    """The cross-framework gate. Holds only because no spiking layer draws from the RNG
    at construction -- if one ever did, the conv/linear weights would shift behind it
    and this is what would catch it."""
    for seed in (0, 1, 7):
        prints = {fw: fingerprint_for(fw, seed) for fw in FRAMEWORKS}
        suite.check(f"seed {seed}: all four share one fingerprint",
                    len(set(prints.values())) == 1, str(prints))


def test_seeding_is_immune_to_prior_rng_use() -> None:
    """seed_model_init is called immediately before construction precisely so that
    anything which drew from the global RNG first -- dataset probing, batch-size
    calibration, another framework's model -- cannot shift the weights."""
    baseline = fingerprint_for("torch", 0)
    cfg = fresh_cfg()
    torch.rand(1000)          # something else drawing from the global RNG
    random.random()
    np.random.rand(10)
    seed_model_init(0)
    after = shared_weight_fingerprint(build_network(lif_factory("torch", cfg), cfg))
    suite.check("prior RNG use does not shift the weights", baseline == after,
                f"{baseline} vs {after}")


def test_building_one_framework_does_not_shift_the_next() -> None:
    """A multi-framework run builds four models in one process, back to back."""
    baseline = fingerprint_for("norse", 0)
    cfg = fresh_cfg()
    seed_model_init(0)
    build_network(lif_factory("torch", cfg), cfg)   # a model built first
    seed_model_init(0)
    after = shared_weight_fingerprint(build_network(lif_factory("norse", cfg), cfg))
    suite.check("an earlier model does not shift a later one", baseline == after,
                f"{baseline} vs {after}")


# ---------------------------------------------------------------------------
# 2. the two fingerprints do different jobs
# ---------------------------------------------------------------------------
def test_shared_fingerprint_ignores_module_naming() -> None:
    """shared_weight_fingerprint excludes parameter NAMES so that differing module
    naming across backends does not change it; weight_fingerprint includes them, so it
    identifies exactly what was trained but is not comparable across backends."""
    shared, full = {}, {}
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        shared[framework] = shared_weight_fingerprint(model)
        full[framework] = weight_fingerprint(model)
    suite.check("shared fingerprint matches across frameworks",
                len(set(shared.values())) == 1, str(shared))
    suite.check("full fingerprint is a stable per-framework identity",
                all(isinstance(v, str) and len(v) == 16 for v in full.values()))


def test_fingerprint_tracks_a_weight_change() -> None:
    """Guards against a fingerprint that is constant for some reason other than the
    weights -- every equality test above would still pass in that case."""
    model, _ = build_model("torch")
    before = shared_weight_fingerprint(model)
    with torch.no_grad():
        model.net.layers[0].weight.add_(1.0)
    suite.check("changing a weight changes the fingerprint",
                before != shared_weight_fingerprint(model))


def test_param_report_contents() -> None:
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        report = param_report(model)
        for key in ["total_trainable", "n_tensors", "shared_trainable",
                    "n_shared_tensors", "fingerprint", "shared_fingerprint"]:
            suite.check(f"param_report has {key}: {framework}", key in report)
        suite.check(f"total trainable is 18,254: {framework}",
                    report["total_trainable"] == 18254, str(report["total_trainable"]))
        suite.check(f"shared equals total (no extra neuron params): {framework}",
                    report["shared_trainable"] == report["total_trainable"],
                    f"{report['shared_trainable']} vs {report['total_trainable']}")


def test_verify_cross_framework_init() -> None:
    reports = {}
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        seed_model_init(cfg.SEED)
        reports[framework] = param_report(
            build_network(lif_factory(framework, cfg), cfg)
        )
    passed, problems = verify_cross_framework_init(reports)
    suite.check("the real four-framework init passes the gate", passed, str(problems))

    broken = {k: dict(v) for k, v in reports.items()}
    broken["sinabs"]["shared_fingerprint"] = "0000000000000000"
    failed, problems = verify_cross_framework_init(broken)
    suite.check("a mismatched fingerprint fails the gate", not failed)
    suite.check("and the problem names the fingerprint",
                any("FINGERPRINT" in p for p in problems), str(problems))

    broken2 = {k: dict(v) for k, v in reports.items()}
    broken2["sinabs"]["shared_trainable"] = 18257
    failed2, problems2 = verify_cross_framework_init(broken2)
    suite.check("a mismatched parameter count fails the gate", not failed2)
    suite.check("and the problem names the count",
                any("COUNT" in p for p in problems2), str(problems2))

    empty_ok, empty_problems = verify_cross_framework_init({})
    suite.check("no reports is a failure, not a pass", not empty_ok, str(empty_problems))


# ---------------------------------------------------------------------------
# 3. data-order seeding, at the generator level
# ---------------------------------------------------------------------------
def test_loader_seed_is_offset_from_the_model_seed() -> None:
    """Batch order and weight init must not share a stream, or changing one silently
    changes the other."""
    suite.check("loader seed is offset", loader_seed(0) == LOADER_SEED_OFFSET,
                str(loader_seed(0)))
    suite.check("the offset is large enough to avoid collisions",
                LOADER_SEED_OFFSET >= 1000, str(LOADER_SEED_OFFSET))
    suite.check("distinct model seeds give distinct loader seeds",
                loader_seed(0) != loader_seed(1))


def test_generators_are_reproducible_and_distinct() -> None:
    def draw(generator) -> list[int]:
        return torch.randperm(20, generator=generator).tolist()

    suite.check("loader_generator is reproducible",
                draw(loader_generator(0)) == draw(loader_generator(0)))
    suite.check("a different seed gives a different order",
                draw(loader_generator(0)) != draw(loader_generator(1)))
    suite.check("split_generator is reproducible",
                draw(split_generator(0)) == draw(split_generator(0)))
    suite.check("loader and split streams differ at the same seed",
                draw(loader_generator(0)) != draw(split_generator(0)))


def test_loader_generator_is_isolated_from_the_global_rng() -> None:
    generator = loader_generator(0)
    baseline = torch.randperm(20, generator=generator).tolist()
    torch.manual_seed(999)
    torch.rand(500)
    again = torch.randperm(20, generator=loader_generator(0)).tolist()
    suite.check("global RNG use does not disturb the loader stream", baseline == again)


def test_reset_loader_order() -> None:
    class FakeLoader:
        def __init__(self, generator=None):
            self.generator = generator

    class FakeWrapper:
        def __init__(self, inner):
            self.loader = inner

    generator = loader_generator(0)
    first = torch.randperm(20, generator=generator).tolist()
    loader = FakeLoader(generator)
    suite.check("reset_loader_order reports success", reset_loader_order(loader, 0) is True)
    suite.check("the next pass replays the identical order",
                torch.randperm(20, generator=loader.generator).tolist() == first)

    wrapped = FakeWrapper(FakeLoader(loader_generator(0)))
    suite.check("it reaches through a PrefetchedLoader-style wrapper",
                reset_loader_order(wrapped, 0) is True)
    suite.check("an unshuffled loader reports False, not an error",
                reset_loader_order(FakeLoader(None), 0) is False)


def test_seed_everything_covers_all_three_rngs() -> None:
    """torchvision's RandomRotation draws from torch, random_split from torch, and
    tonic's transforms can reach for numpy -- so all three have to be seeded."""
    seed_everything(0)
    drawn = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(0)
    again = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    suite.check("seed_everything reproduces stdlib random", drawn[0] == again[0])
    suite.check("seed_everything reproduces numpy", drawn[1] == again[1])
    suite.check("seed_everything reproduces torch", drawn[2] == again[2])

    seed_everything(1)
    different = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    suite.check("a different seed changes all three", different != drawn)


def test_determinism_flag_is_not_forced() -> None:
    """torch.use_deterministic_algorithms() is deliberately NOT set: it forces slower
    kernels and would change the very latency numbers this pipeline measures."""
    seed_everything(0)
    suite.check("deterministic algorithms are not forced on",
                torch.are_deterministic_algorithms_enabled() is False)


def main() -> int:
    return suite.run([
        test_same_seed_reproduces_within_a_framework,
        test_different_seed_changes_the_weights,
        test_all_frameworks_share_one_fingerprint_per_seed,
        test_seeding_is_immune_to_prior_rng_use,
        test_building_one_framework_does_not_shift_the_next,
        test_shared_fingerprint_ignores_module_naming,
        test_fingerprint_tracks_a_weight_change,
        test_param_report_contents,
        test_verify_cross_framework_init,
        test_loader_seed_is_offset_from_the_model_seed,
        test_generators_are_reproducible_and_distinct,
        test_loader_generator_is_isolated_from_the_global_rng,
        test_reset_loader_order,
        test_seed_everything_covers_all_three_rngs,
        test_determinism_flag_is_not_forced,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
