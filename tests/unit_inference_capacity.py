"""Unit tests for SNNTester's capacity-metrics accumulation over the full test set.

    python tests/unit_inference_capacity.py

CPU-only, no real dataset -- a tiny synthetic loader stands in for the test set, per
the existing _harness.py conventions (build_model, fresh_cfg, spike_input).
"""
from __future__ import annotations

import torch

from _harness import Suite, build_model, fresh_cfg, spike_input
from learning.inference import SNNTester

suite = Suite("unit_inference_capacity")


def _capacity_test_cfg(compute_capacity_metrics: bool):
    cfg = fresh_cfg()
    cfg.COMPUTE_CAPACITY_METRICS = compute_capacity_metrics
    cfg.LATENCY_SAMPLES = 0  # skip the separate bs=1 latency pass, irrelevant here
    return cfg


def test_capacity_metrics_populated_over_the_whole_test_set() -> None:
    """With the flag on, running a multi-batch test loader must accumulate every
    batch into capacity_metrics, covering every hooked layer including lif_out --
    not just the last batch seen."""
    cfg = _capacity_test_cfg(compute_capacity_metrics=True)
    model, _ = build_model("sj", cfg)
    # 5 small batches, standing in for "however many batches calibration hands you" --
    # the accumulation must not assume one fixed batch size.
    loader = [
        (spike_input(cfg, time_steps=4, batch=4, seed=i),
         torch.randint(0, cfg.NUM_CLASSES, (4,)))
        for i in range(5)
    ]
    tester = SNNTester(model, loader, cfg, torch.device("cpu"))
    results = tester.run(csv_path="/tmp/snn_unit_inference_capacity_test.csv")

    capacity = results["capacity_metrics"]
    suite.check("capacity_metrics populated for every hooked layer",
                set(capacity.keys()) == set(model.net.named_lif_layers().keys()),
                f"got keys={list(capacity.keys())}")
    for name, values in capacity.items():
        needed = {"participation_ratio", "participation_ratio_normalized",
                  "spike_entropy", "spike_entropy_normalized", "mutual_info_zy"}
        suite.check(f"{name}: has all 5 capacity fields", needed <= set(values.keys()),
                    f"{name} -> {values}")
        suite.check(f"{name}: participation_ratio_normalized is in [0, ~1.5]",
                    0.0 <= values["participation_ratio_normalized"] <= 1.5,
                    f"got {values['participation_ratio_normalized']}")
    suite.check("tester.capacity_metrics attribute matches the returned dict",
                tester.capacity_metrics == capacity)


def test_run_resumes_recording_even_if_model_arrives_paused() -> None:
    """Regression test: in a real run, SNNTrainer.train() pauses model.activity and
    never resumes it (learning/training.py:362), and main.py hands that SAME model
    object to SNNTester -- so SNNTester.run() must resume recording itself, on entry,
    regardless of the incoming state. The other tests in this file all build a fresh,
    never-paused model, so they would pass even if run() relied on the caller having
    already resumed recording -- this test pauses the model BEFORE constructing
    SNNTester to specifically catch that gap."""
    cfg = _capacity_test_cfg(compute_capacity_metrics=True)
    model, _ = build_model("sj", cfg)
    model.activity.pause()  # simulate arriving straight out of SNNTrainer.train()
    suite.check("model really is paused before the test pass starts",
                model.activity.paused is True)

    loader = [
        (spike_input(cfg, time_steps=4, batch=4, seed=i),
         torch.randint(0, cfg.NUM_CLASSES, (4,)))
        for i in range(5)
    ]
    tester = SNNTester(model, loader, cfg, torch.device("cpu"))
    results = tester.run(csv_path="/tmp/snn_unit_inference_capacity_test_paused.csv")

    capacity = results["capacity_metrics"]
    suite.check("capacity_metrics still populated when the model arrived paused",
                set(capacity.keys()) == set(model.net.named_lif_layers().keys())
                and len(capacity) > 0,
                f"got keys={list(capacity.keys())}")
    for name, values in capacity.items():
        needed = {"participation_ratio", "participation_ratio_normalized",
                  "spike_entropy", "spike_entropy_normalized", "mutual_info_zy"}
        suite.check(f"{name}: has all 5 capacity fields when model arrived paused",
                    needed <= set(values.keys()), f"{name} -> {values}")
    # cv_isi_mean reads from the same recordings() call as capacity metrics -- this
    # was silently broken by the exact same bug (see empirical evidence in
    # experiments/ex6/results/runs/20260902_164726_sj_seed0.json, where it and
    # total_synops_energy_pj are both exactly 0.0 despite nonzero total_spikes).
    suite.check("cv_isi_mean is also populated when the model arrived paused",
                results["cv_isi_mean"] != 0.0, f"got {results['cv_isi_mean']}")
    suite.check("model's paused state is restored to what it was on entry",
                model.activity.paused is True,
                f"expected still-paused (as it was before run()), got {model.activity.paused}")


def test_capacity_metrics_empty_when_flag_is_off() -> None:
    cfg = _capacity_test_cfg(compute_capacity_metrics=False)
    model, _ = build_model("sj", cfg)
    loader = [
        (spike_input(cfg, time_steps=4, batch=4, seed=i),
         torch.randint(0, cfg.NUM_CLASSES, (4,)))
        for i in range(3)
    ]
    tester = SNNTester(model, loader, cfg, torch.device("cpu"))
    results = tester.run(csv_path="/tmp/snn_unit_inference_capacity_test_off.csv")
    suite.check("capacity_metrics stays empty when the flag is off",
                results["capacity_metrics"] == {})


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_capacity_metrics_populated_over_the_whole_test_set,
        test_run_resumes_recording_even_if_model_arrives_paused,
        test_capacity_metrics_empty_when_flag_is_off,
    ]))
