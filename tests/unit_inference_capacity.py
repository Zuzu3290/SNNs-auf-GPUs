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
        test_capacity_metrics_empty_when_flag_is_off,
    ]))
