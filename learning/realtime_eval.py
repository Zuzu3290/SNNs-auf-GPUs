"""
Real-time suitability evaluation — p99 (tail) latency vs. a per-dataset
deadline, across multiple seeds. See docs/frameworks/realtime_nir_evaluation.md.

Wraps SNNTester rather than re-measuring latency itself: SNNTester.run()
already times every batch and reports p50/p90/p99 per-sample latency. What
that single run can't answer is whether the deadline holds up reliably
across different random initialisations, not just on the one run that
happened to be measured.
"""
from __future__ import annotations

import csv
import os
import statistics
import torch
from skeleton import Settings
from event_data_workflow.workflow_config import WorkflowSettings
from learning.inference import SNNTester


class RealTimeLatencyEvaluator:
    """
    seeds: model_factory is called once per seed, after torch.manual_seed(seed)
    — a fresh model per seed, matching the seed semantics already documented
    in CLAUDE.md (seed reproduces weight initialisation, network_architecture.yaml
    only fixes shapes). Latency is largely independent of weight *values* for a
    fixed architecture, so this mainly measures cross-run system jitter plus
    any weight-dependent variance; it is not a substitute for timing the exact
    checkpoint that will actually ship. Pass a model_factory that loads a
    trained checkpoint if that distinction matters for your use.
    """

    def __init__(self, model_factory, test_loader, cfg: Settings, device: torch.device, dataset_name: str | None = None):
        self.model_factory = model_factory
        self.test_loader   = test_loader
        self.cfg           = cfg
        self.device        = device
        self.wf            = WorkflowSettings()
        self.dataset_name  = dataset_name or cfg.DATASET_NAME
        self.deadline_ms   = self.wf.REALTIME_DEADLINE_MS.get(self.dataset_name)

    def evaluate(self, seeds: tuple[int, ...] = (0, 1, 2), csv_dir: str = "./outputs/data/realtime") -> dict:
        if self.deadline_ms is None:
            raise ValueError(
                f"No real-time deadline configured for dataset '{self.dataset_name}' — "
                f"add it to configuration/data_workflow.yaml under realtime.deadline_ms. "
                f"Configured datasets: {list(self.wf.REALTIME_DEADLINE_MS)}"
            )

        os.makedirs(csv_dir, exist_ok=True)
        per_seed = []

        for seed in seeds:
            torch.manual_seed(seed)
            model  = self.model_factory()
            tester = SNNTester(model, self.test_loader, self.cfg, self.device)
            result = tester.run(csv_path=f"{csv_dir}/seed_{seed}.csv")
            p99    = result["p99_latency_per_sample_ms"]
            per_seed.append({
                "seed":           seed,
                "p50_ms":         result["median_latency_per_sample_ms"],
                "p90_ms":         result["p90_latency_per_sample_ms"],
                "p99_ms":         p99,
                "accuracy":       result["overall_accuracy"],
                "meets_deadline": p99 <= self.deadline_ms,
            })

        p99_values = [r["p99_ms"] for r in per_seed]
        summary = {
            "dataset":                self.dataset_name,
            "deadline_ms":            self.deadline_ms,
            "num_seeds":              len(seeds),
            "p99_mean_ms":            statistics.mean(p99_values),
            "p99_stdev_ms":           statistics.stdev(p99_values) if len(p99_values) > 1 else 0.0,
            "p99_worst_ms":           max(p99_values),
            "seeds_meeting_deadline": sum(r["meets_deadline"] for r in per_seed),
            "seeds_total":            len(seeds),
            "verdict":                "PASS" if all(r["meets_deadline"] for r in per_seed) else "FAIL",
            "per_seed":               per_seed,
        }

        self.print_report(summary)
        self.write_summary_csv(summary, f"{csv_dir}/summary.csv")
        return summary

    def print_report(self, summary: dict) -> None:
        print("\n[REAL-TIME SUITABILITY]")
        print(f"  Dataset          : {summary['dataset']}")
        print(f"  Deadline (p99)   : {summary['deadline_ms']:.1f} ms")
        print(f"  Seeds run        : {summary['num_seeds']}")
        for r in summary["per_seed"]:
            status = "PASS" if r["meets_deadline"] else "FAIL"
            print(f"    seed {r['seed']:>3} | p50 {r['p50_ms']:.2f}ms  p90 {r['p90_ms']:.2f}ms  "
                  f"p99 {r['p99_ms']:.2f}ms  acc {r['accuracy'] * 100:.1f}%  [{status}]")
        print(f"  p99 mean ± stdev : {summary['p99_mean_ms']:.2f} ± {summary['p99_stdev_ms']:.2f} ms")
        print(f"  p99 worst seed   : {summary['p99_worst_ms']:.2f} ms")
        print(f"  Verdict          : {summary['verdict']} "
              f"({summary['seeds_meeting_deadline']}/{summary['seeds_total']} seeds meet the deadline)")

    def write_summary_csv(self, summary: dict, path: str) -> None:
        rows = summary["per_seed"]
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Real-time evaluation summary saved -> {path}")
