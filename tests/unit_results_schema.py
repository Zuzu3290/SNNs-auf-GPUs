"""Unit tests for the layers.csv schema bump (v2 -> v3) that adds the 5 capacity-
metric columns.

    python tests/unit_results_schema.py

CPU-only, no dataset, no download, no model -- pure schema and CSV-writer checks.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from _harness import Suite
from skeleton.results import LAYER_COLUMNS, SCHEMA_VERSION, append_row
from skeleton.results_collect import build_layer_rows

suite = Suite("unit_results_schema")

NEW_COLUMNS = [
    "grad_norm_mean", "participation_ratio", "spike_entropy",
    "mutual_info_xz", "mutual_info_zy",
]


def test_schema_version_bumped() -> None:
    suite.check("SCHEMA_VERSION is 3", SCHEMA_VERSION == 3, f"got {SCHEMA_VERSION}")


def test_layer_columns_include_all_five_new_fields() -> None:
    missing = [c for c in NEW_COLUMNS if c not in LAYER_COLUMNS]
    suite.check("all 5 new columns are present", not missing, f"missing={missing}")


def test_layer_row_with_new_columns_writes_and_leaves_empty_when_none() -> None:
    """A row that doesn't set the new columns must still write, with empty cells --
    the existing 'empty cell, not a shifted header' guarantee."""
    tmp = Path(tempfile.mkdtemp(prefix="snn_unit_")) / "layers.csv"
    row_with_values = {c: None for c in LAYER_COLUMNS}
    row_with_values.update({
        "schema_version": SCHEMA_VERSION, "run_id": "test_run", "layer_index": 0,
        "layer_type": "lif1:Test", "grad_norm_mean": 0.5, "participation_ratio": 2.3,
        "spike_entropy": 1.1, "mutual_info_xz": 0.2, "mutual_info_zy": 0.3,
    })
    row_without_values = {c: None for c in LAYER_COLUMNS}
    row_without_values.update({
        "schema_version": SCHEMA_VERSION, "run_id": "test_run", "layer_index": 1,
        "layer_type": "lif2:Test",
    })
    append_row(tmp, LAYER_COLUMNS, row_with_values)
    append_row(tmp, LAYER_COLUMNS, row_without_values)
    text = tmp.read_text()
    lines = text.strip().splitlines()
    suite.check("header written once, two data rows follow", len(lines) == 3,
                f"got {len(lines)} lines")
    suite.check("second row's new columns are empty cells", ",,,," in lines[2]
                or lines[2].endswith(",,,,"), f"row2={lines[2]!r}")


class _FakeLIF:
    def __init__(self, spike_slots=0):
        self.spike_slots = spike_slots  # 0 -> build_layer_rows falls to the snapshot path

    def neurons(self):
        return 4


class _FakeNet:
    def named_lif_layers(self):
        return {"lif1": _FakeLIF(), "lif_out": _FakeLIF()}


class _FakeModel:
    def __init__(self):
        self.net = _FakeNet()


def test_build_layer_rows_includes_capacity_and_grad_norm_when_provided() -> None:
    import torch as _torch
    model = _FakeModel()
    activity_snapshot = {
        "lif1": _torch.zeros(2, 3, 4),   # [T, B, N]-shaped, any nonzero values fine
        "lif_out": _torch.ones(2, 3, 4),
    }
    capacity_metrics = {
        "lif1": {"participation_ratio": 1.5, "spike_entropy": 0.9,
                 "mutual_info_xz": 0.1, "mutual_info_zy": 0.2},
        "lif_out": {"participation_ratio": 2.5, "spike_entropy": 1.9,
                    "mutual_info_xz": 0.3, "mutual_info_zy": 0.4},
    }
    grad_norm_means = {"lif1": 0.05, "lif_out": 0.02}

    rows = build_layer_rows(model, activity_snapshot, capacity_metrics, grad_norm_means)

    suite.check("one row per named layer", len(rows) == 2, f"got {len(rows)}")
    by_type = {r["layer_type"]: r for r in rows}
    lif1_row = next(r for name, r in by_type.items() if name.startswith("lif1"))
    out_row = next(r for name, r in by_type.items() if name.startswith("lif_out"))
    suite.check("lif1 row carries its participation_ratio",
                lif1_row["participation_ratio"] == 1.5, f"got {lif1_row}")
    suite.check("lif_out row carries its grad_norm_mean",
                out_row["grad_norm_mean"] == 0.02, f"got {out_row}")


def test_build_layer_rows_leaves_new_columns_none_without_capacity_data() -> None:
    """Backward compatible: calling with no capacity_metrics/grad_norm_means args
    (as every pre-existing call site does) must still work and leave the new fields
    None."""
    import torch as _torch
    model = _FakeModel()
    activity_snapshot = {"lif1": _torch.zeros(2, 3, 4), "lif_out": _torch.ones(2, 3, 4)}
    rows = build_layer_rows(model, activity_snapshot)
    suite.check("rows still produced with no capacity args", len(rows) == 2)
    suite.check("new columns are None, not KeyError",
                all(r.get("participation_ratio") is None for r in rows))


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_schema_version_bumped,
        test_layer_columns_include_all_five_new_fields,
        test_layer_row_with_new_columns_writes_and_leaves_empty_when_none,
        test_build_layer_rows_includes_capacity_and_grad_norm_when_provided,
        test_build_layer_rows_leaves_new_columns_none_without_capacity_data,
    ]))
