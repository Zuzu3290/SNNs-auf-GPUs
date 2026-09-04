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


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_schema_version_bumped,
        test_layer_columns_include_all_five_new_fields,
        test_layer_row_with_new_columns_writes_and_leaves_empty_when_none,
    ]))
