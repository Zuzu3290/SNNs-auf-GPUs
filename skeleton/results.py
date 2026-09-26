"""Writing results: three CSVs plus one JSON per run.

Ported from the SNNs_2 comparison pipeline so that its plotting layer (F0-F6) can read
this pipeline's output unchanged. The schema below IS that contract -- the figures group
by `run_id`, `framework` and `seed`, so those three columns are what make a comparison
plottable at all.

ADDITIVE. This writes NEW files alongside the pipeline's own `training_results.csv` and
`test.csv`, which are untouched and keep their richer per-epoch columns. Nothing here
replaces or rewrites them.

Four guarantees, all inherited from the original:

  1. Fixed column set. A metric that could not be measured writes an EMPTY CELL, never a
     missing column -- so a CPU run with no NVML still produces a readable row.
  2. `schema_version` in every row.
  3. A header mismatch is a loud error, not a silent append that misaligns every
     subsequent row.
  4. Append-only. Rows are never rewritten, reordered or deduplicated.

Append-only is what makes the one-framework-per-invocation workflow work: run sinabs in
one Colab cell, snntorch in the next, change the seed and run again, and the rows
accumulate into one `runs.csv`. No multi-framework driver is needed, and the number of
seeds per framework does not have to be uniform -- one seed for one framework and three
for another is a valid file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 3


class ResultsError(Exception):
    """Raised when writing results would corrupt an existing file."""


# ---------------------------------------------------------------------------
# The schema. Change these and you must bump SCHEMA_VERSION.
# Kept column-for-column identical to SNNs_2 so its figures need no adapter.
# ---------------------------------------------------------------------------
RUN_COLUMNS: list[str] = [
    # identity
    "schema_version", "run_id", "timestamp", "framework", "seed",
    "config_path", "config_hash",
    # setup
    "dataset", "time_steps", "batch_size", "num_workers",
    "denoise_us", "epochs", "optimizer", "lr", "surrogate",
    # integrity -- the fairness evidence travels WITH the numbers
    "trainable_params", "weight_fingerprint",
    # task performance. test_accuracy_pct is the CLASSIFICATION score and stays empty
    # on a regression row; task_metric/task_score carry whichever score that
    # application actually has (rmse for pose, epe for flow, accuracy_pct for the two
    # classification sets) so one column can be plotted across all four.
    "test_accuracy_pct", "train_loss_final", "test_loss_final",
    "task_metric", "task_score", "task_score_secondary",
    # speed
    "train_time_s", "train_time_per_epoch_s",
    "inference_throughput_samples_per_s",
    "inference_latency_bs1_ms", "inference_latency_bs1_mean_ms",
    "inference_latency_bs1_p90_ms",
    "inference_latency_amortised_p50_ms", "inference_latency_amortised_p90_ms",
    "inference_latency_amortised_p99_ms",
    # TRAINING-time forward and backward, meaned over epochs. Separate from every
    # inference_* column above: those time a trained model answering, these time the
    # learning step, and the backward is the half that carries BPTT's unroll.
    "train_forward_latency_ms", "train_backward_latency_ms",
    # architecture -- no longer fixed, so a row must say what network produced it
    "neuron_model", "conv_blocks", "fc_hidden", "credit_assignment",
    # activity / internal dynamics. "activation density" is the fair name: for a
    # spiking layer it is the spike rate, for a ReLU layer the fraction of units
    # emitting anything. Same measurement either way -- see relu_activation.py.
    "spike_rate_pct", "test_activation_density_pct",
    "spikes_per_neuron_per_inference", "cv_isi_mean",
    "total_spikes_test", "input_to_output_ratio",
    # operation counts and the energy MODELS built on them (estimates, not
    # measurements -- the measured NVML joules are the block below)
    # SynOps kept SEPARATE for the two phases rather than merged. Training runs the
    # backward pass and many epochs; inference runs one forward pass on a trained
    # network. A single figure spanning both answers neither question -- the deployment
    # cost is the inference one, the cost of getting there is the training one.
    "energy_per_sample_pj",
    "infer_synops_energy_per_sample_pj", "train_synops_energy_pj_total",
    "train_synops_energy_pj_per_epoch",
    # memory
    "peak_memory_train_mb", "peak_memory_infer_mb",
    "peak_reserved_train_mb", "peak_reserved_infer_mb",
    # energy (training run only)
    "nvml_update_interval_ms", "idle_power_cold_w", "idle_power_after_train_w",
    "train_energy_duration_s", "train_energy_j", "train_energy_dynamic_j",
    "infer_energy_j", "infer_energy_dynamic_j",
    "energy_warnings",
    # runtime GPU diagnostics -- a latency or energy figure is not comparable
    # across rows without them (SNN_GPU_Evaluation_Metrics.md 2.4b)
    "gpu_temp_c", "sm_clock_mhz", "mem_clock_mhz",
    # environment
    "gpu_name", "driver", "cuda", "torch_version", "framework_version",
    "python_version", "platform",
    # free text
    "notes",
]

EPOCH_COLUMNS: list[str] = [
    "schema_version", "run_id", "epoch",
    "train_loss", "train_accuracy_pct",
    "test_loss", "test_accuracy_pct",
    "epoch_train_time_s", "spike_rate_pct",
    # Measured every epoch by SNNTrainer.measure_activity() and written to the run's own
    # training_results.csv, but previously stopping there -- so the per-epoch trend of
    # the three metrics the comparison is actually about could not be plotted across
    # frameworks. Forward and backward are reported separately because the backward pass
    # through T unrolled timesteps is where BPTT's cost lives, and it is roughly twice
    # the forward on every framework measured.
    "forward_latency_ms", "backward_latency_ms",
    "cv_isi_mean", "synops_energy_pj",
]

LAYER_COLUMNS: list[str] = [
    "schema_version", "run_id", "layer_index", "layer_type",
    "neurons", "total_spikes", "opportunities", "spike_rate_pct",
    # CV(ISI) PER LAYER, not only the network-wide mean. Averaging it across layers
    # hides the thing it exists to show: firing regularity changes with depth, and a
    # network whose first layer fires like a clock while its second bursts has the same
    # mean as one that is uniformly irregular. The per-layer figures are the diagnostic;
    # the mean is a summary of them, kept in runs.csv for one-number comparison only.
    "cv_isi", "dense_macs_downstream",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def make_run_id(framework: str, seed: int, when: datetime | None = None) -> str:
    """`<timestamp>_<framework>_seed<n>` -- unique, sortable, self-describing."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{framework}_seed{seed}"


def config_hash(config: dict[str, Any]) -> str:
    """Hash of the fully resolved config, so two runs stay distinguishable even if the
    config file is edited afterwards."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


FRAMEWORK_PACKAGES = {
    "torch": "snntorch", "snntorch": "snntorch",
    "norse": "norse", "sj": "spikingjelly", "spikingjelly": "spikingjelly",
    "sinabs": "sinabs",
}


def environment_info(framework: str) -> dict[str, Any]:
    """What the run actually ran on. Recorded per run because a latency or energy figure
    is meaningless without it -- the same code on another GPU is a different number."""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu_name": None,
        "driver": None,
        "cuda": None,
        "framework_version": None,
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        except Exception:
            pass
        try:
            import pynvml

            pynvml.nvmlInit()
            info["driver"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(info["driver"], bytes):
                info["driver"] = info["driver"].decode()
        except Exception:
            pass

    package = FRAMEWORK_PACKAGES.get(framework)
    if package:
        try:
            import importlib

            info["framework_version"] = getattr(
                importlib.import_module(package), "__version__", None
            )
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _format(value: Any) -> str:
    """None becomes an empty cell. Everything else becomes its plain text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def append_row(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    """Append one row, refusing to corrupt an existing file.

    Unknown keys are an error rather than being dropped: a typo in a column name would
    otherwise silently lose a measurement.
    """
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ResultsError(
            f"unknown column(s) for {path.name}: {unknown}. "
            "Add them to the schema in skeleton/results.py and bump SCHEMA_VERSION."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_header: list[str] | None = None
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), None)

    if existing_header is not None and existing_header != columns:
        missing = sorted(set(columns) - set(existing_header))
        extra = sorted(set(existing_header) - set(columns))
        raise ResultsError(
            f"{path} was written with a different schema, so appending would misalign "
            f"every row.\n"
            f"  expected now but absent on disk: {missing or 'none'}\n"
            f"  on disk but no longer expected:  {extra or 'none'}\n"
            "Rename or move the old file to keep it, then re-run."
        )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if existing_header is None:
            writer.writerow(columns)
        writer.writerow([_format(row.get(column)) for column in columns])


def write_run_json(results_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    """The complete record for one run: full config, versions, every metric.

    Exists because a run is only reproducible if the WHOLE config travels with its
    numbers, and a YAML stuffed into a CSV cell makes the CSV unreadable.
    """
    path = results_dir / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_results(
    results_dir: Path,
    run_row: dict[str, Any],
    epoch_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write all four artefacts for one run. Appends; never rewrites."""
    results_dir = Path(results_dir)
    run_id = run_row["run_id"]

    for row in list(epoch_rows) + list(layer_rows):
        row.setdefault("run_id", run_id)
        row.setdefault("schema_version", SCHEMA_VERSION)
    run_row.setdefault("schema_version", SCHEMA_VERSION)

    paths = {
        "runs": results_dir / "runs.csv",
        "epochs": results_dir / "epochs.csv",
        "layers": results_dir / "layers.csv",
    }
    append_row(paths["runs"], RUN_COLUMNS, run_row)
    for row in epoch_rows:
        append_row(paths["epochs"], EPOCH_COLUMNS, row)
    for row in layer_rows:
        append_row(paths["layers"], LAYER_COLUMNS, row)

    if json_payload is not None:
        paths["json"] = write_run_json(results_dir, run_id, json_payload)
    return paths
