"""The results of a run: the schema of runs.csv, epochs.csv and layers.csv, the rows built from the trainer and tester, and the writer."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import pynvml
import sys
from datetime import datetime
from pathlib import Path

import torch

SCHEMA_VERSION = 4
GB_TO_MB = 1024.0


class ResultsError(Exception):
    """Raised when writing results would corrupt an existing file."""


# Changing a column list means bumping SCHEMA_VERSION; a metric that was not measured is an empty cell, never a missing column.
RUN_COLUMNS: list[str] = [
    "schema_version", "run_id", "timestamp", "framework", "seed", "config_hash",
    "dataset", "time_steps", "batch_size", "num_workers", "denoise_us", "epochs", "optimizer", "lr", "surrogate",
    "trainable_params", "weight_fingerprint",
    "test_accuracy_pct", "train_loss_final",
    "train_time_s", "inference_throughput_samples_per_s",
    "inference_latency_bs1_ms", "inference_latency_bs1_p90_ms",
    "train_forward_latency_ms", "train_backward_latency_ms",
    "conv_blocks", "fc_hidden",
    "spike_rate_pct", "test_activation_density_pct", "test_output_spike_rate_pct",
    "infer_synops_energy_per_sample_pj", "train_synops_energy_pj_total",
    "peak_memory_train_mb", "peak_memory_infer_mb",
    "idle_power_cold_w", "idle_power_after_train_w",
    "train_energy_j", "train_energy_dynamic_j", "infer_energy_j", "infer_energy_dynamic_j",
    "gpu_name", "driver", "cuda", "torch_version", "framework_version", "python_version", "platform",
    "notes",
]

EPOCH_COLUMNS: list[str] = [
    "schema_version", "run_id", "epoch", "train_loss", "train_accuracy_pct",
    "epoch_train_time_s", "spike_rate_pct", "forward_latency_ms", "backward_latency_ms", "synops_energy_pj",
]

LAYER_COLUMNS: list[str] = [
    "schema_version", "run_id", "layer_index", "layer_type", "neurons", "total_spikes", "opportunities", "spike_rate_pct",
    "dense_macs_downstream",
]

FRAMEWORK_PACKAGES = {"torch": "snntorch", "norse": "norse", "sj": "spikingjelly", "sinabs": "sinabs"}


def make_run_id(framework: str, seed: int, when=None) -> str:
    """`<timestamp>_<framework>_seed<n>`: unique, sortable, self-describing."""
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{framework}_seed{seed}"


def environment_info(framework: str) -> dict:
    """GPU, driver, CUDA and library versions the run actually ran on."""
    info: dict = {"torch_version": torch.__version__, "python_version": sys.version.split()[0],
                            "platform": platform.platform(), "gpu_name": None, "driver": None, "cuda": None,
                            "framework_version": None}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        info["driver"] = driver.decode() if isinstance(driver, bytes) else driver
    package = FRAMEWORK_PACKAGES.get(framework)
    if package:
        info["framework_version"] = importlib.metadata.version(package)
    return info


def pct(value):
    return None if value is None else float(value) * 100.0


def last(seq):
    return seq[-1] if seq else None


def numeric_over_epochs(epoch_log: list[dict], key: str) -> list[float]:
    return [float(v) for v in (row.get(key) for row in (epoch_log or [])) if isinstance(v, (int, float))]


def peak_over_epochs(epoch_log: list[dict], key: str):
    """Largest per-epoch value of `key` (GB), in MB."""
    values = numeric_over_epochs(epoch_log, key)
    return max(values) * GB_TO_MB if values else None


def sum_over_epochs(epoch_log: list[dict], key: str):
    values = numeric_over_epochs(epoch_log, key)
    return sum(values) if values else None


def mean_over_epochs(epoch_log: list[dict], key: str):
    values = numeric_over_epochs(epoch_log, key)
    return sum(values) / len(values) if values else None


def build_epoch_rows(epoch_log: list[dict]) -> list[dict]:
    """One row per epoch from SNNTrainer.epoch_log."""
    return [{
        "epoch": entry.get("epoch"),
        "train_loss": entry.get("train_loss"),
        "train_accuracy_pct": pct(entry.get("train_accuracy")),
        "epoch_train_time_s": entry.get("epoch_duration_s"),
        "spike_rate_pct": pct(entry.get("spike_rate")),
        "forward_latency_ms": entry.get("forward_latency_ms"),
        "backward_latency_ms": entry.get("backward_latency_ms"),
        "synops_energy_pj": entry.get("synops_energy_pj"),
    } for entry in epoch_log]


def layer_row(index: int, name: str, layer, neurons, spikes: float, slots: int, dense_macs) -> dict:
    return {
        "layer_index": index,
        "layer_type": f"{name}:{type(layer).__name__}",
        "neurons": neurons,
        "total_spikes": spikes,
        "opportunities": slots,
        "spike_rate_pct": (spikes / slots) * 100.0 if slots else None,
        "dense_macs_downstream": (dense_macs or {}).get(name),
    }


def build_layer_rows(model, activity_snapshot=None,
                     dense_macs=None) -> list[dict]:
    """One row per spiking layer, from the layer's own spike counter or else the ActivityMonitor recording."""
    rows: list[dict] = []
    source = getattr(model, "net", None) or model
    named = source.named_lif_layers() if hasattr(source, "named_lif_layers") else {}
    for index, (name, layer) in enumerate(named.items()):
        slots = getattr(layer, "spike_slots", 0)
        total = getattr(layer, "spike_total", 0.0)
        if isinstance(total, torch.Tensor):
            total = float(total.item())
        if slots:
            neurons = layer.neurons() if hasattr(layer, "neurons") else 0
            rows.append(layer_row(index, name, layer, neurons, total, slots, dense_macs))
            continue
        recorded = (activity_snapshot or {}).get(name)
        if recorded is None:
            continue
        neurons = 1
        for dim in (recorded.shape[2:] if recorded.dim() > 2 else ()):
            neurons *= int(dim)
        rows.append(layer_row(index, name, layer, neurons or None, float(recorded.float().sum().item()),
                              int(recorded.numel()), dense_macs))
    return rows


def build_run_row(cfg, model, train_results: dict, test_results: dict, epoch_log: list[dict],
                  params: dict, timesteps=None, num_workers=None, notes: str = "",
                  run_id=None, latency=None, density=None) -> dict:
    """One row summarising the run; `latency` is the real batch-size-1 timing from utilities.measure_latency."""
    framework = cfg.FRAMEWORK
    seed = getattr(cfg, "SEED", None)
    energy_total = sum(e.get("energy_j_total") or 0.0 for e in epoch_log) or None
    energy_dynamic = sum(e.get("energy_j_dynamic") or 0.0 for e in epoch_log) or None
    train_time = sum(e.get("epoch_duration_s") or 0.0 for e in epoch_log) or None
    idle_values = [e.get("idle_power_w") for e in epoch_log if e.get("idle_power_w")]
    neuron = model.describe_neuron() if hasattr(model, "describe_neuron") else {}
    accuracy = test_results.get("overall_accuracy")
    infer_peak = test_results.get("gpu_mem_peak_gb")
    row = {
        "run_id": run_id or make_run_id(framework, seed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": framework,
        "seed": seed,
        "config_hash": cfg.CONFIG_HASH,
        "dataset": getattr(cfg, "DATASET_NAME", None),
        "time_steps": timesteps if timesteps is not None else cfg.N_TIME_BINS,
        "batch_size": getattr(cfg, "BATCH_SIZE", None),
        "num_workers": num_workers,
        "denoise_us": cfg.DENOISE_FILTER_TIME_US,
        "epochs": getattr(cfg, "EPOCHS", None),
        "optimizer": getattr(cfg, "OPTIMIZER", None),
        "lr": getattr(cfg, "LEARNING_RATE", None),
        "surrogate": neuron.get("surrogate"),
        "trainable_params": params.get("total_trainable"),
        "weight_fingerprint": params.get("shared_fingerprint"),
        "test_accuracy_pct": pct(accuracy),
        "train_loss_final": last(train_results.get("loss_history") or []),
        "train_time_s": train_time,
        "inference_throughput_samples_per_s": test_results.get("throughput_samples_per_s"),
        "inference_latency_bs1_ms": (latency or {}).get("latency_ms"),
        "inference_latency_bs1_p90_ms": (latency or {}).get("latency_p90_ms"),
        "train_forward_latency_ms": mean_over_epochs(epoch_log, "forward_latency_ms"),
        "train_backward_latency_ms": mean_over_epochs(epoch_log, "backward_latency_ms"),
        "conv_blocks": "-".join(f"{b['out']}C{b['kernel']}" for b in getattr(cfg, "CONV_BLOCKS", []) or []) or None,
        "fc_hidden": getattr(cfg, "FC_HIDDEN", None),
        "spike_rate_pct": pct(last(train_results.get("spike_rate_history") or [])),
        "test_activation_density_pct": pct((density or {}).get("hidden_mean")),
        "test_output_spike_rate_pct": pct(test_results.get("output_spike_rate")),
        "infer_synops_energy_per_sample_pj": test_results.get("synops_energy_per_sample_pj"),
        "train_synops_energy_pj_total": sum_over_epochs(epoch_log, "synops_energy_pj"),
        "peak_memory_train_mb": peak_over_epochs(epoch_log, "gpu_mem_peak_gb"),
        "peak_memory_infer_mb": infer_peak * GB_TO_MB if infer_peak else None,
        "idle_power_cold_w": idle_values[0] if idle_values else None,
        "idle_power_after_train_w": idle_values[-1] if idle_values else None,
        "train_energy_j": energy_total,
        "train_energy_dynamic_j": energy_dynamic,
        "infer_energy_j": test_results.get("gpu_energy_j_total"),
        "infer_energy_dynamic_j": test_results.get("gpu_energy_j_dynamic"),
        "notes": notes,
    }
    row.update(environment_info(framework))
    return row


def format_cell(value) -> str:
    """None becomes an empty cell; everything else its plain text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def append_row(path: Path, columns: list[str], row: dict) -> None:
    """Append one row; an unknown key or a header that differs from the file on disk raises."""
    unknown = sorted(set(row) - set(columns))
    if unknown:
        raise ResultsError(f"unknown column(s) for {path.name}: {unknown}. Add them to the schema in "
                           "skeleton/results_collection.py and bump SCHEMA_VERSION.")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_header = None
    if path.is_file() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), None)
    if existing_header is not None and existing_header != columns:
        missing = sorted(set(columns) - set(existing_header))
        extra = sorted(set(existing_header) - set(columns))
        raise ResultsError(f"{path} was written with a different schema, so appending would misalign every row.\n"
                           f"  expected now but absent on disk: {missing or 'none'}\n"
                           f"  on disk but no longer expected:  {extra or 'none'}\n"
                           "Rename or move the old file to keep it, then re-run.")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if existing_header is None:
            writer.writerow(columns)
        writer.writerow([format_cell(row.get(column)) for column in columns])


def write_run_json(results_dir: Path, run_id: str, payload: dict) -> Path:
    """The complete record of one run, including the full config."""
    path = results_dir / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_results(results_dir: Path, run_row: dict, epoch_rows: list[dict],
                  layer_rows: list[dict], json_payload=None) -> dict[str, Path]:
    """Append the run's rows to runs.csv, epochs.csv and layers.csv, and write its JSON."""
    results_dir = Path(results_dir)
    run_id = run_row["run_id"]
    for row in list(epoch_rows) + list(layer_rows):
        row.setdefault("run_id", run_id)
        row.setdefault("schema_version", SCHEMA_VERSION)
    run_row.setdefault("schema_version", SCHEMA_VERSION)
    paths = {"runs": results_dir / "runs.csv", "epochs": results_dir / "epochs.csv", "layers": results_dir / "layers.csv"}
    append_row(paths["runs"], RUN_COLUMNS, run_row)
    for row in epoch_rows:
        append_row(paths["epochs"], EPOCH_COLUMNS, row)
    for row in layer_rows:
        append_row(paths["layers"], LAYER_COLUMNS, row)
    if json_payload is not None:
        paths["json"] = write_run_json(results_dir, run_id, json_payload)
    return paths
