"""The results of a run: the schema of training_runs.csv, inference_runs.csv, epochs.csv and layers.csv, the rows built from the trainer and tester, and the writer."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import logging
import platform
import pynvml
import sys
from datetime import datetime
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6  # 6: training SynOps columns dropped (SynOps is measured at inference); 5: runs.csv split into training_runs.csv and inference_runs.csv
GB_TO_MB = 1024.0


class ResultsError(Exception):
    """Raised when writing results would corrupt an existing file."""


ENVIRONMENT_COLUMNS: list[str] = [
    "gpu_name", "cpu_name", "driver", "cuda", "torch_version", "framework_version", "python_version", "platform",
]

# Changing a column list means bumping SCHEMA_VERSION; a metric that was not measured is an empty cell, never a missing column.
TRAINING_RUN_COLUMNS: list[str] = [
    "schema_version", "run_id", "timestamp", "seed", "config_hash",
    "dataset", "time_steps", "batch_size", "num_workers", "denoise_us", "epochs", "optimizer", "lr", "surrogate",
    "conv_blocks", "fc_hidden", "trainable_params", "weight_fingerprint",
    "train_loss_final", "train_time_s", "train_forward_latency_ms", "train_backward_latency_ms",
    "spike_rate_pct", "peak_memory_train_mb",
    "idle_power_cold_w", "idle_power_after_train_w", "train_energy_j", "train_energy_dynamic_j",
    *ENVIRONMENT_COLUMNS, "notes",
]

# One row per test pass; several can point at the same training run_id. GPU columns are empty when inference ran on the cpu.
INFERENCE_RUN_COLUMNS: list[str] = [
    "schema_version", "run_id", "timestamp", "inference_device", "dataset", "time_steps", "batch_size",
    "test_accuracy_pct", "inference_throughput_samples_per_s",
    "inference_latency_bs1_ms", "inference_latency_bs1_p90_ms",
    "test_activation_density_pct", "test_output_spike_rate_pct", "infer_synops_energy_per_sample_pj",
    "peak_memory_infer_mb", "infer_energy_j", "infer_energy_dynamic_j",
    *ENVIRONMENT_COLUMNS, "notes",
]

EPOCH_COLUMNS: list[str] = [
    "schema_version", "run_id", "epoch", "train_loss", "train_accuracy_pct",
    "epoch_train_time_s", "spike_rate_pct", "forward_latency_ms", "backward_latency_ms",
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
    """GPU, CPU, driver, CUDA and library versions the run actually ran on."""
    info: dict = {"torch_version": torch.__version__, "python_version": sys.version.split()[0],
                            "platform": platform.platform(), "cpu_name": platform.processor() or None,
                            "gpu_name": None, "driver": None, "cuda": None, "framework_version": None}
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
    } for entry in epoch_log]


def layer_row(index: int, name: str, layer, neurons, spikes: float, slots: int, dense_macs) -> dict:
    return {
        "layer_index": index,
        "layer_type": f"{name}:{type(layer).__module__.rsplit('.', 1)[-1]}",
        "neurons": neurons,
        "total_spikes": spikes,
        "opportunities": slots,
        "spike_rate_pct": (spikes / slots) * 100.0 if slots else None,
        "dense_macs_downstream": (dense_macs or {}).get(name),
    }


def build_layer_rows(model, dense_macs=None) -> list[dict]:
    """One row per spiking layer, from the spike counters filled by the activation-density pass."""
    rows: list[dict] = []
    for index, (name, layer) in enumerate(model.neuron_layers.items()):
        if not layer.spike_slots:
            continue
        total = layer.spike_total
        if isinstance(total, torch.Tensor):
            total = float(total.item())
        rows.append(layer_row(index, name, layer, layer.neurons(), total, layer.spike_slots, dense_macs))
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
    surrogate = neuron.get("surrogate")
    if isinstance(surrogate, dict):  # a block, e.g. {type: atan, alpha: 2.0} -> "atan(alpha=2.0)"; a plain value is written as it is
        args = {k: v for k, v in surrogate.items() if k != "type"}
        surrogate = f"{surrogate.get('type')}({', '.join(f'{k}={v}' for k, v in args.items())})"
    accuracy = test_results.get("overall_accuracy")
    inference_device = test_results.get("inference_device")
    on_gpu = inference_device == "cuda"  # on the cpu the GPU readings are only its idle draw, so they stay empty
    infer_peak = test_results.get("gpu_mem_peak_gb") if on_gpu else None
    row = {
        "inference_device": inference_device,
        "run_id": run_id or make_run_id(framework, seed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
        "surrogate": surrogate,
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
        "peak_memory_train_mb": peak_over_epochs(epoch_log, "gpu_mem_peak_gb"),
        "peak_memory_infer_mb": infer_peak * GB_TO_MB if infer_peak else None,
        "idle_power_cold_w": idle_values[0] if idle_values else None,
        "idle_power_after_train_w": idle_values[-1] if idle_values else None,
        "train_energy_j": energy_total,
        "train_energy_dynamic_j": energy_dynamic,
        "infer_energy_j": test_results.get("gpu_energy_j_total") if on_gpu else None,
        "infer_energy_dynamic_j": test_results.get("gpu_energy_j_dynamic") if on_gpu else None,
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
    """Append the run to training_runs.csv and inference_runs.csv (each takes its own columns), epochs.csv and layers.csv, and write its JSON."""
    results_dir = Path(results_dir)
    run_id = run_row["run_id"]
    for row in list(epoch_rows) + list(layer_rows):
        row.setdefault("run_id", run_id)
        row.setdefault("schema_version", SCHEMA_VERSION)
    run_row.setdefault("schema_version", SCHEMA_VERSION)
    paths = {"training": results_dir / "training_runs.csv", "inference": results_dir / "inference_runs.csv",
             "epochs": results_dir / "epochs.csv", "layers": results_dir / "layers.csv"}
    append_row(paths["training"], TRAINING_RUN_COLUMNS, {k: run_row.get(k) for k in TRAINING_RUN_COLUMNS})
    append_row(paths["inference"], INFERENCE_RUN_COLUMNS, {k: run_row.get(k) for k in INFERENCE_RUN_COLUMNS})
    for row in epoch_rows:
        append_row(paths["epochs"], EPOCH_COLUMNS, row)
    for row in layer_rows:
        append_row(paths["layers"], LAYER_COLUMNS, row)
    if json_payload is not None:
        paths["json"] = write_run_json(results_dir, run_id, json_payload)
    return paths


def record_run(results_dir: Path, cfg, model, run_id: str, train_results: dict, test_results: dict,
               epoch_log: list[dict], params: dict, num_workers, latency, density) -> dict[str, Path]:
    """Build every row for one finished run, write them all, and print where they went."""
    run_row = build_run_row(
        cfg, model, train_results=train_results, test_results=test_results, epoch_log=epoch_log,
        params=params, timesteps=test_results.get("timesteps"), num_workers=num_workers,
        latency=latency, density=density, run_id=run_id,
        notes=" ".join(f"{k}={v}" for k, v in cfg.OVERRIDES.items()),
    )
    paths = write_results(
        results_dir,
        run_row=run_row,
        epoch_rows=build_epoch_rows(epoch_log),
        layer_rows=build_layer_rows(model, dense_macs=test_results.get("dense_macs_per_layer")),
        json_payload={"run": run_row,
                      "neuron": model.describe_neuron(),
                      "test": test_results},
    )
    logger.info(f"results (run_id {run_id})\n"
                f"  training : {paths['training']}\n"
                f"  inference: {paths['inference']}\n"
                f"  epochs   : {paths['epochs']}\n"
                f"  layers   : {paths['layers']}\n"
                f"  record   : {paths['json']}")
    return paths
