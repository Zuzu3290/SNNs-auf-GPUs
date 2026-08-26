"""Map this pipeline's objects onto the results schema.

Kept separate from skeleton/results.py so that module stays a pure schema+writer with no
knowledge of trainers, testers or models -- the glue lives here.

Every lookup is defensive (`.get`, `getattr`). A metric this pipeline does not measure
becomes None, which `results.append_row` writes as an empty cell. That is deliberate: a
CPU run with no NVML, or a framework whose ActivityMonitor recorded nothing, must still
produce a readable row rather than raising at the very end of a long run.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import torch

from skeleton.results import environment_info, make_run_id

GB_TO_MB = 1024.0


def _pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def _last(seq) -> Any:
    return seq[-1] if seq else None


def build_epoch_rows(epoch_log: list[dict]) -> list[dict]:
    """One row per epoch, from SNNTrainer.epoch_log.

    test_loss and test_accuracy_pct stay EMPTY: this pipeline evaluates once at the end
    rather than per epoch, so there is no honest per-epoch test figure to record. The
    learning-curve figures read train_accuracy_pct and tolerate the test columns being
    absent.
    """
    rows = []
    for entry in epoch_log:
        rows.append({
            "epoch": entry.get("epoch"),
            "train_loss": entry.get("train_loss"),
            "train_accuracy_pct": _pct(entry.get("train_accuracy")),
            "test_loss": None,
            "test_accuracy_pct": None,
            "epoch_train_time_s": entry.get("epoch_duration_s"),
            "spike_rate_pct": _pct(entry.get("spike_rate")),
        })
    return rows


def build_layer_rows(model, activity_snapshot: dict | None = None) -> list[dict]:
    """One row per spiking layer.

    Two sources, in order of preference:

      1. BaseLIF's own spike counters, when spike counting was switched on. Covers every
         layer including lif_out, and is the same measurement SNNs_2 records.
      2. The ActivityMonitor snapshot the trainer already collects. Free -- no extra pass
         -- but it only covers the HOOKED layers (lif1, lif2), not the output layer.

    Returns an empty list when neither is available, which simply means no layers.csv
    rows for this run rather than a failure.
    """
    rows: list[dict] = []
    named = model.net.named_lif_layers() if hasattr(model, "net") else {}

    for index, (name, layer) in enumerate(named.items()):
        neurons = layer.neurons() if hasattr(layer, "neurons") else 0
        slots = getattr(layer, "spike_slots", 0)
        total = getattr(layer, "spike_total", 0.0)
        if isinstance(total, torch.Tensor):
            total = float(total.item())

        if slots:  # source 1: the layer counted its own spikes
            rows.append({
                "layer_index": index,
                "layer_type": f"{name}:{type(layer).__name__}",
                "neurons": neurons,
                "total_spikes": total,
                "opportunities": slots,
                "spike_rate_pct": (total / slots) * 100.0 if slots else None,
            })
            continue

        # source 2: the monitor's recording for this layer, if it has one
        recorded = (activity_snapshot or {}).get(name)
        if recorded is None:
            continue
        opportunities = int(recorded.numel())
        spikes = float(recorded.float().sum().item())
        per_sample = recorded.shape[2:] if recorded.dim() > 2 else ()
        neuron_count = 1
        for dim in per_sample:
            neuron_count *= int(dim)
        rows.append({
            "layer_index": index,
            "layer_type": f"{name}:{type(layer).__name__}",
            "neurons": neuron_count or None,
            "total_spikes": spikes,
            "opportunities": opportunities,
            "spike_rate_pct": (spikes / opportunities) * 100.0 if opportunities else None,
        })
    return rows


def build_run_row(
    cfg,
    wf,
    model,
    run_info: dict,
    train_results: dict,
    test_results: dict,
    epoch_log: list[dict],
    params: dict,
    timesteps: int | None = None,
    num_workers: int | None = None,
    notes: str = "",
    run_id: str | None = None,
    latency: dict | None = None,
) -> dict:
    """One row summarising the whole run.

    `latency` is utilities.measure_latency()'s dict -- REAL batch-size-1 timings. None
    leaves those three columns empty rather than filling them with the amortised
    per-sample figure, which is a different measurement (see below).
    """
    framework = cfg.FRAMEWORK
    seed = getattr(cfg, "SEED", None)

    # Energy and power come from the per-epoch log, which is where this pipeline records
    # them. Summed over epochs for energy, averaged for the idle baselines.
    energy_total = sum(e.get("energy_j_total") or 0.0 for e in epoch_log) or None
    energy_dynamic = sum(e.get("energy_j_dynamic") or 0.0 for e in epoch_log) or None
    train_time = sum(e.get("epoch_duration_s") or 0.0 for e in epoch_log) or None
    idle_values = [e.get("idle_power_w") for e in epoch_log if e.get("idle_power_w")]

    neuron = model.describe_neuron() if hasattr(model, "describe_neuron") else {}
    env = environment_info(framework)

    row = {
        "run_id": run_id or make_run_id(framework, seed),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "framework": framework,
        "seed": seed,
        "config_path": run_info.get("config_path"),
        "config_hash": run_info.get("config_hash"),

        "dataset": getattr(cfg, "DATASET_NAME", None),
        "time_steps": timesteps if timesteps is not None else getattr(wf, "N_TIME_BINS", None),
        "batch_size": getattr(cfg, "BATCH_SIZE", None),
        "num_workers": num_workers,
        "binarize": getattr(wf, "BINARIZE", None),
        "denoise_us": getattr(wf, "DENOISE_FILTER_TIME_US", None),
        "epochs": getattr(cfg, "EPOCHS", None),
        "optimizer": getattr(cfg, "OPTIMIZER", None),
        "lr": getattr(cfg, "LEARNING_RATE", None),
        "surrogate": neuron.get("surrogate"),

        "trainable_params": params.get("total_trainable"),
        "weight_fingerprint": params.get("shared_fingerprint"),

        "test_accuracy_pct": _pct(test_results.get("overall_accuracy")),
        "train_loss_final": _last(train_results.get("loss_history") or []),
        # This pipeline does not compute a test LOSS, only accuracy -- left empty rather
        # than filled with something that is not it.
        "test_loss_final": None,

        "train_time_s": train_time,
        "train_time_per_epoch_s": (train_time / len(epoch_log)) if train_time and epoch_log else None,
        "inference_throughput_samples_per_s": test_results.get("throughput_samples_per_s"),
        # REAL batch-size-1 measurements, from utilities.measure_latency -- one sample at
        # a time with a synchronise around each, the MLPerf Single-Stream convention.
        #
        # These used to be filled from the test pass's per-sample figures, which are a
        # BATCH time divided by the batch size: throughput under batching, not latency.
        # It is systematically optimistic (a single arriving event cannot use that
        # parallelism) and its p90 describes batch-to-batch variation, since every sample
        # in a batch carries the same divided value. Same column names as SNNs_2, and now
        # the same measurement behind them.
        "inference_latency_bs1_ms": (latency or {}).get("latency_ms"),
        "inference_latency_bs1_mean_ms": (latency or {}).get("latency_mean_ms"),
        "inference_latency_bs1_p90_ms": (latency or {}).get("latency_p90_ms"),

        "spike_rate_pct": _pct(_last(train_results.get("spike_rate_history") or [])),

        # Only a peak for the inference phase is reported by this pipeline, and in GB.
        "peak_memory_train_mb": None,
        "peak_memory_infer_mb": (test_results.get("gpu_mem_peak_gb") or 0) * GB_TO_MB
                                 if test_results.get("gpu_mem_peak_gb") else None,
        "peak_reserved_train_mb": None,
        "peak_reserved_infer_mb": (test_results.get("max_memory_reserved_gb") or 0) * GB_TO_MB
                                   if test_results.get("max_memory_reserved_gb") else None,

        "nvml_update_interval_ms": None,
        "idle_power_cold_w": idle_values[0] if idle_values else None,
        "idle_power_after_train_w": idle_values[-1] if idle_values else None,
        "train_energy_duration_s": train_time,
        "train_energy_j": energy_total,
        "train_energy_dynamic_j": energy_dynamic,
        "energy_warnings": None,

        "notes": notes,
    }
    row.update(env)
    return row
