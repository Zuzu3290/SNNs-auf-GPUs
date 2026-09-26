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


def _peak_over_epochs(epoch_log: list[dict], key: str) -> float | None:
    """The largest per-epoch value of `key`, in MB.

    SNNTrainer records GPU memory once per epoch, so the run-level peak is the max over
    those samples. Reported rather than left empty: these two columns were hardcoded
    None with a comment saying this pipeline only measures the inference phase, which
    stopped being true once the per-epoch GPU report was added -- the numbers were
    already in epoch_log and were being printed every epoch.
    """
    values = [row.get(key) for row in (epoch_log or [])]
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return max(numeric) * GB_TO_MB if numeric else None


def _sum_over_epochs(epoch_log: list[dict], key: str) -> float | None:
    """Total of a per-epoch measurement across the whole run."""
    values = [row.get(key) for row in (epoch_log or [])]
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(numeric) if numeric else None


def _mean_over_epochs(epoch_log: list[dict], key: str) -> float | None:
    """Mean of a per-epoch measurement. None when no epoch recorded it, so the column
    stays empty rather than reporting a zero that looks like a measurement."""
    values = [row.get(key) for row in (epoch_log or [])]
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


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
            "forward_latency_ms": entry.get("forward_latency_ms"),
            "backward_latency_ms": entry.get("backward_latency_ms"),
            "cv_isi_mean": entry.get("cv_isi_mean"),
            "synops_energy_pj": entry.get("synops_energy_pj"),
        })
    return rows


def build_layer_rows(model, activity_snapshot: dict | None = None,
                      cv_isi: dict | None = None,
                      dense_macs: dict | None = None) -> list[dict]:
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
    # The classification models keep their layers in .net; the dense ones (flow,
    # detection) hold them directly. Either answers named_lif_layers().
    source = getattr(model, "net", None) or model
    named = source.named_lif_layers() if hasattr(source, "named_lif_layers") else {}

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
                "cv_isi": (cv_isi or {}).get(name),
                "dense_macs_downstream": (dense_macs or {}).get(name),
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
            "cv_isi": (cv_isi or {}).get(name),
            "dense_macs_downstream": (dense_macs or {}).get(name),
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
    density: dict | None = None,
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
        "denoise_us": getattr(wf, "DENOISE_FILTER_TIME_US", None),
        "epochs": getattr(cfg, "EPOCHS", None),
        "optimizer": getattr(cfg, "OPTIMIZER", None),
        "lr": getattr(cfg, "LEARNING_RATE", None),
        "surrogate": neuron.get("surrogate"),

        "trainable_params": params.get("total_trainable"),
        "weight_fingerprint": params.get("shared_fingerprint"),

        "test_accuracy_pct": _pct(test_results.get("overall_accuracy")),
        # Filled by whichever entry point ran: accuracy for classification, rmse for
        # pose, epe for flow. One column so the four applications share a figure.
        "task_metric": test_results.get("task_metric",
                                         "accuracy_pct" if test_results.get("overall_accuracy") is not None else None),
        "task_score": test_results.get("task_score", _pct(test_results.get("overall_accuracy"))),
        "task_score_secondary": test_results.get("task_score_secondary"),
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

        # What network produced this row. Recorded because the architecture is now
        # per-dataset (convolution.blocks), and because `cnn` rows sit in the same
        # file as the four SNN rows -- without neuron_model a reader cannot tell the
        # control apart from the thing it is controlling for.
        "neuron_model": neuron.get("neuron"),
        "conv_blocks": "-".join(
            f"{block['out']}C{block['kernel']}" for block in getattr(cfg, "CONV_BLOCKS", []) or []
        ) or None,
        "fc_hidden": getattr(cfg, "FC_HIDDEN", None),
        "credit_assignment": test_results.get("credit_assignment"),

        # Internal dynamics, measured on the TEST pass. These were computed by
        # SNNTester all along and reached only that run's own test.csv, so no figure
        # could compare them across frameworks -- which is what they exist for.
        "test_activation_density_pct": _pct((density or {}).get("hidden_mean")),
        "spikes_per_neuron_per_inference": test_results.get("avg_spikes_per_neuron_per_inference"),
        "cv_isi_mean": test_results.get("cv_isi_mean"),
        "total_spikes_test": test_results.get("total_spikes"),
        "input_to_output_ratio": test_results.get("framework_ratio"),

        # Energy MODELS, per sample. Kept in their own block, and named _pj, so they
        # are never read as the measured joules recorded further down.
        "energy_per_sample_pj": test_results.get("energy_per_sample_pj"),
        "infer_synops_energy_per_sample_pj": test_results.get("synops_energy_per_sample_pj"),
        "train_synops_energy_pj_total": _sum_over_epochs(epoch_log, "synops_energy_pj"),
        "train_synops_energy_pj_per_epoch": _mean_over_epochs(epoch_log, "synops_energy_pj"),

        # Amortised per-sample latency (batch time / B) -- throughput, NOT
        # single-stream latency. The inference_latency_bs1_* columns above are the
        # real batch-1 measurement; both are reported because they answer different
        # questions (SNN_GPU_Evaluation_Metrics.md 2.3).
        "inference_latency_amortised_p50_ms": test_results.get("median_latency_per_sample_ms"),
        "inference_latency_amortised_p90_ms": test_results.get("p90_latency_per_sample_ms"),
        "inference_latency_amortised_p99_ms": test_results.get("p99_latency_per_sample_ms"),
        "train_forward_latency_ms": _mean_over_epochs(epoch_log, "forward_latency_ms"),
        "train_backward_latency_ms": _mean_over_epochs(epoch_log, "backward_latency_ms"),

        "infer_energy_j": test_results.get("gpu_energy_j_total"),
        "infer_energy_dynamic_j": test_results.get("gpu_energy_j_dynamic"),
        "gpu_temp_c": test_results.get("gpu_temp_c"),
        "sm_clock_mhz": test_results.get("sm_clock_mhz"),
        "mem_clock_mhz": test_results.get("mem_clock_mhz"),

        # Peaks for BOTH phases, in MB. The training figures are the max across the
        # per-epoch samples in epoch_log; the inference ones come from the test run.
        #
        # peak_memory_*  = GPU memory in use, read from NVML
        # peak_reserved_* = PyTorch's caching-allocator high-water mark, which includes
        #                   memory held but not currently in use, so it is the larger
        #                   number and the one that decides whether a batch size fits.
        "peak_memory_train_mb": _peak_over_epochs(epoch_log, "gpu_mem_peak_gb"),
        "peak_memory_infer_mb": (test_results.get("gpu_mem_peak_gb") or 0) * GB_TO_MB
                                 if test_results.get("gpu_mem_peak_gb") else None,
        "peak_reserved_train_mb": _peak_over_epochs(epoch_log, "max_memory_reserved_gb"),
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
