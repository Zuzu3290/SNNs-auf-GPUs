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
import torch.nn as nn

from skeleton.results import environment_info, make_run_id

GB_TO_MB = 1024.0


def _pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def _last(seq) -> Any:
    return seq[-1] if seq else None


def _architecture(model, cfg) -> dict:
    """Every architecture fact a size-ladder row needs, read off the LIVE model.

    Measured rather than read back from the config, on the same principle as
    SpikingNet.measure_flat_features(): the built layer list is the authority, and a
    config value only says what was *asked for*. A row that reports the request rather
    than the result cannot be trusted to describe what actually ran.

    The parameter split is three-way and the boundary matters:

        conv_params        every Conv2d -- the feature extractor, what Factor A sweeps
        fc_hidden_params   every Linear EXCEPT the last -- what Factor B adds
        classifier_params  the LAST Linear -- flatten_width x num_classes

    The classifier is separated because its size follows the SENSOR RESOLUTION, not the
    conv widths under test. On a 180x240 sensor it reaches ~7.7M parameters against a
    ~10k conv stack, so "we doubled the network" would be a statement about that one
    matrix. Keeping it in its own column makes the share visible per run.

    fc_hidden_layers / fc_hidden_size are counted from the model too, so they read 0 and
    empty today and start reporting real values the moment FC hidden layers are added --
    no schema change and no config key needed for that to work.

    Everything is guarded: a model without `.net`, or one whose probe fails, yields an
    all-None dict and the row still writes. A bookkeeping error must never cost a
    finished run.
    """
    blank = {
        "total_neurons": None, "neurons_per_layer": None,
        "conv1_out": None, "conv2_out": None,
        "conv1_kernel": None, "conv2_kernel": None, "pool_kernel": None,
        "fc_hidden_layers": None, "fc_hidden_size": None, "flatten_width": None,
        "conv_params": None, "fc_hidden_params": None, "classifier_params": None,
    }

    net = getattr(model, "net", None)
    if net is None:
        return blank

    out = dict(blank)

    # ---- neurons: the size axis, measured by a shape probe --------------------
    try:
        counts = model.neuron_counts()
    except Exception:  # a diagnostic must not cost a finished run
        counts = {}
    if counts:
        out["total_neurons"] = sum(counts.values())
        # pipe-separated, never comma -- this lands in a CSV cell
        out["neurons_per_layer"] = "|".join(f"{name}:{n}" for name, n in counts.items())

    # ---- one walk of the real layer list --------------------------------------
    convs = [layer for layer in net.layers if isinstance(layer, nn.Conv2d)]
    linears = [layer for layer in net.layers if isinstance(layer, nn.Linear)]

    if convs:
        out["conv1_out"] = convs[0].out_channels
        out["conv1_kernel"] = convs[0].kernel_size[0]
        out["conv_params"] = sum(p.numel() for layer in convs for p in layer.parameters())
    if len(convs) > 1:
        out["conv2_out"] = convs[1].out_channels
        out["conv2_kernel"] = convs[1].kernel_size[0]

    pools = [layer for layer in net.layers if isinstance(layer, nn.MaxPool2d)]
    if pools:
        kernel = pools[0].kernel_size
        out["pool_kernel"] = kernel[0] if isinstance(kernel, tuple) else kernel

    if linears:
        # The first Linear is whatever consumes the flattened feature map, whether or
        # not hidden layers exist -- so its in_features IS the measured flatten width.
        out["flatten_width"] = linears[0].in_features
        out["classifier_params"] = sum(p.numel() for p in linears[-1].parameters())
        hidden = linears[:-1]
        out["fc_hidden_layers"] = len(hidden)
        out["fc_hidden_size"] = hidden[0].out_features if hidden else None
        out["fc_hidden_params"] = sum(
            p.numel() for layer in hidden for p in layer.parameters()
        ) if hidden else 0

    return out


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


def build_layer_rows(
    model, activity_snapshot: dict | None = None,
    capacity_metrics: dict | None = None,
    grad_norm_means: dict | None = None,
) -> list[dict]:
    """One row per spiking layer.

    Two sources for the base spike-rate fields, in order of preference:

      1. BaseLIF's own spike counters, when spike counting was switched on. Covers every
         layer including lif_out, and is the same measurement SNNs_2 records.
      2. The ActivityMonitor snapshot the trainer already collects. Free -- no extra pass
         -- and now covers every hooked layer including lif_out (see
         frameworks/snn_model.py's dynamic hooking fix).

    capacity_metrics and grad_norm_means are optional, keyed by the same LIF slot
    names -- populated only when training.compute_capacity_metrics was on for this run
    (see learning/training.py's SNNTrainer.last_capacity_metrics / grad_norm_means).
    Left None for every field when either dict is absent or has no entry for a given
    layer, matching this file's "unmeasured metric writes an empty cell" convention.

    Returns an empty list when neither spike-rate source is available, which simply
    means no layers.csv rows for this run rather than a failure.
    """
    capacity_metrics = capacity_metrics or {}
    grad_norm_means = grad_norm_means or {}
    rows: list[dict] = []
    named = model.net.named_lif_layers() if hasattr(model, "net") else {}

    def _capacity_fields(name: str) -> dict:
        values = capacity_metrics.get(name) or {}
        return {
            "participation_ratio": values.get("participation_ratio"),
            "participation_ratio_normalized": values.get("participation_ratio_normalized"),
            "spike_entropy": values.get("spike_entropy"),
            "spike_entropy_normalized": values.get("spike_entropy_normalized"),
            "mutual_info_zy": values.get("mutual_info_zy"),
            "grad_norm_mean": grad_norm_means.get(name),
        }

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
                **_capacity_fields(name),
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
            **_capacity_fields(name),
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

        # architecture / size axis (schema v2) -- see _architecture() for why each of
        # these is measured off the live model rather than read back from the config.
        **_architecture(model, cfg),

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
