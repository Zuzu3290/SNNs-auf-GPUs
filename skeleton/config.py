"""Run settings: the three YAML files in configuration/, plus the --framework flag."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configuration"
BASE_FILES = ("SNN_module.yaml", "network_architecture.yaml", "data_workflow.yaml")

# training.framework selector -> neuron: key in network_architecture.yaml
FW_TO_CFG_KEY = {"torch": "snntorch", "norse": "norse", "sj": "spikingjelly", "sinabs": "sinabs"}

INFERENCE_DEVICES = ("cuda", "cpu")


def load_config() -> dict:
    """The three base files merged into one dict."""
    config: dict = {}
    for name in BASE_FILES:
        with open(CONFIG_DIR / name, encoding="utf-8") as file:
            config.update(yaml.safe_load(file))
    return config


class Settings:
    def __init__(self, config: dict):
        self.config = config
        c = config
        # 12-character code of the merged YAML: two results rows with the same code used the same settings.
        self.CONFIG_HASH = hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

        conv = c["convolution"]
        self.CONV_BLOCKS  = conv["blocks"]
        self.POOL_KERNEL  = conv["pool_kernel"]
        self.CONV_PADDING = conv["padding"]
        self.FC_HIDDEN    = conv["fc_hidden"] or []
        self.BATCH_NORM   = conv["batch_norm"]
        self.NEURON       = c["neuron"]

        # Input shape and class count come from the dataset registry, set by apply_dataset_shape().
        self.SENSOR_H = self.SENSOR_W = self.IN_CHANNELS = self.NUM_CLASSES = None

        t = c["training"]
        self.FRAMEWORK            = t["framework"]
        self.DEVICE               = t["device"]
        self.SEED                 = t["seed"]
        self.EPOCHS               = t["epochs"]
        self.BATCH_SIZE           = t["batch_size"]
        self.ITERA                = t["iterations_per_epoch"] or None  # null or 0 = one full pass
        self.CALIBRATE_BATCH_SIZE = t["calibrate_batch_size"]
        self.WARMUP_ITERATIONS    = t["warmup_iterations"]
        self.LATENCY_SAMPLES      = t["latency_samples"]
        self.OPTIMIZER            = t["optimizer"]["type"]  # a torch.optim class name, e.g. NAdam, Adam, SGD
        self.OPTIMIZER_ARGS       = {k: v for k, v in t["optimizer"].items() if k != "type"}
        self.LEARNING_RATE        = t["optimizer"]["lr"]
        self.WEIGHT_DECAY         = t["optimizer"]["weight_decay"]
        self.LOSS_FN              = t["loss"]
        self.LR_SCHEDULER         = t["lr_scheduler"]
        self.USE_AMP              = t["use_amp"]
        self.GRAD_ACCUM_STEPS     = max(1, t["grad_accum_steps"])
        self.ACTIVITY_REG_ENABLED     = t["activity_reg_enabled"]
        self.ACTIVITY_REG_MIN_RATE    = t["activity_reg_min_rate"]
        self.ACTIVITY_REG_MAX_RATE    = t["activity_reg_max_rate"]
        self.ACTIVITY_REG_LAMBDA_LOW  = t["activity_reg_lambda_low"]
        self.ACTIVITY_REG_LAMBDA_HIGH = t["activity_reg_lambda_high"]

        self.INFERENCE_DEVICE = c["inference"]["device"]  # null = ask at the prompt

        self.DATASET_NAME = None  # chosen from the menu at startup, set by main

        out = c["output"]
        self.PLOT_DIR        = out["plot_dir"]
        self.DATA_DIR        = out["data_dir"]
        self.SAVE_CHECKPOINT = out["save_checkpoint"]

        b = c["binning"]
        self.FRAME_MODE             = b["mode"]
        self.N_TIME_BINS            = b["n_time_bins"]
        self.TIME_WINDOW_US         = int(b["time_window_ms"] * 1000)
        self.DENOISE_FILTER_TIME_US = b["denoise_filter_time_us"]  # null disables the filter

        self.TEMPORAL_SLICING_ENABLED = c["temporal"]["enabled"]
        self.SLICE_DURATION_US        = c["temporal"]["slice_duration_us"]
        self.SLICE_OVERLAP_US         = c["temporal"]["overlap_us"]
        self.CACHE_PATH               = c["cache"]["path"]

        self.AUGMENT_ENABLED          = c["augmentation"]["enabled"]
        self.AUGMENT_ROTATION_DEG     = c["augmentation"]["rotation_degrees"]

        r = c["resource_policy"]
        self.MEMORY_SAFETY_MARGIN_GB       = r["memory_safety_margin_gb"]
        self.MEMORY_CACHE_THRESHOLD_GB     = r["memory_cache_threshold_gb"]
        self.BATCH_VRAM_FRACTION           = r["batch_vram_fraction"]
        self.BATCH_VRAM_BAND_MIN           = r["batch_vram_band_min"]
        self.MAX_BATCH_SIZE                = r["max_batch_size"]
        self.MEMORY_TIER_HEADROOM_FRACTION = r["memory_tier_headroom_fraction"]
        self.DISK_TIER_HEADROOM_MULTIPLE   = r["disk_tier_headroom_multiple"]
        self.WORKER_RAM_FRACTION           = r["worker_ram_fraction"]
        self.DATALOADER_WORKER_TIMEOUT_S   = r["dataloader_worker_timeout_s"]
        self.WORKER_COUNT = None if r["calibrate_workers"] else r["worker_count_fallback"]  # None = sized from free RAM

    def apply_dataset_shape(self, sensor_h: int, sensor_w: int, in_channels: int, num_classes: int):
        """Set input shape and class count from the dataset registry; must run before the model is built."""
        self.SENSOR_H    = sensor_h
        self.SENSOR_W    = sensor_w
        self.IN_CHANNELS = in_channels
        self.NUM_CLASSES = num_classes

    def resolve_iterations(self, train_loader) -> int:
        """Batches per epoch: the whole training set when iterations_per_epoch is null, else that number capped at it."""
        inner = getattr(train_loader, "loader", train_loader)
        full_pass = len(inner)
        requested = self.ITERA
        self.ITERA = full_pass if requested is None else min(requested, full_pass)
        if requested is None:
            note = "auto (full pass)"
        elif self.ITERA < requested:
            note = f"capped to the full pass; config asked for {requested}"
        else:
            note = f"capped by config at {requested} of {full_pass}"
        covered, total = self.ITERA * self.BATCH_SIZE, len(inner.dataset)
        logger.info(f"[ITERATIONS] {self.ITERA} batches/epoch x {self.BATCH_SIZE} batch = "
                    f"{covered}/{total} samples ({covered / total * 100:.1f}%)  -- {note}")
        return self.ITERA

    def display(self, net):
        """Log the values this run actually uses, every YAML setting (only the chosen framework's neuron), and the network's shapes."""
        neuron_key = FW_TO_CFG_KEY[self.FRAMEWORK]
        lines = ["=" * 76,
                 f"Framework {self.FRAMEWORK}   dataset {self.DATASET_NAME}   train on {self.DEVICE}   "
                 f"inference on {self.INFERENCE_DEVICE}",
                 f"Batch size {self.BATCH_SIZE}   iterations/epoch {self.ITERA}   config hash {self.CONFIG_HASH}"]
        for section, values in self.config.items():
            if section == "neuron":
                section, values = f"neuron: {neuron_key}", values[neuron_key]
            lines.append(f"\n[{section}]")
            lines += [f"  {key:<30}: {value}" for key, value in values.items()]

        lines.append("\n[network: shape leaving each layer, one time step]")
        activation = torch.zeros(1, self.IN_CHANNELS, self.SENSOR_H, self.SENSOR_W, device=next(net.parameters()).device)
        lines.append(f"  {'input':<30}: {tuple(activation.shape)}")
        was_training = net.training
        net.eval()  # BatchNorm running statistics must not see the probe
        with torch.no_grad():
            for layer in net.layers:
                activation = layer(activation)
                label = type(layer).__name__
                if isinstance(layer, nn.Conv2d):
                    label += f"({layer.in_channels}->{layer.out_channels}, k{layer.kernel_size[0]})"
                elif isinstance(layer, nn.MaxPool2d):
                    label += f"({layer.kernel_size})"
                elif isinstance(layer, nn.Linear):
                    label += f"({layer.in_features}->{layer.out_features})"
                lines.append(f"  {label:<30}: {tuple(activation.shape)}")
        net.reset()
        net.train(was_training)
        lines.append("=" * 76)
        logger.info("\n" + "\n".join(lines))


def load_settings() -> Settings:
    """Settings from configuration/, with a --framework override recorded in OVERRIDES; asks for the inference device when it is null."""
    parser = argparse.ArgumentParser(description="Train and evaluate one SNN framework on one event dataset.")
    parser.add_argument("--framework", default=None, choices=sorted(FW_TO_CFG_KEY), help="overrides training.framework")
    args = parser.parse_args()
    cfg = Settings(load_config())
    cfg.OVERRIDES = {}
    if args.framework is not None:
        cfg.FRAMEWORK = cfg.OVERRIDES["framework"] = args.framework
    if cfg.INFERENCE_DEVICE is None:
        print("\nRun inference on:\n  1) cuda\n  2) cpu\n  (set inference.device in the config to skip this prompt)")
        cfg.INFERENCE_DEVICE = INFERENCE_DEVICES[int(input("Enter number: ").strip()) - 1]
    return cfg
