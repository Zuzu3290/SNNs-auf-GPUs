"""Run settings: the three YAML files in configuration/, plus the --framework flag."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configuration"
BASE_FILES = ("SNN_module.yaml", "network_architecture.yaml", "data_workflow.yaml")

# training.framework selector -> neuron: key in network_architecture.yaml
FW_TO_CFG_KEY = {"torch": "snntorch", "norse": "norse", "sj": "spikingjelly", "sinabs": "sinabs"}


def load_config() -> dict:
    """The three base files merged into one dict."""
    config: dict = {}
    for name in BASE_FILES:
        with open(CONFIG_DIR / name, encoding="utf-8") as file:
            config.update(yaml.safe_load(file))
    return config


def config_hash(config: dict) -> str:
    """Hash of the merged config, so a results row shows which settings produced it."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


class Settings:
    def __init__(self, config: dict):
        self.config = config
        self.CONFIG_HASH = config_hash(config)
        c = config

        conv = c["convolution"]
        self.CONV_BLOCKS  = conv["blocks"]
        self.POOL_KERNEL  = conv["pool_kernel"]
        self.CONV_PADDING = conv["padding"]
        self.FC_HIDDEN    = conv["fc_hidden"] or []
        self.BATCH_NORM   = conv["batch_norm"]
        self.NEURON       = c["neuron"]

        # Input shape and class count come from the dataset registry, set by apply_dataset_shape().
        self.SENSOR_H = self.SENSOR_W = self.IN_CHANNELS = self.NUM_CLASSES = self.FC_IN = None

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
        self.OPTIMIZER            = t["optimizer"]["type"]
        self.LEARNING_RATE        = t["optimizer"]["lr"]
        self.WEIGHT_DECAY         = t["optimizer"]["weight_decay"]
        self.SGD_MOMENTUM         = t["optimizer"]["momentum"]
        self.LOSS_FN              = t["loss"]
        self.LR_SCHEDULER         = t["lr_scheduler"]
        self.USE_AMP              = t["use_amp"]
        self.GRAD_ACCUM_STEPS     = max(1, t["grad_accum_steps"])
        self.ACTIVITY_REG_ENABLED     = t["activity_reg_enabled"]
        self.ACTIVITY_REG_MIN_RATE    = t["activity_reg_min_rate"]
        self.ACTIVITY_REG_MAX_RATE    = t["activity_reg_max_rate"]
        self.ACTIVITY_REG_LAMBDA_LOW  = t["activity_reg_lambda_low"]
        self.ACTIVITY_REG_LAMBDA_HIGH = t["activity_reg_lambda_high"]

        self.DATASET_NAME = c["dataset"]["name"]  # null = ask at the prompt

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
        self.CACHE_PATH               = c["cache"]["path"]

        r = c["resource_policy"]
        self.MEMORY_SAFETY_MARGIN_GB       = r["memory_safety_margin_gb"]
        self.MEMORY_CACHE_THRESHOLD_GB     = r["memory_cache_threshold_gb"]
        self.GPU_PRESSURE_THRESHOLD        = r["gpu_pressure_threshold"]
        self.BATCH_VRAM_FRACTION           = r["batch_vram_fraction"]
        self.BATCH_VRAM_BAND_MIN           = r["batch_vram_band_min"]
        self.MAX_BATCH_SIZE                = r["max_batch_size"]
        self.MEMORY_TIER_HEADROOM_FRACTION = r["memory_tier_headroom_fraction"]
        self.DISK_TIER_HEADROOM_MULTIPLE   = r["disk_tier_headroom_multiple"]
        self.WORKER_RAM_FRACTION           = r["worker_ram_fraction"]
        self.DATALOADER_WORKER_TIMEOUT_S   = r["dataloader_worker_timeout_s"]
        self.CALIBRATE_WORKERS             = r["calibrate_workers"]
        self.WORKER_COUNT_FALLBACK         = r["worker_count_fallback"]

    @property
    def worker_count_override(self):
        """None when calibrate_workers is true (adaptive sizing); worker_count_fallback otherwise."""
        return None if self.CALIBRATE_WORKERS else self.WORKER_COUNT_FALLBACK

    def compute_fc_in(self, sensor_h: int, sensor_w: int) -> int:
        """Flattened size after every conv+pool block; H and W separately so non-square sensors work."""
        h, w = sensor_h, sensor_w
        for block in self.CONV_BLOCKS:
            h = (h + 2 * self.CONV_PADDING - block["kernel"] + 1) // self.POOL_KERNEL
            w = (w + 2 * self.CONV_PADDING - block["kernel"] + 1) // self.POOL_KERNEL
        return self.CONV_BLOCKS[-1]["out"] * h * w

    def apply_dataset_shape(self, sensor_h: int, sensor_w: int, in_channels: int, num_classes: int):
        """Set input shape and class count from the dataset registry; must run before the model is built."""
        self.SENSOR_H    = sensor_h
        self.SENSOR_W    = sensor_w
        self.IN_CHANNELS = in_channels
        self.NUM_CLASSES = num_classes
        self.FC_IN       = self.compute_fc_in(sensor_h, sensor_w)

    def resolve_iterations(self, train_loader) -> int:
        """Cap ITERA at one full pass of the real loader (len() already accounts for drop_last)."""
        full_pass = len(getattr(train_loader, "loader", train_loader))
        self.ITERA = full_pass if self.ITERA is None else min(self.ITERA, full_pass)
        return self.ITERA

    def display(self, net):
        """Print the run's settings and the built network layer by layer."""
        W   = 76
        fw  = self.FRAMEWORK.upper()
        sep = "-" * (W - 4)

        def section(title):
            print(f"\n  [{title}]")
            print(f"  {sep}")

        def row(label, value, lw=22):
            print(f"    {label:<{lw}}: {value}")

        print()
        print("=" * W)
        print(f"{'SNN CONFIGURATION':^{W}}")
        print(f"{'Framework : ' + fw + '   |   Device : ' + self.DEVICE:^{W}}")
        print("=" * W)
        row("Config hash", self.CONFIG_HASH)

        section(f"NEURON — {fw}")
        for key, val in self.NEURON[FW_TO_CFG_KEY[self.FRAMEWORK]].items():
            if isinstance(val, dict):  # e.g. surrogate: {type:..., alpha:...}
                row(key, ", ".join(f"{k}={v}" for k, v in val.items()))
            else:
                row(key, val)

        section("TRAINING")
        row("Seed",               self.SEED)
        row("Epochs",             self.EPOCHS)
        row("Iterations / epoch", self.ITERA)
        row("Batch size",         self.BATCH_SIZE)
        row("Optimizer",          f"{self.OPTIMIZER}   lr {self.LEARNING_RATE}   weight decay {self.WEIGHT_DECAY}")
        row("Loss",               self.LOSS_FN)
        row("LR scheduler",       self.LR_SCHEDULER)
        row("AMP",                "on" if self.USE_AMP else "off")
        row("Activity reg",       "on" if self.ACTIVITY_REG_ENABLED else "off")

        section("DATA")
        row("Dataset",   self.DATASET_NAME)
        row("Time bins", self.N_TIME_BINS if self.FRAME_MODE == "n_time_bins" else f"{self.TIME_WINDOW_US} µs windows")
        row("Denoise",   f"{self.DENOISE_FILTER_TIME_US} µs" if self.DENOISE_FILTER_TIME_US else "off")

        self.display_layers(net, section, row)

        print()
        print("=" * W)
        print()

    def display_layers(self, net, section, row) -> None:
        """The built network layer by layer, with the shape leaving each layer for one time step."""
        section("NETWORK — shapes for one time step")
        device = next(net.parameters()).device
        activation = torch.zeros(1, self.IN_CHANNELS, self.SENSOR_H, self.SENSOR_W, device=device)
        row("input", tuple(activation.shape), lw=34)
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
                row(label, tuple(activation.shape), lw=34)
        net.reset()
        net.train(was_training)


def settings_from_cli() -> Settings:
    """Settings from configuration/, with an optional --framework override recorded in OVERRIDES."""
    parser = argparse.ArgumentParser(description="Train and evaluate one SNN framework on one event dataset.")
    parser.add_argument("--framework", default=None, choices=sorted(FW_TO_CFG_KEY), help="overrides training.framework")
    args = parser.parse_args()
    cfg = Settings(load_config())
    cfg.OVERRIDES = {}
    if args.framework is not None:
        cfg.FRAMEWORK = cfg.OVERRIDES["framework"] = args.framework
    return cfg
