"""Run settings: the three YAML files in configuration/, an optional overlay, and the command-line flags."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configuration"
BASE_FILES = ("SNN_module.yaml", "network_architecture.yaml", "data_workflow.yaml")

# training.framework selector -> neuron: / neuron_types: key in network_architecture.yaml
FW_TO_CFG_KEY = {"torch": "snntorch", "norse": "norse", "sj": "spikingjelly", "sinabs": "sinabs"}
MODEL_CHOICES = sorted(FW_TO_CFG_KEY)


class NeuronSpecError(Exception):
    """A neuron parameter is missing or not one of the allowed choices."""


def read_yaml(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Nested dicts merge key by key; any other value, lists included, is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(overlay_path: str | None = None) -> dict:
    """The three base files merged into one dict, with the overlay (if any) merged on top."""
    config: dict = {}
    for name in BASE_FILES:
        config = deep_merge(config, read_yaml(CONFIG_DIR / name))
    if overlay_path:
        config = deep_merge(config, read_yaml(overlay_path))
    return config


def config_hash(config: dict) -> str:
    """Hash of the merged config, so a results row shows which settings produced it."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def choice(value: Any, label: str, allowed) -> Any:
    if value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}, got {value!r}")
    return value


def widths(value) -> list[int]:
    """fc_hidden: null (none), one width, or a list of widths."""
    if value is None:
        return []
    return [int(v) for v in value] if isinstance(value, list) else [int(value)]


class Settings:
    def __init__(self, config: dict | None = None, config_path: str | None = None):
        self.config = config if config is not None else load_config(config_path)
        self.CONFIG_PATH = config_path
        self.CONFIG_HASH = config_hash(self.config)
        c = self.config

        conv = c["convolution"]
        self.CONV_BLOCKS  = [{"out": int(b["out"]), "kernel": int(b["kernel"])} for b in conv["blocks"]]
        self.POOL_KERNEL  = int(conv["pool_kernel"])
        self.CONV_PADDING = int(conv["padding"])
        self.FC_HIDDEN    = widths(conv["fc_hidden"])
        self.BATCH_NORM   = bool(conv["batch_norm"])
        # Shortcuts to the first two blocks, for the hand-written two-block model in frameworks/snn_lif.py.
        self.CONV1_OUT    = self.CONV_BLOCKS[0]["out"]
        self.CONV1_KERNEL = self.CONV_BLOCKS[0]["kernel"]
        self.CONV2_OUT    = self.CONV_BLOCKS[min(1, len(self.CONV_BLOCKS) - 1)]["out"]
        self.CONV2_KERNEL = self.CONV_BLOCKS[min(1, len(self.CONV_BLOCKS) - 1)]["kernel"]
        self.NEURON_TYPES = c["neuron_types"]
        self.NEURON       = c["neuron"]

        # Input shape and class count come from the dataset registry, set by apply_dataset_shape().
        self.SENSOR_H = self.SENSOR_W = self.IN_CHANNELS = self.NUM_CLASSES = self.FC_IN = None

        t = c["training"]
        self.FRAMEWORK            = choice(t["framework"], "training.framework", MODEL_CHOICES)
        self.DEVICE               = t["device"]
        self.SEED                 = int(t["seed"])
        self.EPOCHS               = int(t["epochs"])
        self.BATCH_SIZE           = int(t["batch_size"])
        self.ITERA                = t["iterations_per_epoch"] or None  # null or 0 = one full pass
        self.CALIBRATE_BATCH_SIZE = bool(t["calibrate_batch_size"])
        self.WARMUP_ITERATIONS    = int(t["warmup_iterations"])
        self.LATENCY_SAMPLES      = int(t["latency_samples"])
        self.OPTIMIZER            = t["optimizer"]["type"]
        self.LEARNING_RATE        = float(t["optimizer"]["lr"])
        self.WEIGHT_DECAY         = float(t["optimizer"]["weight_decay"])
        self.SGD_MOMENTUM         = float(t["optimizer"]["momentum"])
        self.LOSS_FN              = t["loss"]
        self.LR_SCHEDULER         = choice(t["lr_scheduler"], "training.lr_scheduler", ["none", "cosine"])
        self.USE_AMP              = bool(t["use_amp"])
        self.GRAD_ACCUM_STEPS     = max(1, int(t["grad_accum_steps"]))
        self.ACTIVITY_REG_ENABLED     = bool(t["activity_reg_enabled"])
        self.ACTIVITY_REG_MIN_RATE    = float(t["activity_reg_min_rate"])
        self.ACTIVITY_REG_MAX_RATE    = float(t["activity_reg_max_rate"])
        self.ACTIVITY_REG_LAMBDA_LOW  = float(t["activity_reg_lambda_low"])
        self.ACTIVITY_REG_LAMBDA_HIGH = float(t["activity_reg_lambda_high"])

        self.DATASET_NAME = c["dataset"]["name"]  # null = ask at the prompt
        self.TASK_TYPE    = "classification"

        out = c["output"]
        self.OUTPUT_DIR      = out["output_dir"]
        self.PLOT_DIR        = out["plot_dir"]
        self.DATA_DIR        = out["data_dir"]
        self.SAVE_CHECKPOINT = bool(out["save_checkpoint"])

        b = c["binning"]
        self.FRAME_MODE             = choice(b["mode"], "binning.mode", ["n_time_bins", "time_window"])
        self.N_TIME_BINS            = int(b["n_time_bins"])
        self.TIME_WINDOW_US         = int(float(b["time_window_ms"]) * 1000)
        self.DENOISE_FILTER_TIME_US = b["denoise_filter_time_us"]  # null disables the filter

        self.TEMPORAL_SLICING_ENABLED = bool(c["temporal"]["enabled"])
        self.SLICE_DURATION_US        = int(c["temporal"]["slice_duration_us"])
        self.CACHE_PATH               = c["cache"]["path"]

        r = c["resource_policy"]
        self.MEMORY_SAFETY_MARGIN_GB       = float(r["memory_safety_margin_gb"])
        self.MEMORY_CACHE_THRESHOLD_GB     = float(r["memory_cache_threshold_gb"])
        self.GPU_PRESSURE_THRESHOLD        = float(r["gpu_pressure_threshold"])
        self.BATCH_VRAM_FRACTION           = float(r["batch_vram_fraction"])
        self.BATCH_VRAM_BAND_MIN           = float(r["batch_vram_band_min"])
        self.MAX_BATCH_SIZE                = int(r["max_batch_size"])
        self.MEMORY_TIER_HEADROOM_FRACTION = float(r["memory_tier_headroom_fraction"])
        self.DISK_TIER_HEADROOM_MULTIPLE   = float(r["disk_tier_headroom_multiple"])
        self.WORKER_RAM_FRACTION           = float(r["worker_ram_fraction"])
        self.DATALOADER_WORKER_TIMEOUT_S   = float(r["dataloader_worker_timeout_s"])
        self.CALIBRATE_WORKERS             = bool(r["calibrate_workers"])
        self.WORKER_COUNT_FALLBACK         = int(r["worker_count_fallback"])

    @property
    def worker_count_override(self) -> int | None:
        """None when calibrate_workers is true (adaptive sizing); worker_count_fallback otherwise."""
        return None if self.CALIBRATE_WORKERS else self.WORKER_COUNT_FALLBACK

    @property
    def active_fw_cfg(self) -> dict:
        """The neuron block of the selected framework."""
        return self.NEURON.get(FW_TO_CFG_KEY[self.FRAMEWORK], {})

    def compute_fc_in(self, sensor_h: int, sensor_w: int) -> int:
        """Flattened size after every conv+pool block; H and W separately so non-square sensors work."""
        h, w = sensor_h, sensor_w
        for block in self.CONV_BLOCKS:
            h = (h + 2 * self.CONV_PADDING - block["kernel"] + 1) // self.POOL_KERNEL
            w = (w + 2 * self.CONV_PADDING - block["kernel"] + 1) // self.POOL_KERNEL
        return self.CONV_BLOCKS[-1]["out"] * h * w

    def apply_dataset_shape(self, sensor_h: int, sensor_w: int, in_channels: int, num_classes: int):
        """Set input shape and class count from the dataset registry; must run before the model is built."""
        self.SENSOR_H    = int(sensor_h)
        self.SENSOR_W    = int(sensor_w)
        self.IN_CHANNELS = int(in_channels)
        self.NUM_CLASSES = int(num_classes)
        self.FC_IN       = self.compute_fc_in(self.SENSOR_H, self.SENSOR_W)

    def resolve_iterations(self, train_loader) -> int:
        """Cap ITERA at one full pass of the real loader (len() already accounts for drop_last)."""
        full_pass = len(getattr(train_loader, "loader", train_loader))
        self.ITERA = full_pass if self.ITERA is None else min(self.ITERA, full_pass)
        return self.ITERA

    def display(self, net=None):
        """Print the configuration of this run, and the built network if given."""
        W      = 76
        fw     = self.FRAMEWORK.upper()
        sep    = "-" * (W - 4)

        def section(title):
            print(f"\n  [{title}]")
            print(f"  {sep}")

        def row(label, value, lw=22):
            print(f"    {label:<{lw}}: {value}")

        def shape(value):
            return "not set yet" if value is None else str(value)

        print()
        print("=" * W)
        print(f"{'SNN CONFIGURATION':^{W}}")
        print(f"{'Framework : ' + fw + '   |   Device : ' + self.DEVICE:^{W}}")
        print("=" * W)
        row("Config", f"{self.CONFIG_PATH or 'configuration/ only'}   hash {self.CONFIG_HASH}")

        section("ARCHITECTURE")
        if self.SENSOR_H is None:
            row("Sensor", "not set yet -- comes from the dataset registry")
        else:
            row("Sensor", f"{self.SENSOR_H} × {self.SENSOR_W}   ({self.IN_CHANNELS} channels)")
        for index, block in enumerate(self.CONV_BLOCKS, start=1):
            row(f"Conv{index}", f"{block['out']} filters   {block['kernel']}×{block['kernel']} kernel")
        row("Pool",            f"{self.POOL_KERNEL}×{self.POOL_KERNEL} MaxPool   "
                               f"(after each of {len(self.CONV_BLOCKS)} conv blocks)")
        row("Conv padding",    str(self.CONV_PADDING))
        row("Batch norm",      "ENABLED" if self.BATCH_NORM else "DISABLED")
        row("FC input (auto)", shape(self.FC_IN))
        row("FC hidden",       " -> ".join(str(w) for w in self.FC_HIDDEN)
                               if self.FC_HIDDEN else "none (Flatten -> output)")
        row("Output classes",  shape(self.NUM_CLASSES))

        cfg_key = FW_TO_CFG_KEY[self.FRAMEWORK]
        section(f"NEURON TYPES — {fw}")
        for layer, ntype in self.NEURON_TYPES.get(cfg_key, {}).items():
            row(layer, ntype)

        section(f"NEURON SPEC — {fw}")
        for key, val in self.active_fw_cfg.items():
            if isinstance(val, dict):  # e.g. surrogate: {type:..., alpha:...}
                row(key, ", ".join(f"{k}={v}" for k, v in val.items()))
            else:
                row(key, f"{val} Hz" if key == "tau_mem_inv" else str(val))

        section("TRAINING")
        row("Seed",               str(self.SEED))
        row("Epochs",             str(self.EPOCHS))
        row("Iterations / epoch", str(self.ITERA))
        row("Batch size",         str(self.BATCH_SIZE))
        row("Optimizer",          self.OPTIMIZER)
        row("Learning rate",      str(self.LEARNING_RATE))
        row("Weight decay",       str(self.WEIGHT_DECAY))
        row("Loss",               self.LOSS_FN)
        row("LR scheduler",       self.LR_SCHEDULER)
        row("Grad accum steps",   str(self.GRAD_ACCUM_STEPS))
        row("AMP (mixed prec.)",  "ENABLED" if self.USE_AMP else "DISABLED")

        section("REGULARIZATION")
        if self.ACTIVITY_REG_ENABLED:
            row("Activity reg", f"ENABLED   min={self.ACTIVITY_REG_MIN_RATE * 100:.0f}%   max={self.ACTIVITY_REG_MAX_RATE * 100:.0f}%")
        else:
            row("Activity reg", "DISABLED")

        section("DATA")
        row("Dataset",     self.DATASET_NAME)
        row("Time bins",   self.N_TIME_BINS if self.FRAME_MODE == "n_time_bins" else f"{self.TIME_WINDOW_US} µs windows")
        row("Denoise",     f"{self.DENOISE_FILTER_TIME_US} µs" if self.DENOISE_FILTER_TIME_US else "off")
        row("Cache",       self.CACHE_PATH)

        section("OUTPUT")
        row("Results dir", self.DATA_DIR)
        row("Plot dir",    self.PLOT_DIR)
        row("Save checkpoint", self.SAVE_CHECKPOINT)

        if net is not None:
            self.display_layers(net, section, row)

        print()
        print("=" * W)
        print()

    def display_layers(self, net, section, row) -> None:
        """The built network layer by layer, with the shape leaving each layer for one time step."""
        import torch
        import torch.nn as nn
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
        linear = next(m for m in net.layers if isinstance(m, nn.Linear))
        row("flatten width measured / FC_IN", f"{linear.in_features} / {self.FC_IN}", lw=34)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The run flags: an optional overlay, and overrides for framework, seed and cache folder."""
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="YAML overlay merged over the three files in configuration/")
    parser.add_argument("--framework", default=None, choices=MODEL_CHOICES,
                        help="overrides training.framework")
    parser.add_argument("--seed", type=int, default=None, metavar="N", help="overrides training.seed")
    parser.add_argument("--cache-root", default=None, metavar="PATH", help="overrides cache.path")
    return parser


def from_args(args: argparse.Namespace) -> Settings:
    """Settings from the config files, with the command-line overrides applied on top."""
    cfg = Settings(config_path=args.config)
    cfg.OVERRIDES = {}
    for name, attr in (("framework", "FRAMEWORK"), ("seed", "SEED"), ("cache_root", "CACHE_PATH")):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, attr, value)
            cfg.OVERRIDES[name] = value
    return cfg


def neuron_cfg(cfg, framework_key: str) -> dict:
    """The neuron block of one framework ('snntorch' | 'norse' | 'spikingjelly' | 'sinabs')."""
    if framework_key not in cfg.NEURON:
        raise NeuronSpecError(f"neuron.{framework_key} missing from network_architecture.yaml")
    return cfg.NEURON[framework_key]


def neuron_value(d: dict, path: str) -> Any:
    """One neuron parameter; a missing key raises, there is no framework default to fall back to."""
    key = path.split(".")[-1]
    if key not in d:
        raise NeuronSpecError(f"neuron.{path} is not set in network_architecture.yaml. Keys present: {sorted(d)}")
    return d[key]


def require_float(d: dict, path: str) -> float:
    return float(neuron_value(d, path))


def require_bool(d: dict, path: str) -> bool:
    return bool(neuron_value(d, path))


def require_choice(d: dict, path: str, allowed) -> str:
    return choice(neuron_value(d, path), f"neuron.{path}", allowed)


def optional_float(d: dict, path: str) -> float | None:
    """A neuron key whose off value is null; the key must still be present."""
    value = neuron_value(d, path)
    return None if value is None else float(value)


def require_surrogate(d: dict, path: str = "surrogate") -> tuple[str, float]:
    """(type, alpha) of a surrogate gradient block."""
    block = neuron_value(d, path)
    return str(block["type"]), float(block["alpha"])


def describe(cfg) -> dict:
    """The whole neuron block flattened to dotted keys."""
    flat: dict[str, Any] = {}
    for fw, params in cfg.NEURON.items():
        for key, value in params.items():
            if isinstance(value, dict):
                for sub, subvalue in value.items():
                    flat[f"{fw}.{key}.{sub}"] = subvalue
            else:
                flat[f"{fw}.{key}"] = value
    return flat
