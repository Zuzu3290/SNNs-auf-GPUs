import yaml
from pathlib import Path

DEFAULT_YAML      = Path(__file__).parent.parent / "configuration" / "SNN_module.yaml"
NETWORK_ARCH_YAML = Path(__file__).parent.parent / "configuration" / "network_architecture.yaml"

# Maps training.framework selector → neuron: / neuron_types: key in network_architecture.yaml
FW_TO_CFG_KEY = {
    "torch":    "snntorch",
    "norse":    "norse",
    "sj":       "spikingjelly",
    "sinabs":   "sinabs",
}


class Settings:
    def __init__(self, config: dict | None = None, overlay: str | None = None):
        """
        config   an already-merged config dict (from skeleton.config_loader.load_config).
                 Pass this when the CLI has already loaded and overlaid everything.
        overlay  path to one experiment overlay to merge over the three base files.

        Both omitted -- `Settings()` -- loads the three base files and nothing else,
        which is exactly how this pipeline behaved before overlays existed. That is
        what keeps `python learning/main.py` working with no arguments.
        """
        from skeleton.config_loader import load_config

        if config is not None and overlay is not None:
            raise ValueError("pass either `config` or `overlay`, not both")
        self.config = config if config is not None else load_config(overlay)
        self.overlay_path = overlay
        self.yaml_path = str(DEFAULT_YAML)  # kept: some callers print it

        training     = self.config.get("training", {})
        dataset      = self.config.get("dataset", {})
        output       = self.config.get("output", {})
        # NOTE: the old `frameworks:` block is gone -- neuron params moved to
        # network_architecture.yaml, optimizer/loss became one shared training setting.

        # The three base files are merged into one dict by the loader (their top-level
        # sections do not collide), so the conv-SNN architecture is read from the same
        # dict as everything else rather than from a second file read here.
        network_arch = self.config
        conv = network_arch.get("convolution", {})


        # Conv-SNN architecture (from network_architecture.yaml)
        self.SENSOR_H     = int(conv.get("sensor_h",     34))
        self.SENSOR_W     = int(conv.get("sensor_w",     34))
        self.IN_CHANNELS  = int(conv.get("in_channels",  2))
        self.CONV1_OUT    = int(conv.get("conv1_out",    12))
        self.CONV1_KERNEL = int(conv.get("conv1_kernel", 5))
        self.CONV2_OUT    = int(conv.get("conv2_out",    32))
        self.CONV2_KERNEL = int(conv.get("conv2_kernel", 5))
        self.POOL_KERNEL  = int(conv.get("pool_kernel",  2))

        # Auto-compute flattened size after both conv+pool stages
        self.FC_IN = self.compute_fc_in(self.SENSOR_H, self.SENSOR_W)

        # Output classes belong to the DATASET REGISTRY, not to this config: the input
        # sensor shape and the class count both arrive via apply_dataset_shape() from the
        # registry entry (learning/main.py, data_pipeline). Only the network INTERNALS --
        # filter counts, kernel sizes, pool size, hidden layers -- are configured here.
        #
        # No placeholder value: unset until apply_dataset_shape() runs, so building a
        # network before a dataset is picked fails loudly instead of silently getting a
        # wrong class count. See the NUM_CLASSES property below.
        self._num_classes = None

        self.NEURON_TYPES = network_arch.get("neuron_types", {})

        # The unified neuron spec: one block per framework, all describing the SAME
        # neuron in each framework's own units. Read through skeleton/neuron_spec.py,
        # which raises on a missing key rather than falling back to a framework default.
        # Deliberately NOT given per-key defaults here -- see that module's header.
        self.NEURON = network_arch.get("neuron", {})

        # Training parameters
        self.EPOCHS                   = int(training.get("epochs", 10))
        # null (or 0) = auto: one epoch is a full pass, derived from dataset size and
        # batch size. A number caps the epoch at that many batches (never more than a
        # full pass). Resolved once the loader exists -- see resolve_iterations().
        _itera                        = training.get("iterations_per_epoch", None)
        self.ITERA                    = None if _itera in (None, 0) else int(_itera)
        self.BATCH_SIZE               = int(training.get("batch_size", 128))
        # When False, BATCH_SIZE above is used as-is and calibrate_batch_size()
        # is never called — see docs/functions.md for why this exists.
        self.CALIBRATE_BATCH_SIZE     = bool(training.get("calibrate_batch_size", True))
        # Untimed forward+backward passes before the timed epochs -- see the YAML comment.
        self.WARMUP_ITERATIONS        = int(training.get("warmup_iterations", 5))

        # Fixes weight init AND batch order. Without it, any measured difference
        # between two frameworks is confounded with initialisation noise -- there was
        # no seeding anywhere in this pipeline before.
        self.SEED                     = int(training.get("seed", 0))

        # ONE optimizer and ONE loss, shared by every framework -- not per-framework.
        # These are plain torch; none of the four SNN libraries supplies them, so giving
        # a framework its own would mean comparing training recipes, not frameworks.
        optimizer_cfg                 = training.get("optimizer", {})
        self.OPTIMIZER                = str(optimizer_cfg.get("type", "nadam"))
        self.LEARNING_RATE            = float(optimizer_cfg.get("lr", 0.002))
        # 0.0 is this pipeline's value. The pre-merge default here was 1e-4, applied to
        # every framework -- a real recipe difference, recorded in docs/merge_decisions.md.
        self.WEIGHT_DECAY             = float(optimizer_cfg.get("weight_decay", 0.0))
        self.SGD_MOMENTUM             = float(optimizer_cfg.get("momentum", 0.0))
        self.LOSS_FN                  = str(training.get("loss", "cross_entropy"))
        self.DEVICE                    = training.get("device", "cuda")
        self.DDP                      = training.get("DDP", "OFF")
        self.USE_AMP                  = bool(training.get("use_amp", True))
        self.GRAD_ACCUM_STEPS         = max(1, int(training.get("grad_accum_steps", 1)))
        # Validated, not compared loosely: the use site tested `== "cosine"`, so any
        # unrecognised string (a typo like "cosinne") silently meant NO scheduler --
        # a training-recipe change with no error.
        self.LR_SCHEDULER             = str(training.get("lr_scheduler", "none")).lower()
        if self.LR_SCHEDULER not in ("none", "cosine"):
            raise ValueError(
                f"training.lr_scheduler={self.LR_SCHEDULER!r} is not supported. "
                "Options: none (constant lr), cosine (anneal to ~0 by the last epoch)."
            )

        self.TRADES_ENABLED           = bool(training.get("trades_enabled", False))
        self.TRADES_EPSILON           = float(training.get("trades_epsilon", 0.05))
        self.TRADES_LAMBDA            = float(training.get("trades_lambda", 6.0))
        self.TRADES_STEPS             = int(training.get("trades_steps", 10))

        self.ACTIVITY_REG_ENABLED     = bool(training.get("activity_reg_enabled", False))
        self.ACTIVITY_REG_MIN_RATE    = float(training.get("activity_reg_min_rate", 0.01))
        self.ACTIVITY_REG_MAX_RATE    = float(training.get("activity_reg_max_rate", 0.50))
        self.ACTIVITY_REG_LAMBDA_LOW  = float(training.get("activity_reg_lambda_low", 0.1))
        self.ACTIVITY_REG_LAMBDA_HIGH = float(training.get("activity_reg_lambda_high", 0.1))

        # Framework selector
        self.FRAMEWORK = training.get("framework", "norse")


        # Dataset control
        # None (the default) means "ask" -- the interactive prompt, as this pipeline
        # has always worked. Set dataset.name in the config to skip it, which anything
        # non-interactive (a Colab cell, a scripted sweep) needs. An unrecognised name
        # raises at startup rather than falling back; see dataset_registry.lookup_dataset.
        self.DATASET_NAME = dataset.get("name", dataset.get("dataset_name", None))
        self.TASK_TYPE    = "classification"  # overwritten by NeuromorphicEncoder.load_raw() once a dataset is picked

        # Output control
        self.OUTPUT_DIR = output.get("output_dir", "./outputs")
        self.PLOT_DIR   = output.get("plot_dir",   "./outputs/plots")
        self.DATA_DIR   = output.get("data_dir",   "./outputs/data")


    @property
    def NUM_CLASSES(self) -> int:
        """Class count from the dataset registry, set once by apply_dataset_shape().

        Raises AttributeError (not a made-up default) when read before that -- a network
        built off a guessed class count would silently have the wrong output layer.
        AttributeError specifically, so getattr(cfg, "NUM_CLASSES", None) (display(),
        below) still gets its intended "not set yet" None rather than the error itself.
        """
        if self._num_classes is None:
            raise AttributeError(
                "cfg.NUM_CLASSES read before apply_dataset_shape() ran -- class count "
                "comes from the dataset registry, not a config default."
            )
        return self._num_classes

    @NUM_CLASSES.setter
    def NUM_CLASSES(self, value: int) -> None:
        self._num_classes = value

    @property
    def active_fw_cfg(self) -> dict:
        """The neuron spec block for whichever framework is currently selected.

        Was the per-framework `frameworks:` block in SNN_module.yaml, which mixed
        neuron parameters with optimizer and loss. The neuron moved to
        network_architecture.yaml's `neuron:` section; optimizer and loss became one
        shared setting, since neither belongs to any of the four SNN libraries.
        """
        if self.FRAMEWORK not in FW_TO_CFG_KEY:
            raise ValueError(
                f"training.framework='{self.FRAMEWORK}' has no FW_TO_CFG_KEY mapping. "
                f"Available: {sorted(FW_TO_CFG_KEY)}."
            )
        return self.NEURON.get(FW_TO_CFG_KEY[self.FRAMEWORK], {})

    def compute_fc_in(self, sensor_h: int, sensor_w: int) -> int:
        """Flattened size after both conv+pool stages — independent H/W so non-square sensors work."""
        h = (sensor_h - self.CONV1_KERNEL + 1) // self.POOL_KERNEL
        h = (h - self.CONV2_KERNEL + 1) // self.POOL_KERNEL
        w = (sensor_w - self.CONV1_KERNEL + 1) // self.POOL_KERNEL
        w = (w - self.CONV2_KERNEL + 1) // self.POOL_KERNEL
        return self.CONV2_OUT * h * w

    def apply_dataset_shape(self, sensor_h: int, sensor_w: int, in_channels: int, num_classes: int):
        """Override conv-input shape and output classes with the selected dataset's actual
        sensor size / class count (from DATASET_REGISTRY), and recompute the dependent
        flattened FC input size. Must run before the model is constructed.

        Regression datasets (DAVIS Camera Pose, DSEC) carry a placeholder num_classes=1
        here — not a real class count. Their actual output shaping is a documented
        follow-up (see docs/Haseeb-open-items.md), not built yet."""
        self.SENSOR_H    = int(sensor_h)
        self.SENSOR_W    = int(sensor_w)
        self.IN_CHANNELS = int(in_channels)
        self.NUM_CLASSES = int(num_classes)
        self.FC_IN       = self.compute_fc_in(self.SENSOR_H, self.SENSOR_W)

    def resolve_iterations(self, train_loader) -> int:
        """Settle ITERA against the real loader. Call once, after the loader is built.

        A full pass is len(DataLoader), NOT ceil(samples / batch): the loader's own
        __len__ already accounts for drop_last=True (floor, not ceil), so this matches
        what actually executes for any drop_last setting.

        Deliberately NOT gated on CALIBRATE_BATCH_SIZE. Epoch length follows from dataset
        size and batch size; whether the batch size came from a VRAM probe or from the
        config is a separate concern. Previously the derivation only ran when the probe
        was on, so turning the probe OFF left a literal YAML value as a hard cap -- e.g.
        400 against a 468-batch full pass silently trained on 85% of the data per epoch.
        """
        loader = getattr(train_loader, "loader", train_loader)
        full_pass = len(loader)
        if self.ITERA is None:
            self.ITERA = full_pass
        else:
            self.ITERA = min(self.ITERA, full_pass)
        return self.ITERA

    def load_yaml(self, yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def display(self):
        W      = 76
        fw     = self.FRAMEWORK.upper()
        fw_cfg = self.active_fw_cfg
        sep    = "-" * (W - 4)

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

        section("ARCHITECTURE")
        row("Sensor",          f"{self.SENSOR_H} × {self.SENSOR_W}   ({self.IN_CHANNELS} channels)")
        row("Conv1",           f"{self.CONV1_OUT} filters   {self.CONV1_KERNEL}×{self.CONV1_KERNEL} kernel")
        row("Conv2",           f"{self.CONV2_OUT} filters   {self.CONV2_KERNEL}×{self.CONV2_KERNEL} kernel")
        row("Pool",            f"{self.POOL_KERNEL}×{self.POOL_KERNEL} MaxPool   (applied twice)")
        row("FC input (auto)", str(self.FC_IN))
        num_classes = getattr(self, "NUM_CLASSES", None)
        row("Output classes",  str(num_classes) if num_classes is not None else "N/A (regression target)")

        cfg_key      = FW_TO_CFG_KEY[self.FRAMEWORK]
        neuron_types = self.NEURON_TYPES.get(cfg_key, {})
        section(f"NEURON TYPES — {fw}")
        for layer, ntype in neuron_types.items():
            row(layer, ntype)

        section(f"NEURON SPEC — {fw}")
        if not fw_cfg:
            row("(none)", f"network_architecture.yaml has no neuron.{cfg_key} block")
        for key, val in fw_cfg.items():
            if isinstance(val, dict):  # e.g. surrogate: {type:..., alpha:...}
                inner = ", ".join(f"{k}={v}" for k, v in val.items())
                row(key, inner)
            else:
                display_val = f"{val} Hz" if key == "tau_mem_inv" else str(val)
                row(key, display_val)

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
        if self.TRADES_ENABLED:
            row("TRADES",       f"ENABLED   eps={self.TRADES_EPSILON}   lambda={self.TRADES_LAMBDA}   steps={self.TRADES_STEPS}")
        else:
            row("TRADES",       "DISABLED")
        if self.ACTIVITY_REG_ENABLED:
            row("Activity reg", f"ENABLED   min={self.ACTIVITY_REG_MIN_RATE * 100:.0f}%   max={self.ACTIVITY_REG_MAX_RATE * 100:.0f}%")
        else:
            row("Activity reg", "DISABLED")

        section("DATASET")
        row("Dataset",   self.DATASET_NAME)

        section("OUTPUT")
        row("Output dir", self.OUTPUT_DIR)
        row("Plot dir",   self.PLOT_DIR)
        row("Data dir",   self.DATA_DIR)

        print()
        print("=" * W)
        print()

if __name__ == "__main__":
    cfg = Settings()
    cfg.display()
