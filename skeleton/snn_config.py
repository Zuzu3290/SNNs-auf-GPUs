import yaml
from pathlib import Path

from skeleton.strict import ConfigKeyError, section

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

        # Every block below is read through skeleton/strict.py: the key must be present
        # and the right type, or the run stops. See that module for the typo that
        # motivated it.
        training     = section(self.config, "training")
        dataset      = section(self.config, "dataset")
        output       = section(self.config, "output")
        # NOTE: the old `frameworks:` block is gone -- neuron params moved to
        # network_architecture.yaml, optimizer/loss became one shared training setting.

        # The three base files are merged into one dict by the loader (their top-level
        # sections do not collide), so the conv-SNN architecture is read from the same
        # dict as everything else rather than from a second file read here.
        network_arch = self.config
        conv = section(self.config, "convolution")

        # The conv-SNN architecture: ONLY the network internals. Filter counts, kernel
        # sizes and the pool size are yours to choose; the input shape and the class
        # count are not -- see the properties below.
        self.CONV1_OUT    = conv.require_int("conv1_out")
        self.CONV1_KERNEL = conv.require_int("conv1_kernel")
        self.CONV2_OUT    = conv.require_int("conv2_out")
        self.CONV2_KERNEL = conv.require_int("conv2_kernel")
        self.POOL_KERNEL  = conv.require_int("pool_kernel")

        # Input shape and output classes belong to the DATASET REGISTRY, not to this
        # config. sensor_h / sensor_w / in_channels / num_classes were removed from
        # network_architecture.yaml because apply_dataset_shape() overwrote all four on
        # every real run -- they read like settings while changing nothing, and a reader
        # editing sensor_h to 128 would have seen it silently ignored.
        #
        # No placeholder values: unset until apply_dataset_shape() runs, so building a
        # network before a dataset is picked fails loudly instead of quietly getting the
        # wrong input shape or output width. FC_IN follows from all of them.
        self._sensor_h = None
        self._sensor_w = None
        self._in_channels = None
        self._num_classes = None
        self._fc_in = None

        self.NEURON_TYPES = network_arch.get("neuron_types", {})

        # The unified neuron spec: one block per framework, all describing the SAME
        # neuron in each framework's own units. Read through skeleton/neuron_spec.py,
        # which raises on a missing key rather than falling back to a framework default.
        # Deliberately NOT given per-key defaults here -- see that module's header.
        self.NEURON = network_arch.get("neuron", {})

        # Training parameters
        self.EPOCHS                   = training.require_int("epochs")
        # null (or 0) = auto: one epoch is a full pass, derived from dataset size and
        # batch size. A number caps the epoch at that many batches (never more than a
        # full pass). Resolved once the loader exists -- see resolve_iterations().
        _itera                        = training.optional_int("iterations_per_epoch")
        self.ITERA                    = None if _itera in (None, 0) else _itera
        self.BATCH_SIZE               = training.require_int("batch_size")
        # When False, BATCH_SIZE above is used as-is and calibrate_batch_size()
        # is never called — see docs/functions.md for why this exists.
        self.CALIBRATE_BATCH_SIZE     = training.require_bool("calibrate_batch_size")
        # Untimed forward+backward passes before the timed epochs -- see the YAML comment.
        self.WARMUP_ITERATIONS        = training.require_int("warmup_iterations")
        # Samples for the batch-size-1 latency pass; 0 skips it entirely.
        self.LATENCY_SAMPLES          = training.require_int("latency_samples")

        # Fixes weight init AND batch order. Without it, any measured difference
        # between two frameworks is confounded with initialisation noise -- there was
        # no seeding anywhere in this pipeline before.
        self.SEED                     = training.require_int("seed")

        # ONE optimizer and ONE loss, shared by every framework -- not per-framework.
        # These are plain torch; none of the four SNN libraries supplies them, so giving
        # a framework its own would mean comparing training recipes, not frameworks.
        optimizer_cfg                 = training.sub("optimizer")
        self.OPTIMIZER                = optimizer_cfg.require_str("type")
        self.LEARNING_RATE            = optimizer_cfg.require_float("lr")
        # 0.0 is this pipeline's value. The pre-merge default here was 1e-4, applied to
        # every framework -- a real recipe difference, recorded in docs/merge_decisions.md.
        self.WEIGHT_DECAY             = optimizer_cfg.require_float("weight_decay")
        self.SGD_MOMENTUM             = optimizer_cfg.require_float("momentum")
        self.LOSS_FN                  = training.require_str("loss")
        self.DEVICE                   = training.require_str("device")
        self.USE_AMP                  = training.require_bool("use_amp")
        # Scalability-study-only diagnostics (Participation Ratio, mutual information,
        # spike entropy, per-layer gradient norms). Off by default: every other
        # experiment's runs.csv/layers.csv output is unaffected either way.
        self.COMPUTE_CAPACITY_METRICS = training.require_bool("compute_capacity_metrics")
        self.GRAD_ACCUM_STEPS         = max(1, training.require_int("grad_accum_steps"))
        # Validated, not compared loosely: the use site tested `== "cosine"`, so any
        # unrecognised string (a typo like "cosinne") silently meant NO scheduler --
        # a training-recipe change with no error.
        self.LR_SCHEDULER             = training.require_choice("lr_scheduler",
                                                                 ["none", "cosine"])

        self.TRADES_ENABLED           = training.require_bool("trades_enabled")
        self.TRADES_EPSILON           = training.require_float("trades_epsilon")
        self.TRADES_LAMBDA            = training.require_float("trades_lambda")
        self.TRADES_STEPS             = training.require_int("trades_steps")

        self.ACTIVITY_REG_ENABLED     = training.require_bool("activity_reg_enabled")
        self.ACTIVITY_REG_MIN_RATE    = training.require_float("activity_reg_min_rate")
        self.ACTIVITY_REG_MAX_RATE    = training.require_float("activity_reg_max_rate")
        self.ACTIVITY_REG_LAMBDA_LOW  = training.require_float("activity_reg_lambda_low")
        self.ACTIVITY_REG_LAMBDA_HIGH = training.require_float("activity_reg_lambda_high")

        # Framework selector
        self.FRAMEWORK = training.require_choice("framework", sorted(FW_TO_CFG_KEY))


        # Dataset control
        # null means "ask" -- the interactive prompt, as this pipeline has always
        # worked. The KEY must still be present: `dataset.name: null` is a stated
        # choice, an absent key is a mistake. Set a name to skip the prompt, which
        # anything non-interactive (a Colab cell, a scripted sweep) needs. An
        # unrecognised name raises at startup rather than falling back; see
        # dataset_registry.lookup_dataset.
        self.DATASET_NAME = dataset.optional_str("name")
        self.TASK_TYPE    = "classification"  # overwritten by NeuromorphicEncoder.load_raw() once a dataset is picked

        # Output control -- the ./outputs layout, used when no --experiment routes the
        # run elsewhere. See learning/main.py for the routed case.
        self.OUTPUT_DIR = output.require_str("output_dir")
        self.PLOT_DIR   = output.require_str("plot_dir")
        self.DATA_DIR   = output.require_str("data_dir")


    # ---- shape, owned by the dataset registry -----------------------------------
    #
    # All five raise AttributeError (not a made-up default) when read before
    # apply_dataset_shape() has run. A network built off a guessed input shape or class
    # count would be silently wrong rather than absent, and every downstream number --
    # parameter count, FC width, accuracy -- would look plausible.
    #
    # AttributeError specifically, so getattr(cfg, "NUM_CLASSES", None) in display()
    # still gets its intended "not set yet" None rather than the exception itself.
    @staticmethod
    def _unset(name: str, what: str) -> AttributeError:
        return AttributeError(
            f"cfg.{name} read before apply_dataset_shape() ran -- {what} comes from the "
            "dataset registry, not from network_architecture.yaml."
        )

    @property
    def SENSOR_H(self) -> int:
        if self._sensor_h is None:
            raise self._unset("SENSOR_H", "sensor height")
        return self._sensor_h

    @SENSOR_H.setter
    def SENSOR_H(self, value: int) -> None:
        self._sensor_h = int(value)

    @property
    def SENSOR_W(self) -> int:
        if self._sensor_w is None:
            raise self._unset("SENSOR_W", "sensor width")
        return self._sensor_w

    @SENSOR_W.setter
    def SENSOR_W(self, value: int) -> None:
        self._sensor_w = int(value)

    @property
    def IN_CHANNELS(self) -> int:
        if self._in_channels is None:
            raise self._unset("IN_CHANNELS", "the channel count")
        return self._in_channels

    @IN_CHANNELS.setter
    def IN_CHANNELS(self, value: int) -> None:
        self._in_channels = int(value)

    @property
    def NUM_CLASSES(self) -> int:
        if self._num_classes is None:
            raise self._unset("NUM_CLASSES", "the class count")
        return self._num_classes

    @NUM_CLASSES.setter
    def NUM_CLASSES(self, value: int) -> None:
        self._num_classes = int(value)

    @property
    def FC_IN(self) -> int:
        """Flattened width into the classifier. Derived from the sensor shape and the
        conv/pool sizes, never configured -- see compute_fc_in()."""
        if self._fc_in is None:
            raise self._unset("FC_IN", "the flattened width (it follows from the sensor)")
        return self._fc_in

    @FC_IN.setter
    def FC_IN(self, value: int) -> None:
        self._fc_in = int(value)

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

    def display(self, output_dirs: dict | None = None):
        """Print the configuration this run is about to use.

        `output_dirs` overrides the OUTPUT section. The config's own OUTPUT_DIR /
        PLOT_DIR / DATA_DIR describe the unrouted layout, so on a run started with
        --experiment (and, on Colab, --results-root pointing at mounted Drive) they
        name a directory nothing was written to. Left to itself this block said
        `./outputs` while every artefact went to Drive. Callers that route output pass
        the real destinations; callers that do not still get the config's own values,
        so the original flow is unchanged.
        """
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

        # The four dataset-owned values raise until apply_dataset_shape() has run, and a
        # REPORT must never be the thing that stops a run -- so each is read through
        # getattr and shown as "not set yet" instead. The rest of the block is config,
        # always present by the time Settings exists.
        def shape(name: str) -> str:
            value = getattr(self, name, None)
            return "not set yet" if value is None else str(value)

        sensor_h, sensor_w = shape("SENSOR_H"), shape("SENSOR_W")
        section("ARCHITECTURE")
        if "not set yet" in (sensor_h, sensor_w):
            row("Sensor", "not set yet -- comes from the dataset registry")
        else:
            row("Sensor", f"{sensor_h} × {sensor_w}   ({shape('IN_CHANNELS')} channels)")
        row("Conv1",           f"{self.CONV1_OUT} filters   {self.CONV1_KERNEL}×{self.CONV1_KERNEL} kernel")
        row("Conv2",           f"{self.CONV2_OUT} filters   {self.CONV2_KERNEL}×{self.CONV2_KERNEL} kernel")
        row("Pool",            f"{self.POOL_KERNEL}×{self.POOL_KERNEL} MaxPool   (applied twice)")
        row("FC input (auto)", shape("FC_IN"))
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
        if output_dirs:
            for label, value in output_dirs.items():
                row(label, value)
        else:
            row("Output dir", self.OUTPUT_DIR)
            row("Plot dir",   self.PLOT_DIR)
            row("Data dir",   self.DATA_DIR)

        print()
        print("=" * W)
        print()

if __name__ == "__main__":
    cfg = Settings()
    cfg.display()
