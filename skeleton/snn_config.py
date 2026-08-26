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
    def __init__(self, yaml_path=str(DEFAULT_YAML)):
        self.yaml_path = yaml_path
        self.config = self.load_yaml(yaml_path)

        architecture = self.config.get("architecture", {})
        training     = self.config.get("training", {})
        dataset      = self.config.get("dataset", {})
        output       = self.config.get("output", {})
        # NOTE: the old `frameworks:` block is gone -- neuron params moved to
        # network_architecture.yaml, optimizer/loss became one shared training setting.

        # Load conv-SNN architecture from network_architecture.yaml
        network_arch = self.load_yaml(str(NETWORK_ARCH_YAML))
        conv = network_arch.get("convolution", {})

        # Legacy MLP architecture params (kept for backward compatibility)
        self.INPUT_SIZE              = int(architecture.get("input_size", 10))
        self.HIDDEN_SIZE             = int(architecture.get("hidden_size", 16))
        self.HIDDEN_LAYERS           = int(architecture.get("hidden_layers", 3))
        self.OUTPUT_SIZE             = int(architecture.get("output_size", 10))
        self.LEAK                    = float(architecture.get("leak", 1.0))
        self.OVERRIDE                = bool(architecture.get("override", False))
        self.NETWORK_STRUCT          = architecture.get("network_struct", "S")
        self.SIMULATOR               = architecture.get("simulator", "OFF")
        self.TEMPORAL_SLICE_DURATION = int(architecture.get("temporal_slice_duration", 15000))
        self.TEMPORAL_OVERLAP        = int(architecture.get("temporal_overlap", 0))
        self.TOTAL_TIME_WINDOW       = int(architecture.get("total_time_window", 30000))

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

        self.NEURON_TYPES = network_arch.get("neuron_types", {})

        # The unified neuron spec: one block per framework, all describing the SAME
        # neuron in each framework's own units. Read through skeleton/neuron_spec.py,
        # which raises on a missing key rather than falling back to a framework default.
        # Deliberately NOT given per-key defaults here -- see that module's header.
        self.NEURON = network_arch.get("neuron", {})

        # Training parameters
        self.EPOCHS                   = int(training.get("epochs", 10))
        self.ITERA                    = int(training.get("iterations_per_epoch", 100))
        self.BATCH_SIZE               = int(training.get("batch_size", 128))
        # When False, BATCH_SIZE above is used as-is and calibrate_batch_size()
        # is never called — see docs/functions.md for why this exists.
        self.CALIBRATE_BATCH_SIZE     = bool(training.get("calibrate_batch_size", True))
        self.NAP_TIMES                = int(training.get("nap_times", 1))

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
        self.ENABLE_PIPELINE_MONITOR  = bool(training.get("enable_pipeline_monitor", True))
        self.LR_SCHEDULER             = training.get("lr_scheduler", "cosine")

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
        self.DATASET_NAME = dataset.get("dataset_name", "MNIST")
        self.TASK_TYPE    = "classification"  # overwritten by NeuromorphicEncoder.load_raw() once a dataset is picked

        # Output control
        self.OUTPUT_DIR = output.get("output_dir", "./outputs")
        self.PLOT_DIR   = output.get("plot_dir",   "./outputs/plots")
        self.DATA_DIR   = output.get("data_dir",   "./outputs/data")

        # Generated network structure
        self.network_structure = self.generate_network_structure()


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
        self.network_structure = self.generate_network_structure()

    def load_yaml(self, yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def generate_network_structure(self):
        """
        Generates a list representing the neuron count per layer:
        [input_layer, hidden1, hidden2, ..., output_layer]

        NETWORK_STRUCT options:
        S = stable
        A = ascending
        D = descending
        """

        layers = []

        # Append input layer separately
        layers.append(self.INPUT_SIZE)

        # Generate hidden layers independently from input size
        if self.OVERRIDE:
            if self.NETWORK_STRUCT == "S" or self.NETWORK_STRUCT is None:
                hidden_layers = [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

            elif self.NETWORK_STRUCT == "A":
                hidden_layers = []
                current = self.HIDDEN_SIZE

                for _ in range(self.HIDDEN_LAYERS):
                    hidden_layers.append(current)
                    current += 4

            elif self.NETWORK_STRUCT == "D":
                hidden_layers = []
                current = self.HIDDEN_SIZE

                for _ in range(self.HIDDEN_LAYERS):
                    hidden_layers.append(current)
                    current = max(2, current // 2)

            else:
                raise ValueError("Invalid NETWORK_STRUCT. Use 'S', 'A', or 'D'.")

        else:
            hidden_layers = [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

        # Append hidden layers
        layers.extend(hidden_layers)

        # Append output layer separately — the real per-dataset class count
        # once apply_dataset_shape() has run, else the legacy YAML default.
        layers.append(getattr(self, "NUM_CLASSES", self.OUTPUT_SIZE))

        return layers

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
        row("Network structure", " -> ".join(str(n) for n in self.network_structure))

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
