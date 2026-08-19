import yaml
from pathlib import Path

DEFAULT_YAML      = Path(__file__).parent.parent / "configuration" / "SNN_module.yaml"
NETWORK_ARCH_YAML = Path(__file__).parent.parent / "configuration" / "network_architecture.yaml"

# Maps training.framework selector → FRAMEWORK_CFG / NEURON_TYPES key
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
        input_cfg    = self.config.get("input", {})
        output       = self.config.get("output", {})
        frameworks   = self.config.get("frameworks", {})

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
        self.NUM_WORKERS             = int(architecture.get("num_workers", 2))

        # Input control (reserved for future use — expose when input_mode is needed)
        # self.INPUT_MODE      = input_cfg.get("input_mode", "2D")
        # self.IMAGE_CHANNELS  = int(input_cfg.get("image_channels", 1))
        # self.IMAGE_HEIGHT    = int(input_cfg.get("image_height", 28))
        # self.IMAGE_WIDTH     = int(input_cfg.get("image_width", 28))

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

        # Training parameters
        self.EPOCHS                   = int(training.get("epochs", 10))
        self.ITERA                    = int(training.get("iterations_per_epoch", 100))
        self.TIMESTEPS                = int(training.get("timesteps", 25))
        self.BATCH_SIZE               = int(training.get("batch_size", 128))
        self.NAP_TIMES                = int(training.get("nap_times", 1))
        self.LEARNING_RATE             = float(training.get("learning_rate", 0.001))
        self.WEIGHT_DECAY              = float(training.get("weight_decay", 0.0001))
        self.DEVICE                    = training.get("device", "cuda")
        self.DDP                      = training.get("DDP", "OFF")
        self.NUM_WORKERS              = int(training.get("num_workers", 4))
        self.PREFETCH_DEPTH            = int(training.get("prefetch_depth", 8))
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

        # Per-framework config blocks
        snt = frameworks.get("snntorch",     {})
        nor = frameworks.get("norse",        {})
        spj = frameworks.get("spikingjelly", {})
        sin = frameworks.get("sinabs",       {})
        bds = frameworks.get("bindsnet",     {})
        spx = frameworks.get("spyx",         {})

        self.FRAMEWORK_CFG = {
            "snntorch": {
                "beta":      float(snt.get("beta", 0.95)),
                "threshold": float(snt.get("threshold", 0.5)),
                "optimizer": snt.get("optimizer", "adam"),
                "loss_fn":   snt.get("loss_fn", "mse_count"),
            },
            "norse": {
                "tau_mem_inv": float(nor.get("tau_mem_inv", 100.0)),
                "threshold":   float(nor.get("threshold", 0.5)),
                "optimizer":   nor.get("optimizer", "adam"),
                "loss_fn":     nor.get("loss_fn", "cross_entropy"),
            },
            "spikingjelly": {
                "tau":       float(spj.get("tau", 2.0)),
                "threshold": float(spj.get("threshold", 0.5)),
                "optimizer": spj.get("optimizer", "adam"),
                "loss_fn":   spj.get("loss_fn", "cross_entropy"),
            },
            "sinabs": {
                "tau_mem":   float(sin.get("tau_mem", 20.0)),
                "threshold": float(sin.get("threshold", 0.5)),
                "optimizer": sin.get("optimizer", "adam"),
                "loss_fn":   sin.get("loss_fn", "cross_entropy"),
            },
            "bindsnet": {
                "nu_pre":    float(bds.get("nu_pre", 0.0001)),
                "nu_post":   float(bds.get("nu_post", 0.01)),
                "threshold": float(bds.get("threshold", 0.5)),
                "optimizer": bds.get("optimizer", "none"),
                "loss_fn":   bds.get("loss_fn", "cross_entropy"),
            },
            "spyx": {
                "beta":      float(spx.get("beta", 0.9)),
                "gamma":     float(spx.get("gamma", 0.9)),
                "threshold": float(spx.get("threshold", 0.5)),
                "optimizer": spx.get("optimizer", "adam"),
                "loss_fn":   spx.get("loss_fn", "cross_entropy"),
            },
        }

        # Backward-compatible shorthands
        self.BETA        = self.FRAMEWORK_CFG["snntorch"]["beta"]
        self.TAU_MEM_INV = self.FRAMEWORK_CFG["norse"]["tau_mem_inv"]
        self.TAU         = self.FRAMEWORK_CFG["spikingjelly"]["tau"]

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
        """Config dict for whichever framework is currently selected."""
        return self.FRAMEWORK_CFG[FW_TO_CFG_KEY[self.FRAMEWORK]]

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

        # Append output layer separately
        layers.append(self.OUTPUT_SIZE)

        return layers

    def display(self):
        W      = 76
        fw     = self.FRAMEWORK.upper()
        fw_cfg = self.active_fw_cfg
        sep    = "─" * (W - 4)

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
        row("Network structure", " → ".join(str(n) for n in self.network_structure))

        cfg_key      = FW_TO_CFG_KEY[self.FRAMEWORK]
        neuron_types = self.NEURON_TYPES.get(cfg_key, {})
        section(f"NEURON TYPES — {fw}")
        for layer, ntype in neuron_types.items():
            row(layer, ntype)

        section(f"FRAMEWORK PARAMS — {fw}")
        for key, val in fw_cfg.items():
            display_val = f"{val} Hz" if key == "tau_mem_inv" else str(val)
            row(key, display_val)

        section("TRAINING")
        row("Epochs",             str(self.EPOCHS))
        row("Iterations / epoch", str(self.ITERA))
        row("Timesteps (T)",      str(self.TIMESTEPS))
        row("Batch size",         str(self.BATCH_SIZE))
        row("Learning rate",      str(self.LEARNING_RATE))
        row("Weight decay",       str(self.WEIGHT_DECAY))
        row("LR scheduler",       self.LR_SCHEDULER)
        row("Grad accum steps",   str(self.GRAD_ACCUM_STEPS))
        row("AMP (mixed prec.)",  "ENABLED" if self.USE_AMP else "DISABLED")
        row("DataLoader workers", str(self.NUM_WORKERS))

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
