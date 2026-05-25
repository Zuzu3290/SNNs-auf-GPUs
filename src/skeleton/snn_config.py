import yaml
from pathlib import Path

# Absolute path to SNN_module.yaml — works regardless of working directory
_DEFAULT_YAML = Path(__file__).parent.parent.parent / "SNN_module.yaml"

class Settings:
    def __init__(self, yaml_path=str(_DEFAULT_YAML)):
        self.yaml_path = yaml_path
        self.config = self.load_yaml_config(yaml_path)

        architecture  = self.config.get("architecture", {})
        training      = self.config.get("training", {})
        frameworks    = self.config.get("frameworks", {})
        dataset       = self.config.get("dataset", {})
        input_cfg     = self.config.get("input", {})
        output        = self.config.get("output", {})

        # Conv-SNN architecture — shared across all frameworks for fair comparison
        self.SENSOR_H    = int(architecture.get("sensor_h",     34))
        self.SENSOR_W    = int(architecture.get("sensor_w",     34))
        self.IN_CHANNELS = int(architecture.get("in_channels",  2))
        self.CONV1_OUT   = int(architecture.get("conv1_out",    12))
        self.CONV1_KERNEL= int(architecture.get("conv1_kernel", 5))
        self.CONV2_OUT   = int(architecture.get("conv2_out",    32))
        self.CONV2_KERNEL= int(architecture.get("conv2_kernel", 5))
        self.POOL_KERNEL = int(architecture.get("pool_kernel",  2))
        self.OUTPUT_SIZE = int(architecture.get("output_size",  10))
        self.THRESHOLD   = float(architecture.get("threshold",  1.0))
        self.RESET_MODE  = str(architecture.get("reset_mode",   "zero"))  # "zero" | "subtract"; Norse only supports "zero"

        # FC_IN computed from architecture — change conv/pool params above, this updates automatically
        _h = (self.SENSOR_H - self.CONV1_KERNEL + 1) // self.POOL_KERNEL
        _h = (_h - self.CONV2_KERNEL + 1) // self.POOL_KERNEL
        self.FC_IN = self.CONV2_OUT * _h * _h

        # Training parameters
        self.FRAMEWORK  = training.get("framework", "torch")
        self.EPOCHS     = int(training.get("epochs", 10))
        self.ITERA      = int(training.get("iterations_per_epoch", 937))
        self.TIMESTEPS  = int(training.get("timesteps", 16))
        self.BATCH_SIZE = int(training.get("batch_size", 64))
        self.LEARNING_RATE = float(training.get("learning_rate", 0.001))
        self.WEIGHT_DECAY  = float(training.get("weight_decay", 0.0001))
        self.NUM_CLASSES   = int(training.get("num_classes", self.OUTPUT_SIZE))
        self.DEVICE        = training.get("device", "cuda")
        self.NUM_WORKERS   = int(training.get("num_workers", 4))
        self.USE_AMP          = bool(training.get("use_amp", True))
        self.GRAD_ACCUM_STEPS = max(1, int(training.get("grad_accum_steps", 1)))
        self.LR_SCHEDULER     = training.get("lr_scheduler", "cosine")

        # Playground — swap without touching Python code
        self.LOSS_FN        = training.get("loss_fn",   "cross_entropy")
        self.OPTIMIZER_TYPE = training.get("optimizer", "adam")
        self.SURROGATE      = training.get("surrogate", "atan")

        # Framework-specific neuron parameters — each framework reads its own section
        self.BETA        = float(frameworks.get("snntorch",     {}).get("beta",        0.95))
        self.TAU_MEM_INV = float(frameworks.get("norse",        {}).get("tau_mem_inv", 50.0))
        self.TAU         = float(frameworks.get("spikingjelly", {}).get("tau",         2.0))

        # Dataset control
        self.DATASET_NAME = dataset.get("dataset_name", "MNIST")
        self.DATA_PATH = dataset.get("data_path", "./data")

        # # Input control
        # self.INPUT_MODE = input_cfg.get("input_mode", "2D")
        # self.IMAGE_CHANNELS = int(input_cfg.get("image_channels", 1))
        # self.IMAGE_HEIGHT = int(input_cfg.get("image_height", 28))
        # self.IMAGE_WIDTH = int(input_cfg.get("image_width", 28))

        # Output control
        self.OUTPUT_DIR = output.get("output_dir", "./outputs")
        self.PLOT_DIR = output.get("plot_dir", "./outputs/plots")
        self.DATA_DIR = output.get("data_dir", "./outputs/data")

    def load_yaml_config(self, yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def display(self):
        print("=" * 60)
        print("SNN Configuration")
        print("=" * 60)

        print(f"Framework            : {self.FRAMEWORK}")
        print(f"Architecture")
        print(f"  Sensor             : {self.SENSOR_H}x{self.SENSOR_W}  ({self.IN_CHANNELS} channels)")
        print(f"  Conv1              : {self.CONV1_OUT} filters, {self.CONV1_KERNEL}x{self.CONV1_KERNEL}")
        print(f"  Conv2              : {self.CONV2_OUT} filters, {self.CONV2_KERNEL}x{self.CONV2_KERNEL}")
        print(f"  Pool kernel        : {self.POOL_KERNEL}")
        print(f"  FC_IN (auto)       : {self.FC_IN}")
        print(f"  Threshold          : {self.THRESHOLD}")
        print(f"Training")
        print(f"  Epochs             : {self.EPOCHS}")
        print(f"  Batch size         : {self.BATCH_SIZE}")
        print(f"  Learning rate      : {self.LEARNING_RATE}")
        print(f"  Device             : {self.DEVICE}")
        print(f"Playground")
        print(f"  Loss               : {self.LOSS_FN}")
        print(f"  Optimizer          : {self.OPTIMIZER_TYPE}")
        print(f"  Surrogate          : {self.SURROGATE}")

        print("=" * 60)

if __name__ == "__main__":
    cfg = Settings()
    cfg.display()