import yaml
from pathlib import Path

# Absolute path to SNN_module.yaml — works regardless of working directory
_DEFAULT_YAML = Path(__file__).parent.parent.parent / "SNN_module.yaml"

class Settings:
    def __init__(self, yaml_path=str(_DEFAULT_YAML)):
        self.yaml_path = yaml_path
        self.config = self.load_yaml_config(yaml_path)

        architecture = self.config.get("architecture", {})
        training = self.config.get("training", {})
        dataset = self.config.get("dataset", {})
        input_cfg = self.config.get("input", {})
        output = self.config.get("output", {})

        # Neural network architecture
        self.INPUT_SIZE = int(architecture.get("input_size", 10))
        self.HIDDEN_SIZE = int(architecture.get("hidden_size", 16))
        self.HIDDEN_LAYERS = int(architecture.get("hidden_layers", 3))
        self.OUTPUT_SIZE = int(architecture.get("output_size", 10))
        self.THRESHOLD = float(architecture.get("threshold", 0.5))
        self.LEAK = float(architecture.get("leak", 1.0))
        self.OVERRIDE = bool(architecture.get("override", False))
        self.NETWORK_STRUCT = architecture.get("network_struct", "S")
        self.SIMULATOR = architecture.get("simulator", "OFF")
        self.TEMPORAL_SLICE_DURATION = int(architecture.get("temporal_slice_duration", 15000))
        self.TEMPORAL_OVERLAP = int(architecture.get("temporal_overlap", 0))
        self.TOTAL_TIME_WINDOW = int(architecture.get("total_time_window", 30000))
        self.NUM_WORKERS = int(architecture.get("num_workers", 2))

        # Training parameters
        self.LOSS_FUNCTION = training.get("loss_function", "CrossEntropy")
        self.OPTIMIZER = training.get("optimizer", "Adam")
        self.EPOCHS = int(training.get("epochs", 10))
        self.ITERA = int(training.get("iterations_per_epoch", 100))
        self.TIMESTEPS = int(training.get("timesteps", 25))
        self.BATCH_SIZE = int(training.get("batch_size", 128))
        self.BETA = float(training.get("beta", 0.95))
        self.NAP_TIMES = int(training.get("nap_times", 1))
        self.LEARNING_RATE = float(training.get("learning_rate", 0.001))
        self.WEIGHT_DECAY = float(training.get("weight_decay", 0.0001))
        self.NUM_CLASSES = int(training.get("num_classes", self.OUTPUT_SIZE))
        self.DEVICE = training.get("device", "cuda")
        self.KERNEL = training.get("kernel", "OFF")
        self.DDP = training.get("DDP", "OFF")
        self.NUM_WORKERS = int(training.get("num_workers", 4))
        self.USE_AMP = bool(training.get("use_amp", True))
        self.GRAD_ACCUM_STEPS = max(1, int(training.get("grad_accum_steps", 1)))
        self.LR_SCHEDULER = training.get("lr_scheduler", "cosine")
        self.TRADES_ENABLED           = bool(training.get("trades_enabled", False))
        self.TRADES_EPSILON           = float(training.get("trades_epsilon", 0.05))
        self.TRADES_LAMBDA            = float(training.get("trades_lambda", 6.0))
        self.TRADES_STEPS             = int(training.get("trades_steps", 10))

        self.ACTIVITY_REG_ENABLED     = bool(training.get("activity_reg_enabled", False))
        self.ACTIVITY_REG_MIN_RATE    = float(training.get("activity_reg_min_rate", 0.01))
        self.ACTIVITY_REG_MAX_RATE    = float(training.get("activity_reg_max_rate", 0.50))
        self.ACTIVITY_REG_LAMBDA_LOW  = float(training.get("activity_reg_lambda_low", 0.1))
        self.ACTIVITY_REG_LAMBDA_HIGH = float(training.get("activity_reg_lambda_high", 0.1))

        self.STDP_ENABLED = bool(training.get("stdp_enabled", False))
        self.STDP_TAU     = float(training.get("stdp_tau", 20.0))
        self.STDP_A_PLUS  = float(training.get("stdp_a_plus", 0.01))
        self.STDP_A_MINUS = float(training.get("stdp_a_minus", 0.01))

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

        # Generated network structure
        self.network_structure = self.generate_network_structure()

    def load_yaml_config(self, yaml_path):
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
        print("=" * 60)
        print("SNN Configuration")
        print("=" * 60)

        print(f"Network architecture : {self.network_structure}")
        print(f"Epochs               : {self.EPOCHS}")
        print(f"Device               : {self.DEVICE}")
        print(f"Kernel               : {self.KERNEL}")
        print(f"Threshold            : {self.THRESHOLD}")

        print("=" * 60)

if __name__ == "__main__":
    cfg = Settings()
    cfg.display()