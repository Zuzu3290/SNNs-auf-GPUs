import yaml

class Settings:
    def __init__(self, yaml_path="SNN_module.yaml"):
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
        self.DEVICE = architecture.get("device", "cuda")
        self.KERNEL = architecture.get("kernel", "OFF")
        self.SIMULATOR = architecture.get("simulator", "OFF")

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