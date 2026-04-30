import math
import yaml


class Settings:
    def __init__(self, yaml_path="SNN_module.yaml"):
        self.yaml_path = yaml_path
        self.config = self.load_yaml_config(yaml_path)

        architecture = self.config.get("architecture", {})
        training = self.config.get("training", {})
        dataset = self.config.get("dataset", {})
        input_cfg = self.config.get("input", {})
        neuromorphic = self.config.get("neuromorphic", {})
        model = self.config.get("model", {})
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

        # Training parameters
        self.LOSS_FUNCTION = training.get("loss_function", "CrossEntropy")
        self.OPTIMIZER = training.get("optimizer", "Adam")
        self.EPOCHS = int(training.get("epochs", 10))
        self.TIMESTEPS = int(training.get("timesteps", 25))
        self.BATCH_SIZE = int(training.get("batch_size", 128))
        self.BETA = float(training.get("beta", 0.95))
        self.NAP_TIMES = int(training.get("nap_times", 1))
        self.LEARNING_RATE = float(training.get("learning_rate", 0.001))
        self.WEIGHT_DECAY = float(training.get("weight_decay", 0.0001))
        self.NUM_CLASSES = int(training.get("num_classes", self.OUTPUT_SIZE))

        # Dataset control
        self.DATASET_TYPE = dataset.get("dataset_type", "digital")
        self.DATASET_NAME = dataset.get("dataset_name", "MNIST")
        self.DATA_PATH = dataset.get("data_path", "./data")

        # Input control
        self.INPUT_MODE = input_cfg.get("input_mode", "2D")
        self.IMAGE_CHANNELS = int(input_cfg.get("image_channels", 1))
        self.IMAGE_HEIGHT = int(input_cfg.get("image_height", 28))
        self.IMAGE_WIDTH = int(input_cfg.get("image_width", 28))

        # Neuromorphic dataset control
        self.USE_TONIC = bool(neuromorphic.get("use_tonic", False))
        self.EVENT_REPRESENTATION = neuromorphic.get("event_representation", "frame")
        self.SENSOR_SIZE = neuromorphic.get("sensor_size", None)

        # Model control
        self.MODEL_TYPE = model.get("model_type", "CSNN")
        self.SAVE_MODEL = bool(model.get("save_model", True))

        # Output control
        self.OUTPUT_DIR = output.get("output_dir", "./outputs")
        self.PLOT_DIR = output.get("plot_dir", "./outputs/plots")
        self.MODEL_DIR = output.get("model_dir", "./outputs/models")
        self.REPORT_DIR = output.get("report_dir", "./outputs/reports")

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

        layers = [self.INPUT_SIZE]
        prev = self.INPUT_SIZE

        if self.OVERRIDE:
            if self.NETWORK_STRUCT == "S" or self.NETWORK_STRUCT is None:
                layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

            elif self.NETWORK_STRUCT == "A":
                for _ in range(self.HIDDEN_LAYERS):
                    layers.append(prev)
                    next_layer = prev + 4
                    prev = next_layer

            elif self.NETWORK_STRUCT == "D":
                temp_layers = []

                last = max(2, math.ceil(self.INPUT_SIZE / (2 ** self.HIDDEN_LAYERS)))
                temp_layers.append(last)

                for _ in range(self.HIDDEN_LAYERS - 1):
                    next_layer = temp_layers[-1] * 2
                    temp_layers.append(next_layer)

                temp_layers.reverse()
                layers += temp_layers

            else:
                raise ValueError(
                    "Invalid NETWORK_STRUCT. Use 'S', 'A', or 'D'."
                )

        else:
            layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

        layers.append(max(self.OUTPUT_SIZE, 3))

        return layers

    def display(self):
        print("=" * 60)
        print("SNN Configuration")
        print("=" * 60)

        print(f"Dataset type      : {self.DATASET_TYPE}")
        print(f"Dataset name      : {self.DATASET_NAME}")
        print(f"Input mode        : {self.INPUT_MODE}")
        print(f"Image shape       : {self.IMAGE_CHANNELS}x{self.IMAGE_HEIGHT}x{self.IMAGE_WIDTH}")
        print(f"Model type        : {self.MODEL_TYPE}")
        print(f"Network structure : {self.network_structure}")
        print(f"Epochs            : {self.EPOCHS}")
        print(f"Batch size        : {self.BATCH_SIZE}")
        print(f"Timesteps         : {self.TIMESTEPS}")
        print(f"Learning rate     : {self.LEARNING_RATE}")
        print(f"Optimizer         : {self.OPTIMIZER}")
        print(f"Loss function     : {self.LOSS_FUNCTION}")
        print("=" * 60)


if __name__ == "__main__":
    cfg = Settings()
    cfg.display()