import math
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # Neural network architecture
        self.INPUT_SIZE = int(os.getenv("INPUT_SIZE", 6))
        self.HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 10))
        self.OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", 6))
        self.THRESHOLD = float(os.getenv("THRESHOLD", 0.8))
        self.LEAK = float(os.getenv("LEAK", 1.5))
        self.OVERRIDE = os.getenv("OVERRIDE", "False").lower() == "true"

        # Training parameters
        self.LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "MSE")
        self.OPTIMIZER = os.getenv("OPTIMIZER", "Adam")
        self.EPOCHS = int(os.getenv("EPOCHS", 100))
        self.TIMESTEPS = int(os.getenv("TIMESTEPS", 50))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
        self.BETA = float(os.getenv("BETA", 0.5))
        self.NAP_TIMES = int(os.getenv("NAP_TIMES", 5))
        self.NETWORK_STRUCT = os.getenv("NETWORK_STRUCT", None)  # S=stable, A=ascending, D=descending

        self.network_structure = self.generate_network_structure()

    def generate_network_structure(self):
        """
        Generates a list representing the neuron count per layer:
        [input_layer, hidden1, hidden2, ..., output_layer]
        respecting constraints:
        - max next_layer = 3 * previous_layer
        - minimum next_layer >= floor(previous_layer / 2) but not below 2
        - output_layer >= 3, unconstrained otherwise
        - stable, ascending, descending based on NETWORK_STRUCT
        """
        layers = [self.INPUT_SIZE]

        if not self.OVERRIDE:
            layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS
            layers.append(max(self.OUTPUT_SIZE, 3))
            return layers
        
        prev = self.INPUT_SIZE
        if self.OVERRIDE==True:
            if self.NETWORK_STRUCT == "S" or self.NETWORK_STRUCT is None:
                layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS
            elif self.NETWORK_STRUCT == "A":  # ascending
                next_layer = self.HIDDEN_SIZE
                layers += [min(prev * 3, prev + 3) for _ in range(self.HIDDEN_LAYERS)]
            else:  # descending
                next_layer = self.HIDDEN_SIZE
                layers += [max(math.ceil(prev / 2), 2) for _ in range(self.HIDDEN_LAYERS)]

            layers.append(next_layer)
            prev = next_layer

        # Append output layer (must be at least 3, unconstrained)
        layers.append(max(self.OUTPUT_SIZE, 3))

        return layers


if __name__ == "__main__":
    cfg = Settings()
    print("Network structure:", cfg.network_structure)