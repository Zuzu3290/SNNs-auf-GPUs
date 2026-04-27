import math
import os
from dotenv import load_dotenv
load_dotenv("SNN_module.env")

class Settings:
    def __init__(self):
        # Neural network architecture
        self.INPUT_SIZE = int(os.getenv("INPUT_SIZE",10))
        self.HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE",16))
        self.HIDDEN_LAYERS = int(os.getenv("HIDDEN_LAYERS",3))
        self.OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE",5))
        self.THRESHOLD = float(os.getenv("THRESHOLD",0.5))
        self.LEAK = float(os.getenv("LEAK",1.0))
        self.OVERRIDE = os.getenv("OVER_RIDE", "false").lower() == "true"        
        self.NETWORK_STRUCT = os.getenv("NETWORK_STRUCT", None)  # S=stable, A=ascending, D=descending
        self.network_structure = self.generate_network_structure()

        # Training parameters
        self.LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "MSE")
        self.OPTIMIZER = os.getenv("OPTIMIZER", "Adam")
        self.EPOCHS = int(os.getenv("EPOCHS", 10))
        self.TIMESTEPS = int(os.getenv("TIMESTEPS", 25))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
        self.BETA = float(os.getenv("BETA", 0.5))
        self.NAP_TIMES = int(os.getenv("NAP_TIMES", 1))


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
        
        prev = self.INPUT_SIZE

        if self.OVERRIDE:
            if self.NETWORK_STRUCT == "S" or self.NETWORK_STRUCT is None:
                layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

            elif self.NETWORK_STRUCT == "A":
                for A in range(self.HIDDEN_LAYERS):
                    layers.append(prev)
                    next_layer = prev + 4  # controlled linear growth
                    prev = next_layer

            else:
                temp_layers = []

                last = max(2, math.ceil(self.INPUT_SIZE / (2 ** self.HIDDEN_LAYERS)))
                temp_layers.append(last)

                for A in range(self.HIDDEN_LAYERS - 1):
                    next_layer = temp_layers[-1] * 2
                    temp_layers.append(next_layer)

                temp_layers.reverse()
                layers += temp_layers

        else:
            layers += [self.HIDDEN_SIZE] * self.HIDDEN_LAYERS

        # Append output layer (must be at least 3, unconstrained)
        layers.append(max(self.OUTPUT_SIZE, 3))
        return layers


if __name__ == "__main__":
    cfg = Settings()
    print("Network structure:", cfg.network_structure)