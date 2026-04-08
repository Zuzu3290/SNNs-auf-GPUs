# model_definition.py
# Defines the neural network architecture for classifying Iris flowers.
# This file is imported by both train.py and inference.py — no training happens here.

import torch
import torch.nn as nn


class IrisClassifier(nn.Module):
    """
    A simple 3-layer fully connected neural network for Iris classification.

    Input:  4 features (sepal length, sepal width, petal length, petal width)
    Output: 3 class scores (Setosa, Versicolor, Virginica)
    """

    def __init__(self):
        super(IrisClassifier, self).__init__()

        # Layer 1: 4 inputs → 16 hidden neurons
        self.layer1 = nn.Linear(4, 16)

        # Layer 2: 16 → 8 hidden neurons
        self.layer2 = nn.Linear(16, 8)

        # Output layer: 8 → 3 class scores
        self.output = nn.Linear(8, 3)

        # ReLU activation adds non-linearity between layers
        self.relu = nn.ReLU()

    def forward(self, x):
        """Defines how data flows through the network."""
        x = self.relu(self.layer1(x))   # Pass through layer 1 + activation
        x = self.relu(self.layer2(x))   # Pass through layer 2 + activation
        x = self.output(x)              # Final output (raw scores / logits)
        return x
