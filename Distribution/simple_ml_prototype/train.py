# train.py
# Loads the Iris dataset, trains the neural network, and saves the weights.
# Run this script ONCE to produce the saved model file.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_definition import IrisClassifier  # Import our model architecture

# ─────────────────────────────────────────────
# 1. Load the Iris dataset
# ─────────────────────────────────────────────
print("Loading Iris dataset...")
iris = load_iris()

X = iris.data    # Features: (150, 4) — sepal/petal length & width
y = iris.target  # Labels:   (150,)   — 0=Setosa, 1=Versicolor, 2=Virginica

print(f"  Total samples: {len(X)}")
print(f"  Classes: {iris.target_names.tolist()}")

# ─────────────────────────────────────────────
# 2. Pre-process: scale features & split data
# ─────────────────────────────────────────────
# Standardize features so they have mean=0 and std=1 — helps training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into 80% training / 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# Convert numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)

# ─────────────────────────────────────────────
# 3. Create the model
# ─────────────────────────────────────────────
print("\nInitializing model...")
model = IrisClassifier()

# CrossEntropyLoss is standard for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer adjusts weights during training
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ─────────────────────────────────────────────
# 4. Train the model
# ─────────────────────────────────────────────
NUM_EPOCHS = 100
print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()  # Set model to training mode

    # Forward pass: compute predictions
    outputs = model(X_train)

    # Compute how wrong the predictions are
    loss = criterion(outputs, y_train)

    # Backward pass: compute gradients
    optimizer.zero_grad()  # Clear old gradients first
    loss.backward()        # Compute new gradients

    # Update weights based on gradients
    optimizer.step()

    # Every 10 epochs, print progress
    if epoch % 10 == 0:
        # Evaluate on test set (no gradient needed here)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            # Pick the class with the highest score
            predicted = torch.argmax(test_outputs, dim=1)
            accuracy = (predicted == y_test).float().mean().item() * 100

        print(f"  Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Loss: {loss.item():.4f}  "
              f"Test Accuracy: {accuracy:.1f}%")

# ─────────────────────────────────────────────
# 5. Save the trained weights
# ─────────────────────────────────────────────
SAVE_PATH = "model/trained_model.pth"

# Save model weights + scaler parameters so inference uses the same scaling
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_mean":      scaler.mean_.tolist(),
    "scaler_scale":     scaler.scale_.tolist(),
    "class_names":      iris.target_names.tolist(),
}, SAVE_PATH)

print(f"\nTraining complete. Model saved to '{SAVE_PATH}'")
