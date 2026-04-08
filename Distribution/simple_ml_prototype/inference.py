# inference.py
# Loads the saved model weights and predicts the Iris class for a given input.
# Run this AFTER training: python inference.py
# Or pass features directly (single prediction): python inference.py 5.1 3.5 1.4 0.2

import sys
import torch

from model_definition import IrisClassifier  # Same architecture used in training

# ─────────────────────────────────────────────
# 1. Greeting & introduction
# ─────────────────────────────────────────────
print("=" * 50)
print("   Iris Flower Classifier")
print("=" * 50)
print()
print("Welcome! This tool uses a trained neural network")
print("to identify the species of an Iris flower.")
print()
print("You will need to provide 4 measurements (in cm):")
print("  1. Sepal length  — the outer leaf of the flower")
print("  2. Sepal width   — how wide that outer leaf is")
print("  3. Petal length  — the inner colourful leaf")
print("  4. Petal width   — how wide that inner leaf is")
print()
print("The model will predict one of these 3 species:")
print("  - Setosa      (small petals, easy to identify)")
print("  - Versicolor  (medium sized)")
print("  - Virginica   (largest petals)")
print()
print("-" * 50)

# ─────────────────────────────────────────────
# 2. Load the saved model + metadata (done ONCE before the loop)
# ─────────────────────────────────────────────
WEIGHTS_PATH = "model/trained_model.pth"
print(f"\nLoading trained model from '{WEIGHTS_PATH}'...")

# Load the checkpoint dictionary saved during training
checkpoint = torch.load(WEIGHTS_PATH, weights_only=True)

# Recreate the model and load the trained weights into it
model = IrisClassifier()
model.load_state_dict(checkpoint["model_state_dict"])

# IMPORTANT: switch to evaluation mode — disables dropout, batch norm updates, etc.
model.eval()

# Retrieve the scaler parameters saved during training
scaler_mean  = checkpoint["scaler_mean"]   # mean of each feature
scaler_scale = checkpoint["scaler_scale"]  # std dev of each feature
class_names  = checkpoint["class_names"]   # ['setosa', 'versicolor', 'virginica']

print("Model ready!\n")
print("-" * 50)

# ─────────────────────────────────────────────
# Helper function: run one prediction and print the result
# ─────────────────────────────────────────────
def predict(features):
    """Takes a list of 4 floats, runs the model, prints the result."""

    # Scale the input the same way training data was scaled
    # Formula: scaled = (value - mean) / std
    scaled = [(features[i] - scaler_mean[i]) / scaler_scale[i] for i in range(4)]
    input_tensor = torch.FloatTensor(scaled).unsqueeze(0)  # shape: (1, 4)

    # Run inference — NO weight updates happen here
    with torch.no_grad():  # Disable gradient computation — we are only predicting
        output       = model(input_tensor)                        # Raw scores for each class
        probabilities = torch.softmax(output, dim=1)              # Convert to probabilities (0–100%)
        predicted_idx = torch.argmax(output, dim=1).item()        # Index of the highest score

    predicted_class = class_names[predicted_idx]
    confidence      = probabilities[0][predicted_idx].item() * 100

    # Print the result
    print()
    print("=" * 50)
    print("   Prediction Result")
    print("=" * 50)
    print(f"\n  Predicted species : {predicted_class.upper()}")
    print(f"  Confidence        : {confidence:.1f}%")

    # Explain confidence level in plain English
    print()
    if confidence >= 90:
        print("  The model is very confident about this prediction.")
    elif confidence >= 70:
        print("  The model is fairly confident, but there is some")
        print("  chance it could be a neighbouring species.")
    else:
        print("  The model is uncertain — your flower's measurements")
        print("  may fall between two species.")

    # Show all class probabilities with a simple text bar chart
    print()
    print("  Breakdown by species:")
    print("  " + "-" * 30)
    for i, name in enumerate(class_names):
        bar = "#" * int(probabilities[0][i].item() * 20)
        print(f"  {name:<12} {probabilities[0][i].item() * 100:5.1f}%  {bar}")
    print()


# ─────────────────────────────────────────────
# 3. Single-shot mode: features passed as command-line arguments
#    Example: python inference.py 5.1 3.5 1.4 0.2
# ─────────────────────────────────────────────
if len(sys.argv) == 5:
    try:
        features = [float(sys.argv[i]) for i in range(1, 5)]
        print(f"Using values from command line: {features}")
        predict(features)
    except ValueError:
        print("Error: all 4 arguments must be numbers.")
        print("Usage: python inference.py <sepal_len> <sepal_wid> <petal_len> <petal_wid>")
        sys.exit(1)
    sys.exit(0)  # Exit after single prediction when args are given

# ─────────────────────────────────────────────
# 4. Interactive loop mode: keep asking until user types 'q'
# ─────────────────────────────────────────────
print("\nType 'q' at any prompt to quit.\n")

while True:
    print("Enter measurements for a new flower:")
    print("(or type 'q' and press Enter to quit)\n")

    try:
        # Check each input for 'q' before converting to float
        raw = input("  Sepal length (cm) : ").strip()
        if raw.lower() == 'q':
            break
        sepal_length = float(raw)

        raw = input("  Sepal width  (cm) : ").strip()
        if raw.lower() == 'q':
            break
        sepal_width = float(raw)

        raw = input("  Petal length (cm) : ").strip()
        if raw.lower() == 'q':
            break
        petal_length = float(raw)

        raw = input("  Petal width  (cm) : ").strip()
        if raw.lower() == 'q':
            break
        petal_width = float(raw)

    except ValueError:
        print("\n  Please enter a valid number (e.g. 5.1). Try again.\n")
        print("-" * 50)
        continue  # Go back to the top of the loop

    # Run prediction with the collected features
    predict([sepal_length, sepal_width, petal_length, petal_width])
    print("-" * 50)

# ─────────────────────────────────────────────
# 5. Goodbye message when user quits
# ─────────────────────────────────────────────
print()
print("=" * 50)
print("  Thank you for using the Iris Classifier!")
print("  Goodbye!")
print("=" * 50)
print()
