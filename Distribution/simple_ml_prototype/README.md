# Simple ML Prototype — Iris Classifier

A beginner-friendly end-to-end ML project using PyTorch and the Iris dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Phase 1 Usage

### Step 1 — Train the model
```bash
python train.py
```
This trains the network and saves weights to `model/trained_model.pth`.

### Step 2 — Run inference
```bash
# Pass 4 features directly:
python inference.py 5.1 3.5 1.4 0.2

# Or run interactively:
python inference.py
```

The 4 numbers are: `sepal_length  sepal_width  petal_length  petal_width` (cm)

## File Structure

```
simple_ml_prototype/
├── data/                   # Dataset files (Iris is loaded from sklearn)
├── model/
│   └── trained_model.pth   # Saved after training
├── model_definition.py     # Neural network architecture
├── train.py                # Training script
├── inference.py            # Inference / prediction script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
