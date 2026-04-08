# Dockerfile
# Packages the trained Iris model + inference script into a portable Docker image.
# After building, anyone can run predictions without installing Python or PyTorch.

# ─────────────────────────────────────────────
# 1. Start from an official Python base image
#    'slim' is a lightweight version — smaller download, faster build
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────
# 2. Set the working directory inside the container
#    All future commands will run from this folder
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# 3. Copy requirements and install dependencies
#    We copy requirements.txt first (before the rest of the code)
#    so Docker can cache this layer — speeds up rebuilds
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# 4. Copy the model architecture and inference script
# ─────────────────────────────────────────────
COPY model_definition.py .
COPY inference.py .

# ─────────────────────────────────────────────
# 5. Copy the saved model weights into the container
#    This is the file produced by train.py
# ─────────────────────────────────────────────
COPY model/trained_model.pth model/trained_model.pth

# ─────────────────────────────────────────────
# 6. Default command — runs inference interactively
#    User can also pass 4 numbers directly:
#    docker run iris-model 5.1 3.5 1.4 0.2
# ─────────────────────────────────────────────
ENTRYPOINT ["python", "inference.py"]
