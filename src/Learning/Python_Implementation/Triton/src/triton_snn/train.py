from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from triton_snn.config import SNNConfig
from triton_snn.models.snn import TritonSNNClassifier


def make_synthetic_batch(cfg: SNNConfig, device: torch.device):
    x = torch.randn(cfg.batch_size, cfg.input_dim, device=device, dtype=torch.float32)
    y = torch.randint(0, cfg.output_dim, (cfg.batch_size,), device=device)
    return x, y


def run_training_demo(epochs: int = 5):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Triton execution.")

    device = torch.device("cuda")
    cfg = SNNConfig()

    model = TritonSNNClassifier(cfg).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()

    history = []
    for epoch in range(1, epochs + 1):
        x, y = make_synthetic_batch(cfg, device=device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "accuracy": float(acc),
        }
        history.append(row)
        print(f"epoch={epoch:02d} loss={row['loss']:.4f} acc={row['accuracy']:.4f}")

    return history
