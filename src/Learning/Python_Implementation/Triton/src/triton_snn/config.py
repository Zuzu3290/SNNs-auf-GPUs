from dataclasses import dataclass

@dataclass
class SNNConfig:
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    time_steps: int = 16
    beta: float = 0.95
    threshold: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 32
