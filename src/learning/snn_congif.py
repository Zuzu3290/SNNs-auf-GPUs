from dataclasses import dataclass
from typing import Literal

@dataclass
class HardwareConfig:
    n_neurons: int
    timesteps: int
    fan_in: int
    energy_target: Literal["low", "medium", "high"] = "medium"
    noise_robustness: Literal["low", "medium", "high"] = "medium"
    multi_processing: Literal["single", "streams", "multi_gpu"] = "streams"