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



# we need to access the  configuzation set within the env file to determine how to set up the SNN model and training loop. 
