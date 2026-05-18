# =============================================================
# model.py — Parameterized Norse SNN
# =============================================================
# Builds an SNN based entirely on config.py settings.
# Supports three neuron types, variable depth, dropout,
# configurable surrogate gradients, and three decoding strategies.
#
# Input shape  : (batch, T, 2, 34, 34)  — N-MNIST batched frames
# Output shape : (batch, 10)            — class scores after decoding

import torch
import torch.nn as nn
import norse.torch as norse
from norse.torch import (
    LIFCell, LIFParameters,
    LIFAdExCell, LIFAdExParameters,
    LIFRecurrentCell,
)

# N-MNIST sensor: 2 polarity channels × 34 × 34 pixels
INPUT_SIZE = 2 * 34 * 34   # = 2312


# ── Neuron parameter builders ──────────────────────────────────
def _lif_params(config: dict) -> LIFParameters:
    """
    Converts config values into Norse LIFParameters.

    tau_mem_inv = 1 / tau_mem
      Norse works with the INVERSE of tau because it multiplies
      instead of divides in the update equation (faster computation).
      e.g. tau_mem=0.02 → tau_mem_inv=50.0

    v_th    = threshold voltage required to fire a spike.
    method  = which surrogate gradient to use during backprop.
    """
    return LIFParameters(
        tau_mem_inv=torch.as_tensor(1.0 / config["tau_mem"]),
        v_th=torch.as_tensor(config["threshold"]),
        method=config["surrogate"],
    )


def _lifadex_params(config: dict) -> LIFAdExParameters:
    """
    LIFAdEx = Adaptive Exponential LIF.
    Same tau_mem and threshold as LIF, but after each spike the
    threshold temporarily rises — making the neuron harder to fire
    again immediately. Models neural fatigue / spike-frequency adaptation.
    Adaptation parameters kept at Norse defaults.
    """
    return LIFAdExParameters(
        tau_mem_inv=torch.as_tensor(1.0 / config["tau_mem"]),
        v_th=torch.as_tensor(config["threshold"]),
        method=config["surrogate"],
    )


# ── SNN Model ─────────────────────────────────────────────────
class SNN(nn.Module):
    """
    Parameterized SNN that reads all settings from config.

    Architecture (LIF / LIFAdEx):
        flatten input
            ↓
        Linear → LIFCell / LIFAdExCell    × num_layers
            ↓
        Linear → LIFCell                  (output layer)
            ↓
        decode spikes → class scores

    Architecture (LIFRecurrent):
        flatten input
            ↓
        LIFRecurrentCell                  × num_layers
        (linear projection is built-in, neurons connect back to each other)
            ↓
        Linear → LIFCell                  (output layer)
            ↓
        decode spikes → class scores
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config      = config
        self.T           = config["timesteps"]
        self.neuron_type = config["neuron_type"]
        self.decoding    = config["decoding"]
        hidden           = config["hidden_size"]
        out              = config["output_size"]
        n_layers         = config["num_layers"]
        dropout_p        = config["dropout_p"]

        # ── Build hidden layers ───────────────────────────────
        self.fc_layers  = nn.ModuleList()   # linear projections
        self.lif_cells  = nn.ModuleList()   # spiking neuron layers
        self.dropouts   = nn.ModuleList()   # optional dropout

        in_size = INPUT_SIZE

        for _ in range(n_layers):
            if self.neuron_type == "LIF":
                self.fc_layers.append(nn.Linear(in_size, hidden))
                self.lif_cells.append(LIFCell(p=_lif_params(config)))

            elif self.neuron_type == "LIFAdEx":
                # Adaptive Exponential LIF — threshold rises after each spike
                # neuron becomes harder to fire again immediately (adaptation)
                self.fc_layers.append(nn.Linear(in_size, hidden))
                self.lif_cells.append(LIFAdExCell(p=_lifadex_params(config)))

            elif self.neuron_type == "LIFRecurrent":
                # recurrent cell has its own internal linear weights
                # AND recurrent weights (neurons connect back to themselves)
                # so no separate nn.Linear is needed
                self.lif_cells.append(
                    LIFRecurrentCell(
                        input_size=in_size,
                        hidden_size=hidden,
                        p=_lif_params(config),
                    )
                )

            self.dropouts.append(
                nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
            )
            in_size = hidden

        # ── Output layer (always plain LIF) ───────────────────
        self.fc_out  = nn.Linear(hidden, out)
        self.lif_out = LIFCell(p=_lif_params(config))

    def forward(self, x: torch.Tensor):
        """
        x: (batch, T, 2, 34, 34)

        Step by step:
          1. Permute to (T, batch, 2, 34, 34)
          2. Flatten spatial dims → (T, batch, 2312)
          3. Loop over T timesteps:
               - pass through each hidden layer (carry state forward)
               - pass through output layer (carry state forward)
               - collect output spikes at each step
          4. Decode collected spikes → (batch, 10) class scores
        """
        # (batch, T, 2, 34, 34) → (T, batch, 2, 34, 34)
        x = x.permute(1, 0, 2, 3, 4)
        T_actual, batch_size = x.shape[0], x.shape[1]

        # flatten: (T, batch, 2, 34, 34) → (T, batch, 2312)
        x = x.reshape(T_actual, batch_size, -1)

        # initialise all neuron states to None
        # Norse creates the correct state tensor on the first call
        hidden_states = [None] * len(self.lif_cells)
        out_state     = None

        output_spikes = []

        for t in range(T_actual):
            z = x[t]   # (batch, 2312) — input spikes at this timestep

            # ── Hidden layers ──────────────────────────────────
            for i, lif_cell in enumerate(self.lif_cells):
                if self.neuron_type in ("LIF", "LIFAdEx"):
                    z = self.fc_layers[i](z)
                    z, hidden_states[i] = lif_cell(z, hidden_states[i])
                elif self.neuron_type == "LIFRecurrent":
                    z, hidden_states[i] = lif_cell(z, hidden_states[i])

                z = self.dropouts[i](z)

            # ── Output layer ───────────────────────────────────
            z = self.fc_out(z)
            z, out_state = self.lif_out(z, out_state)

            output_spikes.append(z)

        # (T, batch, 10)
        spikes = torch.stack(output_spikes, dim=0)
        return self._decode(spikes)

    def _decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Convert output spike trains into class scores.

        spikes: (T, batch, 10)
        returns: (batch, 10)

        rate:        sum spikes over T — neuron that fired most wins
        max:         did neuron fire at all? (0 or 1 per neuron per sample)
        first_spike: neuron that fired earliest = most confident
                     neurons that never fire score 0
        """
        if self.decoding == "rate":
            return spikes.sum(dim=0)

        elif self.decoding == "max":
            return spikes.max(dim=0).values

        elif self.decoding == "first_spike":
            T = spikes.shape[0]
            fired = spikes > 0                              # (T, batch, 10)
            first_t = torch.where(
                fired.any(dim=0),
                fired.float().argmax(dim=0).float(),
                torch.full_like(fired.float().argmax(dim=0).float(), float(T))
            )
            return (T - first_t).float()

        else:
            raise ValueError(f"Unknown decoding: {self.decoding}")


def build_model(config: dict) -> SNN:
    """Entry point — call this from train.py."""
    model = SNN(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model        : SNN ({config['neuron_type']})")
    print(f"Hidden layers: {config['num_layers']} × {config['hidden_size']} neurons")
    print(f"Surrogate    : {config['surrogate']}")
    print(f"Decoding     : {config['decoding']}")
    print(f"Parameters   : {n_params:,}")
    return model
