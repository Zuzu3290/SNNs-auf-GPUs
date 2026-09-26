"""The LIF neuron as arithmetic, owned by no framework: ground truth for equivalence_check and the spec the cuSNN kernel must reproduce.

    tau * dv/dt = -(v - v_leak) + R * I(t),  spike when v >= v_th, then hard reset

Held constant across a step, both the exact solution and forward Euler reduce to
v[k+1] = v_leak + beta*(v[k] - v_leak) + gain*I[k]; they differ only in beta,
exp(-dt/tau) against 1 - dt/tau. So beta is what is stored and tau is what is reported.
Exact is the default: closed-form, stable for every dt, and what the hardware does.
Forward only -- the backward pass is where the four frameworks legitimately differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

EXACT = "exact"
EULER = "euler"
SCHEMES = (EXACT, EULER)


def retention_from_tau(tau: float, dt: float = 1.0, scheme: str = EXACT) -> float:
    """Membrane time constant -> per-step retention factor beta."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if scheme == EXACT:
        return math.exp(-dt / tau)
    if scheme == EULER:
        return 1.0 - dt / tau
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def tau_from_retention(beta: float, dt: float = 1.0, scheme: str = EXACT) -> float:
    """Per-step retention factor beta -> membrane time constant, in units of dt."""
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must lie strictly between 0 and 1, got {beta}")
    if scheme == EXACT:
        return -dt / math.log(beta)
    if scheme == EULER:
        return dt / (1.0 - beta)
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def decay_to_params(decay: float, dt: float = 0.001) -> dict:
    """One per-step retention factor written in each framework's own parameterisation."""
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must lie strictly between 0 and 1, got {decay}")
    return {
        "snntorch": {"beta": decay},
        "spikingjelly": {"tau": tau_from_retention(decay, 1.0, EULER)},
        # Norse locks input gain to (1 - decay), so input_scale has to move with it or the
        # neuron receives a different amount of current at every point of a decay sweep.
        "norse": {"tau_mem_inv": (1.0 - decay) / dt, "input_scale": 1.0 / (1.0 - decay)},
        "sinabs": {"tau_mem": tau_from_retention(decay, 1.0, EXACT)},
    }


@dataclass
class ReferenceLIF:
    """beta is the per-step retention; gain is kept independent of it, as the project's configs do."""

    beta: float = 0.9
    gain: float = 1.0
    v_th: float = 1.0
    v_reset: float = 0.0
    v_leak: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"beta must lie in [0, 1), got {self.beta}")

    def tau(self, dt: float = 1.0, scheme: str = EXACT) -> float:
        """The time constant this retention corresponds to under the given scheme."""
        return tau_from_retention(self.beta, dt, scheme)

    def step(self, v: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One timestep: decay, integrate, fire, hard reset in the same step. Returns (spikes, membrane)."""
        v = self.v_leak + self.beta * (v - self.v_leak) + self.gain * current
        spikes = (v >= self.v_th).to(v.dtype)
        v = torch.where(spikes > 0, torch.full_like(v, self.v_reset), v)
        return spikes, v

    def simulate(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """current [T, ...] -> (spikes [T, ...], membrane after each step [T, ...])."""
        v = torch.full_like(current[0], float(self.v_leak))
        spikes, trace = [], []
        for step in range(current.shape[0]):
            spike, v = self.step(v, current[step])
            spikes.append(spike)
            trace.append(v)
        return torch.stack(spikes), torch.stack(trace)

    def describe(self, dt: float = 1.0) -> dict:
        """Both readings of time, for the record."""
        return {
            "neuron": "reference_lif",
            "beta": self.beta,
            "gain": self.gain,
            "v_th": self.v_th,
            "v_reset": self.v_reset,
            "v_leak": self.v_leak,
            "dt": dt,
            "tau_exact": self.tau(dt, EXACT),
            "tau_euler": self.tau(dt, EULER),
            "reset": "hard, same step",
        }
