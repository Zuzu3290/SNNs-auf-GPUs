"""The LIF neuron written out as plain arithmetic, independent of any framework, for studying the model itself.

    tau * dv/dt = -(v - v_leak) + R * I(t),  spike when v >= v_th, then hard reset

Discretised, v[k+1] = v_leak + beta*(v[k] - v_leak) + gain*I[k], with beta = exp(-dt/tau) (exact) or
1 - dt/tau (Euler). Forward pass only; not imported by the pipeline.
"""
from __future__ import annotations
import math
import torch


class ReferenceLIF:
    def __init__(self, beta: float = 0.9, gain: float = 1.0, v_th: float = 1.0, v_reset: float = 0.0, v_leak: float = 0.0):
        self.beta = beta        # share of the voltage kept per step (0.9 = keeps 90%, leaks 10%)
        self.gain = gain        # how strongly the input current raises the voltage
        self.v_th = v_th        # voltage at which the neuron fires
        self.v_reset = v_reset  # voltage right after a spike
        self.v_leak = v_leak    # resting voltage the neuron leaks back towards

    @staticmethod
    def beta_from_tau(tau: float, dt: float = 1.0, scheme: str = "exact") -> float:
        """Turns a time constant tau (how slowly the voltage leaks away) into beta, for steps of length dt.

        using exp(-dt/tau); "euler" uses the simpler 1 - dt/tau, which is close when dt is much smaller than tau.
        """
        return math.exp(-dt / tau) if scheme == "exact" else 1.0 - dt / tau

    def tau(self, dt: float = 1.0, scheme: str = "exact") -> float:
        """The reverse: the time constant tau that this neuron's beta stands for, with steps of length dt."""
        return -dt / math.log(self.beta) if scheme == "exact" else dt / (1.0 - self.beta)

    def step(self, v: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One time step: leak towards rest, add the input, fire where the voltage reaches v_th, and reset those neurons."""
        v = self.v_leak + self.beta * (v - self.v_leak) + self.gain * current
        spikes = (v >= self.v_th).to(v.dtype)
        v = torch.where(spikes > 0, torch.full_like(v, self.v_reset), v)
        return spikes, v

    def simulate(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs step() over a whole input: current [T, ...] in, (spikes [T, ...], voltage after each step [T, ...]) out."""
        v = torch.full_like(current[0], float(self.v_leak))
        spikes, trace = [], []
        for step in range(current.shape[0]):
            spike, v = self.step(v, current[step])
            spikes.append(spike)
            trace.append(v)
        return torch.stack(spikes), torch.stack(trace)

    def describe(self, dt: float = 1.0) -> dict:
        return {
            "neuron": "reference_lif",
            "beta": self.beta,
            "gain": self.gain,
            "v_th": self.v_th,
            "v_reset": self.v_reset,
            "v_leak": self.v_leak,
            "dt": dt,
            "tau_exact": self.tau(dt, "exact"),
            "tau_euler": self.tau(dt, "euler"),
            "reset": "hard, same step",
        }
