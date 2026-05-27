"""
Runtime: executes an ExecutionPlan against real tensors.

For each timestep the runtime walks the plan steps:
  - AtomicStep  → calls node.attrs["module"](x) using the original PyTorch layer
  - FusedStep   → calls the stored spikingjelly/snntorch neuron for that group,
                  which manages membrane state internally across timestep calls.
                  Falls back to a pure-PyTorch LIF step when no module is stored.

The runtime owns the timestep loop so the compiler controls execution order,
which is the hook point for future custom-kernel dispatch.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from compiler.src.ir import OpType
from compiler.src.planner import AtomicStep, ExecutionPlan, FusedStep

logger = logging.getLogger(__name__)


def lif_step(
    x:         torch.Tensor,
    mem:       torch.Tensor,
    beta:      float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-step soft-reset LIF: integrate → threshold → spike → reset."""
    mem_new = beta * mem + x
    spikes  = (mem_new >= threshold).float()
    mem_new = mem_new - spikes * threshold
    return spikes, mem_new


def execute(plan: ExecutionPlan, x: torch.Tensor) -> torch.Tensor:
    """
    Run the execution plan over time-first input x [T, B, ...].
    Returns accumulated spike tensor [B, C].
    """
    T          = x.size(0)
    mem_state: Dict[int, torch.Tensor] = {}
    spike_acc: Optional[torch.Tensor]  = None

    for t in range(T):
        current = x[t]

        for step in plan.steps:

            if isinstance(step, AtomicStep):
                node = step.node

                if node.op in (OpType.INPUT, OpType.OUTPUT, OpType.AGGREGATE):
                    continue

                module = node.attrs.get("module")
                if module is not None:
                    current = module(current)

            elif isinstance(step, FusedStep):
                neuron = step.nodes[0].attrs.get("module")

                if neuron is not None:
                    current = neuron(current)
                else:
                    gid = step.group_id
                    if gid not in mem_state:
                        mem_state[gid] = torch.zeros_like(current)
                    beta      = float(step.attrs.get("beta",      0.95))
                    threshold = float(step.attrs.get("threshold", 1.0))
                    current, mem_state[gid] = lif_step(
                        current, mem_state[gid], beta, threshold
                    )

        spike_acc = current if spike_acc is None else spike_acc + current

    return spike_acc
