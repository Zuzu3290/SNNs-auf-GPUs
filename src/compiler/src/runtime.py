"""
Runtime: executes an ExecutionPlan against real tensors.

Dispatch priority for each plan step
-------------------------------------
AtomicStep  → calls node.attrs["module"](x) using the original PyTorch layer,
              so weight tensors and optimizer state stay intact.

FusedStep   → tries the CUDA kernel first (lif_fused_step), which runs
              membrane_update + threshold + spike_gen + reset in a single kernel
              launch and owns the membrane state for that group.
              Falls back to the original SNNTorch/SpikingJelly neuron module
              (state managed internally by the framework) when the tensor is on
              CPU or when the CUDA kernel is unavailable.
              Final fallback: pure-PyTorch soft-reset LIF with explicit mem state.

The runtime owns the timestep loop, which is the hook point for future
custom kernel dispatch (e.g. fusing the entire T-step loop into one launch).
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
    """Pure-PyTorch soft-reset LIF — used as final fallback."""
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
                beta      = float(step.attrs.get("beta",      0.95))
                threshold = float(step.attrs.get("threshold", 1.0))
                gid       = step.group_id
                used_cuda = False

                # ── Priority 1: fused CUDA kernel ─────────────────────────────
                if current.is_cuda:
                    try:
                        from compiler.cuda_ops import _load_ops, lif_fused_step
                        if _load_ops() is not None:
                            if gid not in mem_state:
                                mem_state[gid] = torch.zeros_like(current)
                            current, mem_state[gid] = lif_fused_step(
                                current, mem_state[gid], beta, threshold
                            )
                            used_cuda = True
                    except Exception as exc:
                        logger.debug("[RUNTIME] CUDA kernel step failed (%s), using fallback", exc)

                if not used_cuda:
                    # ── Priority 2: framework neuron module (manages own state) ──
                    neuron = step.nodes[0].attrs.get("module")
                    if neuron is not None:
                        out     = neuron(current)
                        current = out[0] if isinstance(out, tuple) else out
                    else:
                        # ── Priority 3: pure-PyTorch LIF ──────────────────────
                        if gid not in mem_state:
                            mem_state[gid] = torch.zeros_like(current)
                        current, mem_state[gid] = lif_step(
                            current, mem_state[gid], beta, threshold
                        )

        spike_acc = current if spike_acc is None else spike_acc + current

    return spike_acc
