"""
Compiler entry point.

compile_model(model, cfg) returns:
  - CompiledModel when cfg.COMPILER_ENABLED is True
  - the original model unchanged when False

CompiledModel routes forward() through the compiler runtime, which
manages the timestep loop and dispatches each layer via the execution plan.
Weight parameters and optimizer remain on the original model.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from skeleton import Settings
from compiler.src.lowering  import lower_to_ir
from compiler.src.scheduler import schedule
from compiler.src.planner   import build_plan, ExecutionPlan
from compiler.src.runtime   import execute

logger = logging.getLogger(__name__)


class CompiledModel(nn.Module):
    """
    Wraps a PyTorch SNN model and routes forward passes through the compiler
    runtime. The runtime calls the original PyTorch layers (stored in IR node
    attrs) so weight updates and optimizer state remain fully intact.
    """

    def __init__(self, model: nn.Module, plan: ExecutionPlan, cfg: Settings):
        super().__init__()
        self.model = model
        self.plan  = plan
        self.cfg   = cfg

        self.net       = getattr(model, "net",       model)
        self.optimizer = getattr(model, "optimizer", None)
        self.loss_fn   = getattr(model, "loss_fn",   None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(self.net)
        except Exception:
            pass

        try:
            from learning.frameworks.activity_reg import clear_hidden_spikes
            clear_hidden_spikes(self.model)
        except Exception:
            pass

        return execute(self.plan, x)

    def reset_state(self):
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def compile_model(model: nn.Module, cfg: Settings) -> nn.Module:
    """
    Lower → schedule → plan the model.
    Returns CompiledModel if enabled, original model otherwise.
    """
    if not cfg.COMPILER_ENABLED:
        logger.info("[COMPILER] Disabled — using standard PyTorch execution")
        return model

    logger.info(
        "[COMPILER] Compiling — backend=%s, fuse_timesteps=%s, log_ir=%s",
        cfg.COMPILER_BACKEND, cfg.COMPILER_FUSE_STEPS, cfg.COMPILER_LOG_IR,
    )

    graph = lower_to_ir(model, cfg)
    graph = schedule(graph, cfg)
    plan  = build_plan(graph)

    if cfg.COMPILER_LOG_IR:
        logger.info("[COMPILER] Execution plan:\n%s", plan.summary())

    logger.info("[COMPILER] Compilation complete")
    return CompiledModel(model, plan, cfg)
