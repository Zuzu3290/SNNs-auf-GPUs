"""
Compiler entry point.

compile_model(model, cfg) → CompiledModel when cfg.COMPILER_ENABLED is True,
                             original model unchanged when False.

Pipeline
--------
  lower_to_ir  →  schedule  →  build_plan  →  CompiledModel

CompiledModel.forward() routes through execute(plan, x) in runtime.py.
The runtime dispatches each FusedStep to the custom CUDA kernel (lif_fused_step)
which fuses membrane_update + threshold + spike_gen + reset into one kernel launch,
replacing four separate PyTorch operations with one.

CUDA kernel is compiled on first use via NVCC (torch.utils.cpp_extension.load).
Compilation is triggered proactively during compile_model() so it completes
before training starts rather than on the first batch.

Note on activity regularization and STDP
-----------------------------------------
The CUDA kernel path owns the membrane state and bypasses the SNNTorch neuron
modules. Forward hooks registered by activity_reg.py on those modules will
therefore not fire during the CUDA path. activity_reg and STDP penalties will
be zero when the compiler is enabled. Disable them in SNN_module.yaml when
using the compiler to avoid a silent no-op.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root → skeleton
sys.path.insert(0, str(Path(__file__).parent.parent))          # src/ → compiler, learning
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
    attrs) or the fused CUDA kernel for LIF groups, keeping weight tensors and
    optimizer state fully intact on the original model.
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
            import snntorch.utils as snn_utils
            snn_utils.reset(self.net)
        except Exception:
            pass

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
    Lower → schedule → plan, then return a CompiledModel that dispatches LIF
    layers to the fused CUDA kernel.
    """
    if not cfg.COMPILER_ENABLED:
        logger.info("[COMPILER] Disabled — using standard PyTorch execution")
        return model

    logger.info(
        "[COMPILER] Compiling — backend=%s  fuse_timesteps=%s  log_ir=%s",
        cfg.COMPILER_BACKEND, cfg.COMPILER_FUSE_STEPS, cfg.COMPILER_LOG_IR,
    )

    if getattr(cfg, "ACTIVITY_REG_ENABLED", False) or getattr(cfg, "STDP_ENABLED", False):
        logger.warning(
            "[COMPILER] activity_reg and STDP hooks are bypassed by the CUDA kernel path. "
            "Set activity_reg_enabled: false and stdp_enabled: false in SNN_module.yaml "
            "when using the compiler, otherwise their penalties will be zero."
        )

    graph = lower_to_ir(model, cfg)
    graph = schedule(graph, cfg)
    plan  = build_plan(graph)

    # Guard: if the plan has no real computation steps the model architecture
    # was not recognised — return the original model rather than a broken wrapper
    from compiler.src.ir import OpType
    useful = [s for s in plan.steps
              if isinstance(s, FusedStep) or (
                  isinstance(s, AtomicStep) and
                  s.node.op not in (OpType.INPUT, OpType.OUTPUT, OpType.AGGREGATE)
              )]
    if not useful:
        logger.warning(
            "[COMPILER] No executable steps in plan — model architecture was not "
            "recognised by the lowering pass. Falling back to standard PyTorch execution."
        )
        return model

    if cfg.COMPILER_LOG_IR:
        logger.info("[COMPILER] Execution plan:\n%s", plan.summary())

    # Trigger NVCC compilation now so it completes before the first training batch.
    if cfg.COMPILER_BACKEND == "cuda":
        try:
            from compiler.cuda_ops import _load_ops
            _load_ops()
        except Exception:
            pass

    logger.info("[COMPILER] Compilation complete")
    return CompiledModel(model, plan, cfg)
