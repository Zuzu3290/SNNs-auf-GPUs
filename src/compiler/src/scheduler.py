"""
Scheduler: applies compiler passes to the ComputeGraph in order.
    1. op_rewrite   — validate and normalise
    2. device_annotation — stamp target device on every node
    3. fusion        — mark LIF quartets as fused groups (if enabled)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # project root → skeleton
sys.path.insert(0, str(Path(__file__).parent.parent.parent))          # src/ → compiler, learning
from skeleton import Settings
from compiler.src.ir import ComputeGraph
import compiler.passes.op_rewrite        as op_rewrite
import compiler.passes.device_annotation as device_annotation
import compiler.passes.fusion            as fusion

logger = logging.getLogger(__name__)


def schedule(graph: ComputeGraph, cfg: Settings) -> ComputeGraph:
    graph = op_rewrite.run(graph)
    graph = device_annotation.run(graph, cfg.COMPILER_BACKEND)

    if cfg.COMPILER_FUSE_STEPS:
        graph = fusion.run(graph)
    else:
        logger.info("[SCHEDULER] Timestep fusion disabled via config")

    logger.info(
        "[SCHEDULER] Done — %d nodes, backend=%s, fuse=%s",
        len(graph.nodes), cfg.COMPILER_BACKEND, cfg.COMPILER_FUSE_STEPS,
    )
    return graph
