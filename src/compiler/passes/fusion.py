"""
Fusion pass: marks consecutive MembraneUpdate → Threshold → SpikeGen → Reset
sequences as a fused group so the runtime executes them as one combined op
instead of four separate dispatches.
"""
from __future__ import annotations

import logging
from compiler.src.ir import ComputeGraph, OpType

logger = logging.getLogger(__name__)

LIF_QUARTET = [OpType.MEMBRANE_UPDATE, OpType.THRESHOLD, OpType.SPIKE_GEN, OpType.RESET]


def run(graph: ComputeGraph) -> ComputeGraph:
    nodes    = graph.topological_order()
    group_id = 0
    i        = 0

    while i < len(nodes):
        window = [nodes[j].op for j in range(i, min(i + 4, len(nodes)))]
        if window == LIF_QUARTET:
            for j in range(i, i + 4):
                nodes[j].fused      = True
                nodes[j].fuse_group = group_id
            logger.debug(
                "[FUSION] Fused LIF group %d: %s → %s",
                group_id, nodes[i].name, nodes[i + 3].name,
            )
            group_id += 1
            i += 4
        else:
            i += 1

    if group_id:
        logger.info("[FUSION] %d fused LIF group(s) marked", group_id)
    return graph
