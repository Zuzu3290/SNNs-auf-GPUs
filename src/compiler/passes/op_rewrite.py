"""
Op-rewrite pass: validates and normalises IR before scheduling.
Warns if a MembraneUpdate is not followed by the expected LIF quartet.
"""
from __future__ import annotations

import logging
from compiler.src.ir import ComputeGraph, OpType

logger = logging.getLogger(__name__)


def run(graph: ComputeGraph) -> ComputeGraph:
    nodes    = graph.topological_order()
    expected = [OpType.THRESHOLD, OpType.SPIKE_GEN, OpType.RESET]

    for i, node in enumerate(nodes):
        if node.op == OpType.MEMBRANE_UPDATE:
            tail = [n.op for n in nodes[i + 1 : i + 4]]
            if tail != expected:
                logger.warning(
                    "[OP_REWRITE] Incomplete LIF quartet at %r — got %s",
                    node.name, [o.name for o in tail],
                )
    return graph
