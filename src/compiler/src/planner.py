"""
Planner: converts a scheduled ComputeGraph into an ordered ExecutionPlan.

Fused LIF quartets become a single FusedStep; all other nodes become AtomicStep.
The planner merges node attrs into each FusedStep so the runtime has everything
it needs without touching the graph again.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from compiler.src.ir import ComputeGraph, IRNode, OpType

logger = logging.getLogger(__name__)


@dataclass
class AtomicStep:
    node: IRNode


@dataclass
class FusedStep:
    nodes:    List[IRNode]
    group_id: int
    attrs:    Dict = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    steps: List

    def summary(self) -> str:
        lines = [f"ExecutionPlan ({len(self.steps)} steps):"]
        for s in self.steps:
            if isinstance(s, FusedStep):
                ops = [n.op.name for n in s.nodes]
                lines.append(f"  FusedStep(group={s.group_id}, ops={ops})")
            else:
                lines.append(f"  AtomicStep({s.node.op.name}, name={s.node.name!r})")
        return "\n".join(lines)


def build_plan(graph: ComputeGraph) -> ExecutionPlan:
    nodes   = graph.topological_order()
    steps   = []
    emitted = set()

    for node in nodes:
        if node.name in emitted:
            continue

        if node.fused and node.fuse_group is not None:
            group  = [n for n in nodes if n.fuse_group == node.fuse_group]
            merged = {}
            for n in group:
                merged.update({k: v for k, v in n.attrs.items() if k != "module"})
            steps.append(FusedStep(nodes=group, group_id=node.fuse_group, attrs=merged))
            for n in group:
                emitted.add(n.name)
        else:
            steps.append(AtomicStep(node=node))
            emitted.add(node.name)

    fused_count = sum(1 for s in steps if isinstance(s, FusedStep))
    logger.info("[PLANNER] %d steps (%d fused LIF groups)", len(steps), fused_count)
    return ExecutionPlan(steps=steps)
