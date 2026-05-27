"""
Device-annotation pass: stamps every IR node with its target device.
"""
from __future__ import annotations

from compiler.src.ir import ComputeGraph, DeviceTag


def run(graph: ComputeGraph, backend: str) -> ComputeGraph:
    tag = DeviceTag.CUDA if backend == "cuda" else DeviceTag.CPU
    for node in graph.nodes:
        node.device = tag
    return graph
