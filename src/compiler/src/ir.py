from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class OpType(Enum):
    INPUT           = auto()
    CONV2D          = auto()
    DENSE           = auto()
    POOL2D          = auto()
    FLATTEN         = auto()
    MEMBRANE_UPDATE = auto()
    THRESHOLD       = auto()
    SPIKE_GEN       = auto()
    RESET           = auto()
    AGGREGATE       = auto()
    OUTPUT          = auto()


class DeviceTag(Enum):
    CPU  = "cpu"
    CUDA = "cuda"


@dataclass
class TensorMeta:
    shape: Tuple[int, ...]
    dtype: str = "float32"


@dataclass
class IRNode:
    op:          OpType
    name:        str
    inputs:      List[str]             = field(default_factory=list)
    output_meta: Optional[TensorMeta]  = None
    device:      DeviceTag             = DeviceTag.CUDA
    fused:       bool                  = False
    fuse_group:  Optional[int]         = None
    attrs:       Dict[str, Any]        = field(default_factory=dict)

    def __repr__(self) -> str:
        fuse = f" [fuse_group={self.fuse_group}]" if self.fused else ""
        return (
            f"IRNode({self.op.name}, name={self.name!r}, "
            f"device={self.device.value}{fuse})"
        )


class ComputeGraph:
    def __init__(self):
        self.nodes:    List[IRNode]      = []
        self.node_map: Dict[str, IRNode] = {}

    def add(self, node: IRNode) -> IRNode:
        self.nodes.append(node)
        self.node_map[node.name] = node
        return node

    def get(self, name: str) -> IRNode:
        return self.node_map[name]

    def topological_order(self) -> List[IRNode]:
        visited: set         = set()
        order:   List[IRNode] = []

        def visit(node: IRNode):
            if node.name in visited:
                return
            visited.add(node.name)
            for inp in node.inputs:
                if inp in self.node_map:
                    visit(self.node_map[inp])
            order.append(node)

        for n in self.nodes:
            visit(n)
        return order

    def __repr__(self) -> str:
        lines = [f"ComputeGraph ({len(self.nodes)} nodes):"]
        for n in self.nodes:
            lines.append(f"  {n}")
        return "\n".join(lines)
