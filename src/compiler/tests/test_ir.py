import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.src.ir import ComputeGraph, DeviceTag, IRNode, OpType


def test_add_and_topological_order():
    g = ComputeGraph()
    g.add(IRNode(op=OpType.INPUT,  name="input"))
    g.add(IRNode(op=OpType.DENSE,  name="dense",  inputs=["input"]))
    g.add(IRNode(op=OpType.OUTPUT, name="output", inputs=["dense"]))

    order = [n.name for n in g.topological_order()]
    assert order == ["input", "dense", "output"], order


def test_node_repr_includes_fuse_group():
    n = IRNode(op=OpType.SPIKE_GEN, name="spk0", fused=True, fuse_group=2)
    assert "fuse_group=2" in repr(n)


def test_graph_repr():
    g = ComputeGraph()
    g.add(IRNode(op=OpType.INPUT, name="input"))
    r = repr(g)
    assert "ComputeGraph" in r and "input" in r


if __name__ == "__main__":
    test_add_and_topological_order()
    test_node_repr_includes_fuse_group()
    test_graph_repr()
    print("test_ir: all passed")
