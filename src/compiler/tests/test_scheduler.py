import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.src.ir import ComputeGraph, DeviceTag, IRNode, OpType
import compiler.passes.device_annotation as device_annotation
import compiler.passes.fusion as fusion


def make_lif_graph() -> ComputeGraph:
    g = ComputeGraph()
    g.add(IRNode(op=OpType.INPUT,           name="input"))
    g.add(IRNode(op=OpType.DENSE,           name="dense",   inputs=["input"]))
    g.add(IRNode(op=OpType.MEMBRANE_UPDATE, name="mem0",    inputs=["dense"]))
    g.add(IRNode(op=OpType.THRESHOLD,       name="thr0",    inputs=["mem0"]))
    g.add(IRNode(op=OpType.SPIKE_GEN,       name="spk0",    inputs=["thr0"]))
    g.add(IRNode(op=OpType.RESET,           name="rst0",    inputs=["spk0", "mem0"]))
    g.add(IRNode(op=OpType.AGGREGATE,       name="agg",     inputs=["rst0"]))
    g.add(IRNode(op=OpType.OUTPUT,          name="output",  inputs=["agg"]))
    return g


def test_device_annotation_cuda():
    g = device_annotation.run(make_lif_graph(), "cuda")
    assert all(n.device == DeviceTag.CUDA for n in g.nodes)


def test_device_annotation_cpu():
    g = device_annotation.run(make_lif_graph(), "cpu")
    assert all(n.device == DeviceTag.CPU for n in g.nodes)


def test_fusion_marks_four_nodes():
    g = fusion.run(make_lif_graph())
    fused = [n for n in g.nodes if n.fused]
    assert len(fused) == 4
    assert all(n.fuse_group == 0 for n in fused)


def test_fusion_ops_are_correct():
    g = fusion.run(make_lif_graph())
    fused_ops = [n.op for n in g.nodes if n.fused]
    assert fused_ops == [
        OpType.MEMBRANE_UPDATE, OpType.THRESHOLD,
        OpType.SPIKE_GEN, OpType.RESET,
    ]


if __name__ == "__main__":
    test_device_annotation_cuda()
    test_device_annotation_cpu()
    test_fusion_marks_four_nodes()
    test_fusion_ops_are_correct()
    print("test_scheduler: all passed")
