import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from compiler.src.ir import ComputeGraph, IRNode, OpType
from compiler.src.planner import build_plan, FusedStep
from compiler.src.runtime import execute, lif_step
import compiler.passes.fusion as fusion


def make_fused_plan():
    g = ComputeGraph()
    g.add(IRNode(op=OpType.INPUT,           name="input"))
    g.add(IRNode(op=OpType.MEMBRANE_UPDATE, name="mem0", inputs=["input"],
                 attrs={"beta": 0.9}))
    g.add(IRNode(op=OpType.THRESHOLD,       name="thr0", inputs=["mem0"],
                 attrs={"threshold": 1.0}))
    g.add(IRNode(op=OpType.SPIKE_GEN,       name="spk0", inputs=["thr0"]))
    g.add(IRNode(op=OpType.RESET,           name="rst0", inputs=["spk0", "mem0"],
                 attrs={"beta": 0.9}))
    g.add(IRNode(op=OpType.AGGREGATE,       name="agg",  inputs=["rst0"]))
    g.add(IRNode(op=OpType.OUTPUT,          name="out",  inputs=["agg"]))
    g = fusion.run(g)
    return build_plan(g)


def test_plan_contains_fused_step():
    plan = make_fused_plan()
    fused = [s for s in plan.steps if isinstance(s, FusedStep)]
    assert len(fused) == 1


def test_execute_returns_correct_shape():
    plan   = make_fused_plan()
    x      = torch.rand(5, 8)    # [T=5, features=8]
    result = execute(plan, x)
    assert result.shape == (8,) or result.shape[-1] == 8


def test_lif_step_fires_above_threshold():
    x   = torch.ones(4) * 2.0
    mem = torch.zeros(4)
    spk, mem_new = lif_step(x, mem, beta=0.9, threshold=1.0)
    assert spk.sum().item() == 4          # all neurons fire
    assert (mem_new < 1.0).all()          # membrane reset


def test_lif_step_silent_below_threshold():
    x   = torch.ones(4) * 0.1
    mem = torch.zeros(4)
    spk, _ = lif_step(x, mem, beta=0.9, threshold=1.0)
    assert spk.sum().item() == 0          # no spikes


if __name__ == "__main__":
    test_plan_contains_fused_step()
    test_execute_returns_correct_shape()
    test_lif_step_fires_above_threshold()
    test_lif_step_silent_below_threshold()
    print("test_runtime: all passed")
