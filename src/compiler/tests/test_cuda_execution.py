import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch


def test_lif_step_on_cuda():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device found")
        return

    from compiler.src.runtime import lif_step

    x   = torch.ones(4, 16, device="cuda") * 1.5
    mem = torch.zeros(4, 16, device="cuda")

    spk, mem_new = lif_step(x, mem, beta=0.9, threshold=1.0)

    assert spk.device.type == "cuda"
    assert spk.shape == (4, 16)
    assert spk.sum().item() == 4 * 16          # all neurons fire
    assert (mem_new < 1.0).all()               # all membranes reset

    print(f"  Spikes: {int(spk.sum().item())}/{spk.numel()}  "
          f"Mean mem after reset: {mem_new.mean().item():.4f}")


def test_execute_on_cuda():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device found")
        return

    from compiler.src.ir import ComputeGraph, IRNode, OpType
    from compiler.src.planner import build_plan
    from compiler.src.runtime import execute
    import compiler.passes.fusion as fusion

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
    plan = build_plan(g)

    x      = torch.rand(10, 32, device="cuda")
    result = execute(plan, x)

    assert result.device.type == "cuda"
    assert result.shape[-1] == 32
    print(f"  Output shape: {tuple(result.shape)}  "
          f"Mean spikes: {result.mean().item():.4f}")


if __name__ == "__main__":
    test_lif_step_on_cuda()
    test_execute_on_cuda()
    print("test_cuda_execution: all passed")
