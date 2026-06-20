"""
BindsNET 0.2.7 (latest on PyPI, unmaintained since ~2022) imports `torch._six`,
which PyTorch removed years ago. The shim below restores just enough of it for
BindsNET's import chain to succeed — narrow and load-bearing, not a style choice.
"""
import sys
import types
import collections.abc

if "torch._six" not in sys.modules:
    _six_shim = types.ModuleType("torch._six")
    _six_shim.container_abcs = collections.abc
    _six_shim.string_classes  = (str,)
    _six_shim.int_classes     = (int,)
    sys.modules["torch._six"] = _six_shim

import torch
import torch.nn as nn
import torch.nn.functional as F
import bindsnet.network as bnet
import bindsnet.network.nodes as bnodes
import bindsnet.network.topology as btopo
import bindsnet.learning as blearning
from bindsnet.network.monitors import Monitor

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface


class _BindsNetStage:
    """
    One Input→Nodes hop trained by local STDP (PostPre): either a Conv2dConnection
    (conv stages) or a plain Connection (the final FC stage — BindsNET flattens
    any source shape automatically, so it doesn't need to match cfg.FC_IN itself).

    BindsNET's MaxPool2dConnection has a real bug in 0.2.7: its firing-rate buffer
    is allocated without the batch dimension (`torch.ones(source.shape)`), so
    `self.firing_rates += s.float()` throws a broadcast error for any batch_size > 1.
    Pooling is therefore done with plain `F.max_pool2d` on the recorded spike train
    BETWEEN stages instead of inside a BindsNET connection — confirmed against the
    real package; see docs/frameworks/additional_frameworks.md for the repro.

    `one_step=True` makes the single connection in each stage propagate spikes to
    the target layer within the same simulation timestep, matching how the other
    3 backends compute a full conv→lif pass per timestep rather than per-layer.
    """

    def __init__(self, source, target, connection, batch_size: int):
        self.network = bnet.Network(batch_size=batch_size)
        self.network.add_layer(source, name="in")
        self.network.add_layer(target, name="out")
        self.network.add_connection(connection, source="in", target="out")

        self.monitor = Monitor(target, state_vars=["s"], batch_size=batch_size)
        self.network.add_monitor(self.monitor, name="mon")

        self.connection = connection
        self.batch_size = batch_size

    def run(self, spikes: torch.Tensor, time: int) -> torch.Tensor:
        self.network.run(inputs={"in": spikes}, time=time, one_step=True)
        out = self.monitor.get("s").float()
        self.network.reset_state_variables()
        return out

    def to(self, device):
        self.network.to(device)
        return self


def build_bindsnet_stage(layer_name: str, in_shape: tuple, out_shape_or_n, cfg: Settings,
                          kernel_size=None, is_fc: bool = False) -> _BindsNetStage:
    """
    Build one conv/FC + STDP stage.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.bindsnet):
      lif — bindsnet.network.nodes.LIFNodes. Only option currently wired up.
    """
    fw_cfg = cfg.FRAMEWORK_CFG["bindsnet"]
    nu     = (fw_cfg["nu_pre"], fw_cfg["nu_post"])

    source = bnodes.Input(shape=in_shape, traces=True)
    if is_fc:
        target     = bnodes.LIFNodes(n=out_shape_or_n, traces=True)
        connection = btopo.Connection(source, target, update_rule=blearning.PostPre, nu=nu)
    else:
        target     = bnodes.LIFNodes(shape=out_shape_or_n, traces=True)
        connection = btopo.Conv2dConnection(source, target, kernel_size=kernel_size, stride=1,
                                             update_rule=blearning.PostPre, nu=nu)

    return _BindsNetStage(source, target, connection, batch_size=cfg.BATCH_SIZE)


class SNN_BINDSNET(ModelInterface):
    """
    BindsNET is the only backend here that doesn't train via backprop at all —
    weights update through real STDP (PostPre: pre-before-post strengthens,
    post-before-pre weakens) applied directly inside Network.run(), the same
    mechanism this project's activity_reg.py STDP loss term only approximates as
    a soft backprop-compatible penalty. is_differentiable() is False so the
    adversarial evaluator skips attack generation, matching the JAX/TF pattern
    documented in docs/frameworks/README.md.

    Architecture is split into 3 independently-trained stages (conv1→lif1,
    conv2→lif2, fc→lif_out) chained by plain-tensor max-pooling — see
    _BindsNetStage's docstring for why.
    """

    def __init__(self, cfg: Settings):
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)
        self.pool_k = cfg.POOL_KERNEL

        h1  = cfg.SENSOR_H - cfg.CONV1_KERNEL + 1
        hp1 = h1 // cfg.POOL_KERNEL
        h2  = hp1 - cfg.CONV2_KERNEL + 1
        hp2 = h2 // cfg.POOL_KERNEL
        self._dims = dict(h1=h1, hp1=hp1, h2=h2, hp2=hp2)

        self.stage1 = build_bindsnet_stage(
            "lif1", (cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W), (cfg.CONV1_OUT, h1, h1),
            cfg, kernel_size=cfg.CONV1_KERNEL,
        ).to(self.device)
        self.stage2 = build_bindsnet_stage(
            "lif2", (cfg.CONV1_OUT, hp1, hp1), (cfg.CONV2_OUT, h2, h2),
            cfg, kernel_size=cfg.CONV2_KERNEL,
        ).to(self.device)
        self.stage3 = build_bindsnet_stage(
            "lif_out", (cfg.CONV2_OUT, hp2, hp2), cfg.NUM_CLASSES,
            cfg, is_fc=True,
        ).to(self.device)

        self.loss_fn = lambda spk_rec, targets: F.cross_entropy(spk_rec.float().sum(0), targets)

    def is_differentiable(self) -> bool:
        return False

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]  (default tensor_format — matches Network.run's own
              [time, batch, *shape] convention, no reshaping needed).
        returns: [T, B, num_classes]
        """
        T = data.size(0)
        pk = self.pool_k

        s1 = self.stage1.run(data, time=T)                                       # [T,B,conv1_out,h1,h1]
        s1_pooled = F.max_pool2d(
            s1.flatten(0, 1), kernel_size=pk, stride=pk
        ).view(T, -1, self.cfg.CONV1_OUT, self._dims["hp1"], self._dims["hp1"])

        s2 = self.stage2.run(s1_pooled, time=T)                                   # [T,B,conv2_out,h2,h2]
        s2_pooled = F.max_pool2d(
            s2.flatten(0, 1), kernel_size=pk, stride=pk
        ).view(T, -1, self.cfg.CONV2_OUT, self._dims["hp2"], self._dims["hp2"])

        out = self.stage3.run(s2_pooled, time=T)                                  # [T,B,num_classes]
        return out

    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        pass  # weights already updated by STDP inside forward()

    def zero_grad(self) -> None:
        pass  # no gradients to zero — STDP, not backprop

    def train_mode(self) -> None:
        for stage in (self.stage1, self.stage2, self.stage3):
            stage.network.train(True)

    def eval_mode(self) -> None:
        for stage in (self.stage1, self.stage2, self.stage3):
            stage.network.train(False)

    def get_lr(self) -> float:
        return 0.0  # no optimizer learning rate — see nu_pre/nu_post in SNN_module.yaml

    def get_state(self) -> dict:
        return {
            "stage1_w": self.stage1.connection.w.detach().cpu(),
            "stage2_w": self.stage2.connection.w.detach().cpu(),
            "stage3_w": self.stage3.connection.w.detach().cpu(),
        }


if __name__ == "__main__":
    from event_data_workflow import NeuromorphicEncoder

    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_BINDSNET(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n BindsNET model ready.")
    print(f"  - Device : {model.device}")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
