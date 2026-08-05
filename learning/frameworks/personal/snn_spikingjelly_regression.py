"""
SpikingJelly dense-regression variant — DSEC optical flow (the only regression
dataset left). Same conv+LIF backbone as snn_spikingjelly.py; the output stage is
a dense decoder (learning/frameworks/personal/dense_head.py) predicting a per-pixel
(flow_x, flow_y) map. Unlike the Torch/Norse/Sinabs variants, forward() reduces
over T itself (mean, not sum) and returns [B, 2, H, W] directly, matching
snn_spikingjelly.py's own T-reduction convention. Trained with flow_masked_mse
against DSEC's own (H, W, 3) flow+valid-mask target layout.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.personal.dense_head import DenseDecoder
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_sj_layer(layer_name: str, cfg: Settings, spike_grad, **kwargs) -> nn.Module:
    """Same neuron factory as snn_spikingjelly.py — see that file for izhikevich/lif."""
    neuron_type = cfg.NEURON_TYPES.get("spikingjelly", {}).get(layer_name, "izhikevich")
    fw_cfg      = cfg.FRAMEWORK_CFG["spikingjelly"]
    tau         = fw_cfg["tau"]
    threshold   = fw_cfg["threshold"]

    if neuron_type == "izhikevich":
        return neuron.IzhikevichNode(tau=tau, v_threshold=threshold, surrogate_function=spike_grad, **kwargs)
    return neuron.LIFNode(tau=tau, v_threshold=threshold, surrogate_function=spike_grad, **kwargs)


class SNN_SJ_REGRESSION(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.ATan()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["spikingjelly"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
            "loss_fn":       "flow_masked_mse",
        }

        # No nn.Flatten() — the decoder needs the spatial feature map, not a flat vector.
        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_sj_layer("lif1", cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_sj_layer("lif2", cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
        ).to(self.device)
        self.decoder = DenseDecoder(cfg, out_channels=2).to(self.device)

        self.optimizer = build_optimizer(list(self.backbone.parameters()) + list(self.decoder.parameters()), fw_cfg)
        # forward() averages over T and returns [B, 2, H, W] already (4-dim) — flow_masked_mse
        # skips the mean(0) reduction in that case, see its own docstring.
        self.loss_fn   = build_loss(fw_cfg, framework="spikingjelly")

        self.activity = ActivityMonitor({'lif1': self.backbone[1], 'lif2': self.backbone[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns the MEAN prediction over T, [B, 2, SENSOR_H, SENSOR_W]."""
        self.activity.clear()
        functional.reset_net(self.backbone)

        T = data.size(0)
        readout_sum = torch.zeros(data.size(1), 2, self.cfg.SENSOR_H, self.cfg.SENSOR_W, device=self.device)
        for step in range(T):
            features = self.backbone(data[step])
            readout_sum = readout_sum + self.decoder(features)

        return readout_sum / T

    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        if scaler is not None:
            scaler.scale(loss).backward()
            if do_step:
                scaler.step(self.optimizer)
                scaler.update()
        else:
            loss.backward()
            if do_step:
                self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def train_mode(self) -> None:
        self.backbone.train()
        self.decoder.train()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.decoder.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_state(self) -> dict:
        return {
            "backbone_state_dict":  self.backbone.state_dict(),
            "decoder_state_dict":   self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
