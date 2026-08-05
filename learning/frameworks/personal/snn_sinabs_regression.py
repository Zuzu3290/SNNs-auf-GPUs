"""
Sinabs dense-regression variant — DSEC optical flow (the only regression dataset
left). Same conv+LIF backbone as snn_sinabs.py; the output stage is a dense
decoder (learning/frameworks/personal/dense_head.py) predicting a per-pixel
(flow_x, flow_y) map, trained with flow_masked_mse against DSEC's own
(H, W, 3) flow+valid-mask target layout.
"""
import torch
import torch.nn as nn
import sinabs.layers as sl

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.personal.dense_head import DenseDecoder
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_sinabs_layer(layer_name: str, cfg: Settings, **kwargs) -> nn.Module:
    """Same neuron factory as snn_sinabs.py — see that file for lif/iaf."""
    neuron_type = cfg.NEURON_TYPES.get("sinabs", {}).get(layer_name, "lif")
    fw_cfg      = cfg.FRAMEWORK_CFG["sinabs"]
    threshold   = torch.as_tensor(fw_cfg["threshold"])

    if neuron_type == "iaf":
        return sl.IAF(spike_threshold=threshold, **kwargs)
    return sl.LIF(tau_mem=fw_cfg["tau_mem"], spike_threshold=threshold, **kwargs)


class SNN_SINABS_REGRESSION(ModelInterface, nn.Module):
    """Batch-first (B, T, ...) like snn_sinabs.py — see that file's docstring for why."""

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["sinabs"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
            "loss_fn":       "flow_masked_mse",
        }

        self.flatten_t = sl.FlattenTime()
        self.conv1     = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.pool1     = nn.MaxPool2d(cfg.POOL_KERNEL)
        self.conv2     = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.pool2     = nn.MaxPool2d(cfg.POOL_KERNEL)
        # No nn.Flatten() to a vector — the decoder needs the spatial feature map.
        self.decoder   = DenseDecoder(cfg, out_channels=2)

        self.lif1 = build_sinabs_layer("lif1", cfg)
        self.lif2 = build_sinabs_layer("lif2", cfg)

        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="sinabs")

        # No hooked layers — see snn_sinabs.py's __init__ for why (Sinabs calls
        # each LIF layer once per forward with the whole (B,T,...) tensor, not
        # once per timestep). ActivityMonitor() with no layer_map is a safe no-op.
        self.activity = ActivityMonitor()

    def tensor_format(self) -> str:
        return "BT"

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [B, T, C, H, W] -> returns [T, B, 2, SENSOR_H, SENSOR_W] (transposed back
        to time-first, matching snn_sinabs.py's convention; loss_fn averages over T)."""
        self.activity.clear()
        for layer in (self.lif1, self.lif2):
            layer.reset_states()

        B, T = data.size(0), data.size(1)

        x = self.flatten_t(data)
        x = self.conv1(x)
        x = x.unflatten(0, (B, T))
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.pool1(x)

        x = self.conv2(x)
        x = x.unflatten(0, (B, T))
        x = self.lif2(x)
        x = x.flatten(0, 1)
        x = self.pool2(x)

        x = self.decoder(x)             # [B*T, 2, SENSOR_H, SENSOR_W]
        x = x.unflatten(0, (B, T))      # [B, T, 2, SENSOR_H, SENSOR_W]

        return x.transpose(0, 1)

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
        self.train()

    def eval_mode(self) -> None:
        self.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_state(self) -> dict:
        return {
            "model_state_dict":     self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
