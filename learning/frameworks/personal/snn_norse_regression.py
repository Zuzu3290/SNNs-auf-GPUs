"""
Norse dense-regression variant — DSEC optical flow (the only regression dataset
left). Same conv+LIF backbone as snn_norse.py; the output stage is a dense decoder
(learning/frameworks/personal/dense_head.py) predicting a per-pixel (flow_x, flow_y)
map, trained with flow_masked_mse against DSEC's own (H, W, 3) flow+valid-mask
target layout.
"""
import torch
import torch.nn as nn
import norse.torch as norse

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.personal.dense_head import DenseDecoder
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_norse_layer(layer_name: str, cfg: Settings) -> nn.Module:
    """Same neuron factory as snn_norse.py — see that file for lif_cell/lif_rec_cell."""
    neuron_type = cfg.NEURON_TYPES.get("norse", {}).get(layer_name, "lif_cell")
    fw_cfg      = cfg.FRAMEWORK_CFG["norse"]

    lif_params = norse.LIFParameters(
        tau_mem_inv = torch.as_tensor(fw_cfg["tau_mem_inv"], dtype=torch.float32),
        v_th        = torch.as_tensor(fw_cfg["threshold"],   dtype=torch.float32),
    )
    if neuron_type == "lif_rec_cell":
        raise NotImplementedError(
            f"lif_rec_cell for layer '{layer_name}' requires input_size and hidden_size. "
            "Subclass SNN_NORSE_REGRESSION and override the layer construction."
        )
    return norse.LIFCell(p=lif_params)


class SNN_NORSE_REGRESSION(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["norse"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
            "loss_fn":       "flow_masked_mse",
        }

        self.conv1   = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.lif1    = build_norse_layer("lif1", cfg)
        self.pool1   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.conv2   = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.lif2    = build_norse_layer("lif2", cfg)
        self.pool2   = nn.MaxPool2d(cfg.POOL_KERNEL)

        # No flatten — the decoder needs the spatial feature map, not a flat vector.
        self.decoder = DenseDecoder(cfg, out_channels=2)

        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="norse")

        self.activity = ActivityMonitor({'lif1': self.lif1, 'lif2': self.lif2})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns [T, B, 2, SENSOR_H, SENSOR_W]."""
        self.activity.clear()
        s1 = s2 = None
        readout_rec = []

        for step in range(data.size(0)):
            x = data[step]

            x = self.conv1(x)
            x, s1 = self.lif1(x, s1)
            x = self.pool1(x)

            x = self.conv2(x)
            x, s2 = self.lif2(x, s2)
            x = self.pool2(x)

            readout_rec.append(self.decoder(x))

        return torch.stack(readout_rec)

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
