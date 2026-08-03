"""
Norse regression variant — shared by every Phase B dataset whose task_type is
"regression" (currently MVSEC and TUM-VIE). Same conv+LIF backbone as snn_norse.py;
the output stage is a plain linear readout (cfg.REGRESSION_OUTPUT_DIM) instead of a
spiking classification head, averaged over time by loss_fn's "mse_regression"
reduction.

Not wired to real training yet — see the same note in snn_torch_regression.py and
docs/Haseeb-open-items.md (target-extraction adapter still needed in SNNTrainer).
"""
import torch
import torch.nn as nn
import norse.torch as norse

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes
from learning.utilities import build_optimizer, build_loss


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
            "loss_fn":       "mse_regression",
        }

        self.conv1   = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.lif1    = build_norse_layer("lif1", cfg)
        self.pool1   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.conv2   = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.lif2    = build_norse_layer("lif2", cfg)
        self.pool2   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.flatten = nn.Flatten()
        # Plain linear readout — continuous output, no spiking output layer.
        self.readout = nn.Linear(cfg.FC_IN, cfg.REGRESSION_OUTPUT_DIM)

        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="norse")

        register_activity_hooks(self, {'lif1': self.lif1, 'lif2': self.lif2})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns [T, B, REGRESSION_OUTPUT_DIM]."""
        clear_hidden_spikes(self)
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

            x = self.flatten(x)
            readout_rec.append(self.readout(x))

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
