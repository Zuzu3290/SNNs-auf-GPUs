"""
SpikingJelly regression variant — shared by every Phase B dataset whose task_type is
"regression" (currently MVSEC and TUM-VIE). Same conv+LIF backbone as
snn_spikingjelly.py; the output stage is a plain linear readout
(cfg.REGRESSION_OUTPUT_DIM) instead of a spiking classification head. Unlike the
Torch/Norse/Sinabs variants, forward() reduces over T itself (mean, not sum — this
is a continuous analog readout, not a spike count) and returns [B, output_dim]
directly, matching snn_spikingjelly.py's own T-reduction convention.

Not wired to real training yet — see the same note in snn_torch_regression.py and
docs/Haseeb-open-items.md (target-extraction adapter still needed in SNNTrainer).
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes
from learning.utilities import build_optimizer, build_loss


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
            "loss_fn":       "mse_regression",
        }

        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_sj_layer("lif1", cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_sj_layer("lif2", cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
        ).to(self.device)
        self.readout = nn.Linear(cfg.FC_IN, cfg.REGRESSION_OUTPUT_DIM).to(self.device)

        self.optimizer = build_optimizer(list(self.backbone.parameters()) + list(self.readout.parameters()), fw_cfg)
        # forward() averages over T and returns [B, output_dim] already — plain nn.MSELoss.
        self.loss_fn   = build_loss(fw_cfg, framework="spikingjelly")

        register_activity_hooks(self, {'lif1': self.backbone[1], 'lif2': self.backbone[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns the MEAN readout over T, [B, output_dim]."""
        clear_hidden_spikes(self)
        functional.reset_net(self.backbone)

        T = data.size(0)
        readout_sum = torch.zeros(data.size(1), self.cfg.REGRESSION_OUTPUT_DIM, device=self.device)
        for step in range(T):
            features = self.backbone(data[step])
            readout_sum = readout_sum + self.readout(features)

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
        self.readout.train()

    def eval_mode(self) -> None:
        self.backbone.eval()
        self.readout.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_state(self) -> dict:
        return {
            "backbone_state_dict":  self.backbone.state_dict(),
            "readout_state_dict":   self.readout.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
