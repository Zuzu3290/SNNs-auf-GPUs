"""
SNNTorch regression variant — shared by every Phase B dataset whose task_type is
"regression" (currently MVSEC and TUM-VIE; see event_data_workflow/data_pipeline.py's
DATASET_REGISTRY). Same conv+LIF backbone as snn_torch.py; the difference is only the
output stage: a plain linear readout (cfg.REGRESSION_OUTPUT_DIM), no spiking output
layer, averaged over time by loss_fn's "mse_regression" reduction instead of a
classification head decoded via spike count.

Not wired to real training yet — SNNTrainer.train() still assumes `targets` arriving
from the DataLoader is a plain tensor. MVSEC/TUM-VIE's collated targets are a list of
raw dicts/tuples (event_data_workflow/data_pipeline.py's pad_events_passthrough_target),
so a target-extraction adapter is still needed before this runs end-to-end. See
docs/Haseeb-open-items.md.
"""
import snntorch as snn
import torch
import torch.nn as nn
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes
from learning.utilities import build_optimizer, build_loss


def build_lif_layer(layer_name: str, cfg: Settings, spike_grad, **kwargs) -> nn.Module:
    """Same neuron factory as snn_torch.py — see that file for the alpha/leaky choice."""
    neuron_type = cfg.NEURON_TYPES.get("snntorch", {}).get(layer_name, "alpha")
    fw_cfg      = cfg.FRAMEWORK_CFG["snntorch"]
    beta        = fw_cfg["beta"]
    threshold   = fw_cfg["threshold"]

    if neuron_type == "alpha":
        return snn.Alpha(
            alpha=beta, beta=max(0.5, beta - 0.1),
            threshold=threshold, spike_grad=spike_grad,
            **kwargs,
        )
    return snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, **kwargs)


class SNN_TORCH_REGRESSION(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            from snntorch import surrogate
            spike_grad = surrogate.atan()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["snntorch"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
            "loss_fn":       "mse_regression",
        }

        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_lif_layer("lif1", cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_lif_layer("lif2", cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
        ).to(self.device)
        # Plain linear readout — continuous output, no spiking nonlinearity at the head.
        self.readout = nn.Linear(cfg.FC_IN, cfg.REGRESSION_OUTPUT_DIM).to(self.device)

        self.optimizer = build_optimizer(list(self.backbone.parameters()) + list(self.readout.parameters()), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="torch")

        register_activity_hooks(self, {'lif1': self.backbone[1], 'lif2': self.backbone[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns [T, B, REGRESSION_OUTPUT_DIM] (raw per-timestep
        analog readout; loss_fn averages over T, there's no spike count to sum)."""
        clear_hidden_spikes(self)
        from snntorch import utils
        utils.reset(self.backbone)

        readout_rec = []
        for step in range(data.size(0)):
            features = self.backbone(data[step])
            readout_rec.append(self.readout(features))

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
