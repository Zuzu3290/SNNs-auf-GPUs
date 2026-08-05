"""
SNNTorch dense-regression variant — DSEC optical flow (the only regression dataset
left; see event_data_workflow/data_pipeline.py's DATASET_REGISTRY). Same conv+LIF
backbone as snn_torch.py; the difference is the output stage — a dense decoder
(learning/frameworks/personal/dense_head.py) predicting a per-pixel (flow_x, flow_y)
map instead of a flat classification/pose vector, trained with flow_masked_mse
(build_loss) against DSEC's own (H, W, 3) flow+valid-mask target layout.
"""
import snntorch as snn
import torch
import torch.nn as nn
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.personal.dense_head import DenseDecoder
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


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
            "loss_fn":       "flow_masked_mse",
        }

        # No nn.Flatten() — the decoder needs the spatial feature map, not a flat vector.
        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_lif_layer("lif1", cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_lif_layer("lif2", cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
        ).to(self.device)
        self.decoder = DenseDecoder(cfg, out_channels=2).to(self.device)

        self.optimizer = build_optimizer(list(self.backbone.parameters()) + list(self.decoder.parameters()), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="torch")

        self.activity = ActivityMonitor({'lif1': self.backbone[1], 'lif2': self.backbone[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W] -> returns [T, B, 2, SENSOR_H, SENSOR_W] (raw per-timestep
        flow prediction; loss_fn averages over T, there's no spike count to sum)."""
        self.activity.clear()
        from snntorch import utils
        utils.reset(self.backbone)

        readout_rec = []
        for step in range(data.size(0)):
            features = self.backbone(data[step])
            readout_rec.append(self.decoder(features))

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
