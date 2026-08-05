import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_sj_layer(layer_name: str, cfg: Settings, spike_grad, **kwargs) -> nn.Module:
    """
    Build a SpikingJelly neuron for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.spikingjelly):
      izhikevich — neuron.IzhikevichNode. Default.
      lif        — neuron.LIFNode.
    """
    neuron_type = cfg.NEURON_TYPES.get("spikingjelly", {}).get(layer_name, "izhikevich")
    fw_cfg      = cfg.FRAMEWORK_CFG["spikingjelly"]
    tau         = fw_cfg["tau"]
    threshold   = fw_cfg["threshold"]

    if neuron_type == "izhikevich":
        return neuron.IzhikevichNode(
            tau=tau, v_threshold=threshold,
            surrogate_function=spike_grad, **kwargs,
        )
    return neuron.LIFNode(
        tau=tau, v_threshold=threshold,
        surrogate_function=spike_grad, **kwargs,
    )


class SNN_SJ(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings, spike_grad=surrogate.ATan()):
        super().__init__()
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["spikingjelly"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_sj_layer("lif1",    cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_sj_layer("lif2",    cfg, spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES),
            build_sj_layer("lif_out", cfg, spike_grad),
        ).to(self.device)

        self.optimizer = build_optimizer(self.net.parameters(), fw_cfg)
        # SpikingJelly forward sums over T and returns [B, C], so build_loss uses
        # nn.CrossEntropyLoss directly (no sum-over-T lambda needed).
        self.loss_fn   = build_loss(fw_cfg, framework="spikingjelly")

        # net[1] = lif1 (after conv1), net[4] = lif2 (after conv2)
        self.activity = ActivityMonitor({'lif1': self.net[1], 'lif2': self.net[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and return the SUM of spikes [B, num_classes]."""
        self.activity.clear()
        functional.reset_net(self.net)

        # data shape is [T, B, C, H, W]
        sum_spikes = torch.zeros(data.size(1), self.cfg.NUM_CLASSES, device=self.device)
        for step in range(data.size(0)):
            sum_spikes = sum_spikes + self.net(data[step])

        return sum_spikes

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
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_state(self) -> dict:
        return {
            "model_state_dict":     self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
