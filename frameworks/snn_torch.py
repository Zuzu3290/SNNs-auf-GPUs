import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import torch.nn as nn
from skeleton.snn_config import Settings
from frameworks.model_interface import ModelInterface
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_lif_layer(layer_name: str, cfg: Settings, spike_grad, **kwargs) -> nn.Module:
    """
    Build an SNNTorch LIF neuron for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.snntorch):
      alpha  — snn.Alpha  (two-compartment: membrane + synaptic decay). Default.
      leaky  — snn.Leaky  (single-compartment).
    """
    neuron_type = cfg.NEURON_TYPES.get("snntorch", {}).get(layer_name, "alpha")
    fw_cfg      = cfg.FRAMEWORK_CFG["snntorch"]
    beta        = fw_cfg["beta"]
    threshold   = fw_cfg["threshold"]

    # snnTorch defaults to reset_mechanism="subtract" (soft reset) and
    # reset_delay=True (applies a spike's reset on the FOLLOWING timestep) —
    # neither matches Norse/SpikingJelly, which both reset immediately to a
    # hard 0. Left at the default, snnTorch's membrane reads a different value
    # than the other two frameworks after every spike, which silently biases
    # any cross-framework comparison. zero/False matches the other two.
    if neuron_type == "alpha":
        return snn.Alpha(
            alpha=beta, beta=max(0.5, beta - 0.1),
            threshold=threshold, spike_grad=spike_grad,
            reset_mechanism="zero", reset_delay=False,
            **kwargs,
        )
    return snn.Leaky(
        beta=beta, threshold=threshold, spike_grad=spike_grad,
        reset_mechanism="zero", reset_delay=False,
        **kwargs,
    )


class SNN_TORCH(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings, spike_grad=surrogate.atan()):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["snntorch"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_lif_layer("lif1",    cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_lif_layer("lif2",    cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES),
            build_lif_layer("lif_out", cfg, spike_grad, init_hidden=True, output=True),
        ).to(self.device)

        self.optimizer = build_optimizer(self.net.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="torch")

        # net[1] = lif1 (after conv1), net[4] = lif2 (after conv2)
        self.activity = ActivityMonitor({'lif1': self.net[1], 'lif2': self.net[4]})

    def synops_layer_map(self) -> dict:
        # lif1's spikes feed net[3] (conv2); lif2's spikes feed net[7] (Linear).
        return {'lif1': self.net[3], 'lif2': self.net[7]}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and collect output spikes."""
        self.activity.clear()
        spk_rec = []
        utils.reset(self.net)  # reset LIF hidden states between batches

        for step in range(data.size(0)):   # dim 0 = timesteps [T, B, C, H, W]
            spk_out, *_ = self.net(data[step])  # Alpha returns (spk, syn, mem)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

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


if __name__ == "__main__":
    from event_data_workflow.data_pipeline import main as load_data

    train_loader, test_loader = load_data()

    cfg       = Settings()
    model     = SNN_TORCH(cfg)
    trainer   = model.get_trainer(train_loader, test_loader)
    inference = model.get_inference(test_loader)

    print("\n Model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")
