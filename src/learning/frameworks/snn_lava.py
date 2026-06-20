"""
UNVERIFIED — written from memory of the lava-dl/SLAYER public API, never run.

`pip install lava-dl` pins `torch<2.4.0,>=2.3.1`; this project runs torch 2.10.0+cu128.
Installing it in this environment silently downgraded torch to a CPU-only 2.3.1 build
and broke CUDA for every other framework — see docs/frameworks/additional_frameworks.md
for the incident writeup. lava-dl needs its own isolated environment (a separate venv/
conda env, or its own stage in the `container` branch's Docker image) — never installed
alongside the rest of this project's dependencies.

Treat every API name below (slayer.block.cuba.*, neuron_params keys, tensor layout) as a
best-effort sketch to verify against the real package once it's installed in that
isolated environment, not as confirmed-working code.
"""

import torch
import torch.nn as nn

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.utilities import build_optimizer, build_loss

try:
    import lava.lib.dl.slayer as slayer
except ImportError as e:
    raise ImportError(
        "lava-dl is not installed (and likely can't be, in this environment — see "
        "the module docstring). Install it in an isolated env/container to use SNN_LAVA."
    ) from e


def build_lava_neuron_params(layer_name: str, cfg: Settings) -> dict:
    """
    CUBA (CUrrent-BAsed LIF) neuron params for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.lava):
      cuba — slayer.block.cuba.*  (current + voltage decay, SLAYER's standard neuron). Default.
    """
    fw_cfg = cfg.FRAMEWORK_CFG["lava"]
    return {
        "threshold":     fw_cfg["threshold"],
        "current_decay": fw_cfg["current_decay"],
        "voltage_decay": fw_cfg["voltage_decay"],
        "tau_grad":      fw_cfg["tau_grad"],
        "scale_grad":    fw_cfg["scale_grad"],
        "requires_grad": True,
    }


class SNN_LAVA(ModelInterface, nn.Module):
    """
    Unlike Sinabs/BindsNET/Spyx, SLAYER blocks ARE standard PyTorch autograd —
    the spiking nonlinearity is a custom Function with a smoothed surrogate
    backward, so this backend trains exactly like Norse/SNNTorch/SpikingJelly
    (full backward_pass(), no JAX/no-op tricks needed). The genuinely different
    thing about it is the tensor layout: SLAYER convolves the synaptic current
    decay directly along the time axis, so it expects time LAST —
    (Batch, Channel, Height, Width, Time) — not first or second like every
    other backend in this project.
    """

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["lava"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        neuron_params = build_lava_neuron_params("lif1", cfg)

        self.blocks = nn.ModuleList([
            slayer.block.cuba.Conv(neuron_params, cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL, stride=1, padding=0, weight_norm=True),
            slayer.block.cuba.Pool(neuron_params, cfg.POOL_KERNEL, stride=cfg.POOL_KERNEL),
            slayer.block.cuba.Conv(neuron_params, cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL, stride=1, padding=0, weight_norm=True),
            slayer.block.cuba.Pool(neuron_params, cfg.POOL_KERNEL, stride=cfg.POOL_KERNEL),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params, cfg.FC_IN, cfg.NUM_CLASSES, weight_norm=True),
        ]).to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="lava")

    def tensor_format(self) -> str:
        """
        Default "TB" — the trainer hands us [T, B, C, H, W] unmodified. We do the
        (B, C, H, W, T) reshape ourselves inside forward() since SLAYER's time-last
        layout isn't one of the two conventions ModelInterface formalises.
        """
        return "TB"

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]  →  internally permuted to [B, C, H, W, T] for SLAYER.
        returns: [T, B, num_classes]
        """
        x = data.permute(1, 2, 3, 4, 0).contiguous()   # [B, C, H, W, T]

        for block in self.blocks:
            x = block(x)

        return x.permute(2, 0, 1).contiguous()         # [T, B, num_classes]

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


if __name__ == "__main__":
    from event_data_workflow import NeuromorphicEncoder

    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_LAVA(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n Lava-dl model ready (UNVERIFIED — run this block first after install).")
    print(f"  - Device : {model.device}")
    print(f"  - FC_IN  : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
