import torch
import torch.nn as nn

from skeleton.snn_config import Settings
from frameworks.model_interface import ModelInterface
from learning.utilities import ActivityMonitor, sum_over_time_cross_entropy

# Base LIF neuron parameters — hardcoded here rather than read from
# SNN_module.yaml's frameworks: block, since this is our own from-scratch
# baseline, not a config-selectable third-party backend like the other
# frameworks in this package.
LIF_BETA:      float = 0.9   # membrane decay per timestep
LIF_THRESHOLD: float = 0.5   # spike threshold


class SurrogateSpike(torch.autograd.Function):
    """Heaviside step forward (spike / no spike); fast-sigmoid surrogate
    gradient backward, since the true Heaviside derivative is zero almost
    everywhere and would kill all learning signal."""

    @staticmethod
    def forward(ctx, mem: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(mem)
        ctx.threshold = threshold
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        mem, = ctx.saved_tensors
        sigmoid = torch.sigmoid(mem - ctx.threshold)
        return grad_output * sigmoid * (1 - sigmoid), None


class LIFCell(nn.Module):
    """Leaky Integrate-and-Fire neuron. Hard reset to 0 on spike, matching
    the reset convention already used by every other framework in this
    package (see snn_torch.py's build_lif_layer)."""

    def __init__(self, beta: float = LIF_BETA, threshold: float = LIF_THRESHOLD):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.mem = None

    def reset_state(self) -> None:
        self.mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        self.mem = self.beta * self.mem + x
        spk = SurrogateSpike.apply(self.mem, self.threshold)
        self.mem = self.mem * (1.0 - spk)
        return spk


class SNN_LIF(ModelInterface, nn.Module):
    """Base LIF SNN — our own from-scratch implementation. No external SNN
    library dependency (unlike SNN_TORCH/SNN_NORSE/SNN_SJ/SNN_SINABS)."""

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        self.conv1   = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.lif1    = LIFCell()
        self.pool1   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.conv2   = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.lif2    = LIFCell()
        self.pool2   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES)
        self.lif_out = LIFCell()

        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        self.loss_fn   = sum_over_time_cross_entropy

        self.activity = ActivityMonitor({'lif1': self.lif1, 'lif2': self.lif2})

    def synops_layer_map(self) -> dict:
        # lif1's spikes feed conv2; lif2's spikes feed fc.
        return {'lif1': self.conv2, 'lif2': self.fc}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]
        returns: [T, B, num_classes]
        """
        self.activity.clear()
        self.lif1.reset_state()
        self.lif2.reset_state()
        self.lif_out.reset_state()

        spk_rec = []
        for step in range(data.size(0)):   # dim 0 = timesteps
            x = data[step]                 # [B, C, H, W]

            x = self.conv1(x)
            x = self.lif1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.lif2(x)
            x = self.pool2(x)

            x = self.flatten(x)
            x = self.fc(x)
            spk = self.lif_out(x)

            spk_rec.append(spk)

        return torch.stack(spk_rec)        # [T, B, num_classes]

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

    encoder = NeuromorphicEncoder(Settings())
    train_loader, test_loader = encoder.get_dataloaders()

    cfg       = Settings()
    model     = SNN_LIF(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n Base LIF model ready.")
    print(f"  - Device : {model.device}")
    print(f"  - FC_IN  : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
