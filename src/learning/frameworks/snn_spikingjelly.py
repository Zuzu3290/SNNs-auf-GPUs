import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes

class SNN_SJ(ModelInterface, nn.Module):
    IN_CHANNELS:    int   = 2
    CONV1_OUT:      int   = 12
    CONV1_KERNEL:   int   = 5
    CONV2_OUT:      int   = 32
    CONV2_KERNEL:   int   = 5
    POOL_KERNEL:    int   = 2
    FC_IN:          int   = 32 * 5 * 5

    def __init__(self, cfg: Settings, spike_grad=surrogate.ATan()):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        tau = getattr(cfg, 'TAU', 2.0) 
        num_classes = cfg.NUM_CLASSES

        self.net = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT, self.CONV1_KERNEL),
            # neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            neuron.IzhikevichNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(self.POOL_KERNEL),

            nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL),
            # neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            neuron.IzhikevichNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(self.POOL_KERNEL),

            nn.Flatten(),
            nn.Linear(self.FC_IN, num_classes),
            # neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            neuron.IzhikevichNode(tau=tau, surrogate_function=spike_grad),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        
        # CHANGE 1: Use CrossEntropyLoss (standard for integer labels like 0-9)
        self.loss_fn = nn.CrossEntropyLoss()

        # net[1] = lif1 (after conv1), net[4] = lif2 (after conv2)
        register_activity_hooks(self, {'lif1': self.net[1], 'lif2': self.net[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and return the SUM of spikes."""
        clear_hidden_spikes(self)
        functional.reset_net(self.net)

        # data shape is [T, B, C, H, W]
        # We sum the output of each timestep directly into a total count
        sum_spikes = 0
        for step in range(data.size(0)):
            sum_spikes += self.net(data[step])

        # CHANGE 2: Return [B, 10] instead of [T, B, 10]
        # This matches what your Trainer expects
        return sum_spikes
    
    # ------------------------------------------------------------------
    # ModelInterface implementation
    # ------------------------------------------------------------------

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