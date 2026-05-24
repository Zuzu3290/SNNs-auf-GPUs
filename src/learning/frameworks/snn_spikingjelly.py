import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate
from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.event_data_workflow.data_pipeline import NeuromorphicEncoder
from learning.training import SNNTrainer


def _surrogate(name: str):
    if name == "fast_sigmoid":
        return surrogate.Sigmoid()
    return surrogate.ATan()


def _optimizer(params, cfg: Settings):
    lr, wd = cfg.LEARNING_RATE, cfg.WEIGHT_DECAY
    if cfg.OPTIMIZER_TYPE == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if cfg.OPTIMIZER_TYPE == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


class SNN(nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        spike_grad  = _surrogate(cfg.SURROGATE)
        tau         = cfg.TAU
        num_classes = cfg.NUM_CLASSES

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),

            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),

            nn.Flatten(),
            nn.Linear(cfg.FC_IN, num_classes),
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
        ).to(self.device)

        self.optimizer = _optimizer(self.net.parameters(), cfg)
        self.loss_fn   = nn.CrossEntropyLoss()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and return the SUM of spikes."""
        functional.reset_net(self.net)

        # data shape is [T, B, C, H, W] — sum over timesteps → [B, num_classes]
        sum_spikes = 0
        for step in range(data.size(0)):
            sum_spikes += self.net(data[step])

        return sum_spikes

    def get_trainer(self, train_loader) -> SNNTrainer:
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader) -> SNNTester:
        return SNNTester(self.net, test_loader, self.cfg, self.device)
