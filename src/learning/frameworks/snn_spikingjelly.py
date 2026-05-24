import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate
from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.event_data_workflow.data_pipeline import NeuromorphicEncoder
from learning.training import SNNTrainer

class SNN_SJ(nn.Module):
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
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(self.POOL_KERNEL),
            
            nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL),
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
            nn.MaxPool2d(self.POOL_KERNEL),
            
            nn.Flatten(),
            nn.Linear(self.FC_IN, num_classes),
            neuron.LIFNode(tau=tau, surrogate_function=spike_grad),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        
        # CHANGE 1: Use CrossEntropyLoss (standard for integer labels like 0-9)
        self.loss_fn = nn.CrossEntropyLoss() 
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and return the SUM of spikes."""
        functional.reset_net(self.net)
        
        # data shape is [T, B, C, H, W]
        # We sum the output of each timestep directly into a total count
        sum_spikes = 0
        for step in range(data.size(0)):
            sum_spikes += self.net(data[step])

        # CHANGE 2: Return [B, 10] instead of [T, B, 10]
        # This matches what your Trainer expects
        return sum_spikes
    
    def get_trainer(self, train_loader) -> SNNTrainer:
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader) -> SNNTester:
        return SNNTester(self.net, test_loader, self.cfg, self.device)