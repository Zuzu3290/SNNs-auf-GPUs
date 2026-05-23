import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn.functional as F

import torch
import torch.nn as nn

from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.data_pipeline import NeuromorphicEncoder
from learning.training import SNNTrainer


class SNN_TORCH(nn.Module):

    IN_CHANNELS:    int   = 2       # polarity channels from event camera (C in sensor_size)
    CONV1_OUT:      int   = 12      # first conv output channels
    CONV1_KERNEL:   int   = 5       # first conv kernel size
    CONV2_OUT:      int   = 32      # second conv output channels
    CONV2_KERNEL:   int   = 5       # second conv kernel size
    POOL_KERNEL:    int   = 2       # maxpool kernel (applied twice)
    FC_IN:          int   = 32 * 5 * 5   # flattened size after both conv+pool stages

                                # neuron and simulation parameters
    def __init__(self, cfg: Settings, spike_grad=surrogate.atan()):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)

        beta        = cfg.BETA
        num_classes = cfg.NUM_CLASSES

        self.net = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT, self.CONV1_KERNEL),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(self.POOL_KERNEL),
            nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(self.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(self.FC_IN, num_classes),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        # cross_entropy on summed spikes — framework-agnostic, matches Norse loss.
        # Replaced SF.mse_count_loss (SNNTorch-specific) for fair framework comparison.
        self.loss_fn = lambda spk_rec, targets: F.cross_entropy(spk_rec.sum(0), targets)
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and collect output spikes."""
        spk_rec = []
        utils.reset(self.net)  # reset LIF hidden states between batches
 
        for step in range(data.size(0)):   # dim 0 = timesteps [T, B, C, H, W]
            spk_out, _ = self.net(data[step])
            spk_rec.append(spk_out)
 
        return torch.stack(spk_rec)
    
    def get_trainer(self, train_loader) -> SNNTrainer:
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader) -> SNNTester:
        return SNNTester(self.net, test_loader, self.cfg, self.device)



if __name__ == "__main__":
    from learning.data_pipeline import main as load_data
 
    train_loader, test_loader = load_data()
 
    cfg       = Settings()
    model     = SNN_TORCH(cfg)
    trainer   = model.get_trainer(train_loader, test_loader)
    inference = model.get_inference(test_loader)
 
    print("\n\u2713 Model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {SNN_TORCH.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")