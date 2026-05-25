# import math  # was used for LIFParameters tau_mem_inv conversion

import norse.torch as norse
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.training import SNNTrainer


class _NorseNet(nn.Module):
    """
    All Norse layers in one module — mirrors the role of nn.Sequential in SNN_TORCH.

    Unlike SNNTorch's Leaky (which hides state internally), norse.LIFCell returns
    (spikes, new_state) and expects the previous state as input. States are initialised
    to None on the first timestep; Norse auto-creates zero tensors from that.
    """

    IN_CHANNELS:  int = 2
    CONV1_OUT:    int = 12
    CONV1_KERNEL: int = 5
    CONV2_OUT:    int = 32
    CONV2_KERNEL: int = 5
    POOL_KERNEL:  int = 2
    FC_IN:        int = 32 * 5 * 5

    # def __init__(self, lif_params: norse.LIFParameters, num_classes: int):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1   = nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT, self.CONV1_KERNEL)
        # self.lif1  = norse.LIFCell(p=lif_params)
        self.lif1    = norse.IzhikevichCell(spiking_method=norse.tonic_spiking)
        self.pool1   = nn.MaxPool2d(self.POOL_KERNEL)

        self.conv2   = nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL)
        # self.lif2  = norse.LIFCell(p=lif_params)
        self.lif2    = norse.IzhikevichCell(spiking_method=norse.tonic_spiking)
        self.pool2   = nn.MaxPool2d(self.POOL_KERNEL)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(self.FC_IN, num_classes)
        # self.lif_out = norse.LIFCell(p=lif_params)
        self.lif_out = norse.IzhikevichCell(spiking_method=norse.tonic_spiking)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]
        returns: [T, B, num_classes]
        """
        s1 = s2 = s_out = None  # None → Norse auto-initialises to zeros on first call
        spk_rec = []

        for step in range(data.size(0)):   # dim 0 = timesteps
            x = data[step]                 # [B, C, H, W]

            x = self.conv1(x)
            x, s1 = self.lif1(x, s1)
            x = self.pool1(x)

            x = self.conv2(x)
            x, s2 = self.lif2(x, s2)
            x = self.pool2(x)

            x = self.flatten(x)
            x = self.fc(x)
            spk, s_out = self.lif_out(x, s_out)

            spk_rec.append(spk)

        return torch.stack(spk_rec)        # [T, B, num_classes]


class SNN_NORSE(nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        # --- Previous: LIFParameters for LIFCell ---
        # TODO: temporary parameter conversion — replace with framework-specific config sections.
        # Norse and SNNTorch use different coordinate systems for the same physical quantity:
        #   SNNTorch: beta (dimensionless, 0–1)     Norse: tau_mem_inv (Hz, typically 10–1000)
        # The conversion is nonlinear, so a single shared YAML value cannot fairly control both.
        # Proper fix: add snntorch/norse sub-sections to SNN_module.yaml and expose them via
        # Settings, so each framework reads its own independently tuned parameters.
        # dt          = 0.001
        # tau_mem_inv = -math.log(cfg.BETA) / dt
        # lif_params = norse.LIFParameters(
        #     tau_mem_inv=torch.as_tensor(tau_mem_inv),
        #     v_th=torch.as_tensor(cfg.THRESHOLD),
        # )
        # self.net = _NorseNet(lif_params, cfg.NUM_CLASSES).to(self.device)

        # --- Current: IzhikevichCell with tonic_spiking behavior ---
        self.net = _NorseNet(cfg.NUM_CLASSES).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        # CrossEntropy on spike counts — sums spikes over T, then maximises correct class logit.
        # Simpler gradient signal than mse_count_loss; typically converges faster for classification.
        self.loss_fn = lambda spk_rec, targets: F.cross_entropy(spk_rec.sum(0), targets)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def get_trainer(self, train_loader) -> SNNTrainer:
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader) -> SNNTester:
        return SNNTester(self.net, test_loader, self.cfg, self.device)

    def get_adversarial_evaluator(self, test_loader):
        from learning.adversarial_robustness import AdversarialEvaluator
        return AdversarialEvaluator(self, test_loader, self.cfg, self.device)


if __name__ == "__main__":
    from learning.event_data_workflow import NeuromorphicEncoder

    encoder = NeuromorphicEncoder(Settings())
    train_loader, test_loader = encoder.get_dataloaders()

    cfg       = Settings()
    model     = SNN_NORSE(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n✓ Norse model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {_NorseNet.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")
