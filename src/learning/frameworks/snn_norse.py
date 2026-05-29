import math

import norse.torch as norse
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes


class SNN_NORSE(ModelInterface, nn.Module):

    IN_CHANNELS:  int = 2       # polarity channels from DVS event camera
    CONV1_OUT:    int = 12      # first conv output channels
    CONV1_KERNEL: int = 5       # first conv kernel size
    CONV2_OUT:    int = 32      # second conv output channels
    CONV2_KERNEL: int = 5       # second conv kernel size
    POOL_KERNEL:  int = 2       # maxpool kernel (applied twice)
    FC_IN:        int = 32 * 5 * 5  # flattened size after both conv+pool stages

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        # Convert beta (SNNTorch convention, 0–1) to tau_mem_inv (Norse convention, Hz).
        # tau_mem_inv = -ln(beta) / dt  where dt=0.001s gives ~51 Hz for beta=0.95.
        # TODO: add a norse-specific section to SNN_module.yaml so tau_mem_inv can be
        # tuned independently from the SNNTorch beta value.
        dt          = 0.001
        tau_mem_inv = -math.log(cfg.BETA) / dt
        lif_params  = norse.LIFParameters(
            tau_mem_inv = torch.as_tensor(tau_mem_inv, dtype=torch.float32),
            v_th        = torch.as_tensor(cfg.THRESHOLD, dtype=torch.float32),
        )

        self.conv1   = nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT, self.CONV1_KERNEL)
        self.lif1    = norse.LIFCell(p=lif_params)
        self.pool1   = nn.MaxPool2d(self.POOL_KERNEL)

        self.conv2   = nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL)
        self.lif2    = norse.LIFCell(p=lif_params)
        self.pool2   = nn.MaxPool2d(self.POOL_KERNEL)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(self.FC_IN, cfg.NUM_CLASSES)
        self.lif_out = norse.LIFCell(p=lif_params)

        self.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        # CrossEntropy on spike counts — sums spikes over T, then maximises correct class logit.
        # Simpler gradient signal than mse_count_loss; typically converges faster for classification.
        self.loss_fn = lambda spk_rec, targets: F.cross_entropy(spk_rec.sum(0), targets)

        # lif1 and lif2 are the hidden LIFCell layers
        register_activity_hooks(self, {'lif1': self.lif1, 'lif2': self.lif2})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]
        returns: [T, B, num_classes]

        Unlike SNNTorch's Leaky (which hides state internally), norse.LIFCell returns
        (spikes, new_state) and expects the previous state as input. States are
        initialised to None on the first timestep; Norse auto-creates zero tensors.
        """
        clear_hidden_spikes(self)
        s1 = s2 = s_out = None
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
    model     = SNN_NORSE(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n✓ Norse model ready.")
    print(f"  - Device : {model.device}")
    print(f"  - FC_IN  : {SNN_NORSE.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
