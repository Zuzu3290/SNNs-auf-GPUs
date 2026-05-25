import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.event_data_workflow.data_pipeline import NeuromorphicEncoder
from learning.training import SNNTrainer


def _optimizer(params, cfg: Settings):
    lr, wd = cfg.LEARNING_RATE, cfg.WEIGHT_DECAY
    if cfg.OPTIMIZER_TYPE == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if cfg.OPTIMIZER_TYPE == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


def _loss(cfg: Settings):
    # SNNTorch forward returns [T, B, num_classes] — sum over T (dim 0) for rate decoding.
    # No fallback: unsupported loss_fn crashes loudly so misconfig is obvious.
    if cfg.LOSS_FN == "cross_entropy":
        return lambda spk_rec, targets: F.cross_entropy(spk_rec.sum(0), targets)
    if cfg.LOSS_FN == "mse_count":
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    raise NotImplementedError(
        f"loss_fn={cfg.LOSS_FN!r} not implemented for SNNTorch — use 'cross_entropy' or 'mse_count'"
    )


class SNN_TORCH(nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)

        # Surrogate is hardcoded across all 3 frameworks for fair comparison.
        # YAML 'surrogate' param is inactive — see FRAMEWORK_CONFIG_ANALYSIS.md §2.
        spike_grad  = surrogate.atan()
        print("[SNNTorch] Surrogate gradient is hardcoded to 'atan' (YAML 'surrogate' inactive)")
        beta        = cfg.BETA
        threshold   = cfg.THRESHOLD
        num_classes = cfg.NUM_CLASSES

        # Reset mode from YAML — SNNTorch supports both "zero" (hard) and "subtract" (soft).
        # No fallback: unknown reset_mode crashes loudly.
        if cfg.RESET_MODE not in ("zero", "subtract"):
            raise NotImplementedError(
                f"reset_mode={cfg.RESET_MODE!r} not supported for SNNTorch — use 'zero' or 'subtract'"
            )
        reset_mech = cfg.RESET_MODE
        print(f"[SNNTorch] Reset mode: '{reset_mech}' (from YAML architecture.reset_mode)")

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, reset_mechanism=reset_mech),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, reset_mechanism=reset_mech),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(cfg.FC_IN, num_classes),
            snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism=reset_mech),
        ).to(self.device)

        self.optimizer = _optimizer(self.net.parameters(), cfg)
        self.loss_fn   = _loss(cfg)

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
        return SNNTester(self, test_loader, self.cfg, self.device)


if __name__ == "__main__":
    cfg       = Settings()
    encoder   = NeuromorphicEncoder(cfg, framework="torch")
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_TORCH(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n✓ Model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {cfg.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")
