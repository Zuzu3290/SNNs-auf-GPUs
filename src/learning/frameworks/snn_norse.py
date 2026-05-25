import norse.torch as norse
import torch
import torch.nn as nn
import torch.nn.functional as F

from skeleton.snn_config import Settings
from learning.inference import SNNTester
from learning.training import SNNTrainer


def _optimizer(params, cfg: Settings):
    lr, wd = cfg.LEARNING_RATE, cfg.WEIGHT_DECAY
    if cfg.OPTIMIZER_TYPE == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if cfg.OPTIMIZER_TYPE == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


def _loss(cfg: Settings):
    # Norse forward returns [T, B, num_classes] — sum over T (dim 0) for rate decoding.
    # No fallback: unsupported loss_fn crashes loudly so misconfig is obvious.
    if cfg.LOSS_FN == "cross_entropy":
        return lambda spk_rec, targets: F.cross_entropy(spk_rec.float().sum(0), targets)
    raise NotImplementedError(
        f"loss_fn={cfg.LOSS_FN!r} not implemented for Norse — use 'cross_entropy'"
    )


class _NorseNet(nn.Module):
    """
    All Norse layers in one module — mirrors the role of nn.Sequential in SNN_TORCH.

    Unlike SNNTorch's Leaky (which hides state internally), norse.LIFCell returns
    (spikes, new_state) and expects the previous state as input. States are initialised
    to None on the first timestep; Norse auto-creates zero tensors from that.
    """

    def __init__(self, cfg: Settings, lif_params: norse.LIFParameters):
        super().__init__()

        self.conv1   = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.lif1    = norse.LIFCell(p=lif_params)
        self.pool1   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.conv2   = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.lif2    = norse.LIFCell(p=lif_params)
        self.pool2   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES)
        self.lif_out = norse.LIFCell(p=lif_params)

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

        # Reset mode from YAML — Norse only supports "zero" (hard reset, V→v_reset=0).
        # Soft reset is a library limitation — crash loudly if requested.
        if cfg.RESET_MODE == "subtract":
            raise NotImplementedError(
                "reset_mode='subtract' not supported for Norse — LIFCell only does hard reset. Use 'zero'."
            )
        if cfg.RESET_MODE != "zero":
            raise NotImplementedError(
                f"reset_mode={cfg.RESET_MODE!r} not supported for Norse — use 'zero'"
            )
        print("[Norse] Reset mode: 'zero' (hard reset to v_reset=0.0; Norse library limitation)")

        # All params come from SNN_module.yaml — no hardcoded fallbacks, no conversions.
        # tau_mem_inv is under frameworks.norse; threshold is shared across frameworks.
        lif_params = norse.LIFParameters(
            tau_mem_inv=torch.as_tensor(cfg.TAU_MEM_INV),
            v_th=torch.as_tensor(cfg.THRESHOLD),
            v_reset=torch.as_tensor(0.0),
        )

        # Surrogate is hardcoded across all 3 frameworks for fair comparison.
        # Norse uses its library default 'SuperSpike' (we don't pass method= to LIFParameters).
        # YAML 'surrogate' param is inactive — see FRAMEWORK_CONFIG_ANALYSIS.md §2.
        print("[Norse] Surrogate gradient is hardcoded to 'SuperSpike' (Norse default; YAML 'surrogate' inactive)")

        self.net = _NorseNet(cfg, lif_params).to(self.device)

        self.optimizer = _optimizer(self.net.parameters(), cfg)
        self.loss_fn   = _loss(cfg)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)

    def get_trainer(self, train_loader) -> SNNTrainer:
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader) -> SNNTester:
        return SNNTester(self.net, test_loader, self.cfg, self.device)


if __name__ == "__main__":
    from learning.event_data_workflow.data_pipeline import NeuromorphicEncoder

    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg, framework="norse")
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_NORSE(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n✓ Norse model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {cfg.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")
