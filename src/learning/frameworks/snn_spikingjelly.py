import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate
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
    # SpikingJelly forward returns [B, num_classes] (already summed over T).
    # No fallback: unsupported loss_fn crashes loudly so misconfig is obvious.
    if cfg.LOSS_FN == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise NotImplementedError(
        f"loss_fn={cfg.LOSS_FN!r} not implemented for SpikingJelly — use 'cross_entropy'"
    )


class SNN(nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.DEVICE)
        # Surrogate is hardcoded across all 3 frameworks for fair comparison.
        # YAML 'surrogate' param is inactive — see FRAMEWORK_CONFIG_ANALYSIS.md §2.
        spike_grad  = surrogate.ATan()
        print("[SpikingJelly] Surrogate gradient is hardcoded to 'ATan' (YAML 'surrogate' inactive)")
        tau         = cfg.TAU
        num_classes = cfg.NUM_CLASSES

        # Reset mode from YAML — SpikingJelly: v_reset=0.0 → hard reset; v_reset=None → soft reset (subtract).
        # No fallback: unknown reset_mode crashes loudly.
        if cfg.RESET_MODE == "zero":
            v_reset = 0.0
        elif cfg.RESET_MODE == "subtract":
            v_reset = None
        else:
            raise NotImplementedError(
                f"reset_mode={cfg.RESET_MODE!r} not supported for SpikingJelly — use 'zero' or 'subtract'"
            )
        print(f"[SpikingJelly] Reset mode: '{cfg.RESET_MODE}' (v_reset={v_reset})")

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),

            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=spike_grad),
            nn.MaxPool2d(cfg.POOL_KERNEL),

            nn.Flatten(),
            nn.Linear(cfg.FC_IN, num_classes),
            neuron.LIFNode(tau=tau, v_reset=v_reset, surrogate_function=spike_grad),
        ).to(self.device)

        self.optimizer = _optimizer(self.net.parameters(), cfg)
        self.loss_fn   = _loss(cfg)

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
