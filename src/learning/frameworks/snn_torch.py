import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch
import torch.nn as nn
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes
from learning.utilities import build_optimizer, build_loss, build_lif_layer


class SNN_TORCH(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings, spike_grad=surrogate.atan()):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["snntorch"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.net = nn.Sequential(
            nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            build_lif_layer("lif1",    cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL),
            build_lif_layer("lif2",    cfg, spike_grad, init_hidden=True),
            nn.MaxPool2d(cfg.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES),
            build_lif_layer("lif_out", cfg, spike_grad, init_hidden=True, output=True),
        ).to(self.device)

        self.optimizer = build_optimizer(self.net.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="torch")

        # net[1] = lif1 (after conv1), net[4] = lif2 (after conv2)
        register_activity_hooks(self, {'lif1': self.net[1], 'lif2': self.net[4]})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Iterate over timesteps and collect output spikes."""
        clear_hidden_spikes(self)
        spk_rec = []
        utils.reset(self.net)  # reset LIF hidden states between batches

        for step in range(data.size(0)):   # dim 0 = timesteps [T, B, C, H, W]
            spk_out, *_ = self.net(data[step])  # Alpha returns (spk, syn, mem)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)

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


if __name__ == "__main__":
    from event_data_workflow.data_pipeline import main as load_data

    train_loader, test_loader = load_data()

    cfg       = Settings()
    model     = SNN_TORCH(cfg)
    trainer   = model.get_trainer(train_loader, test_loader)
    inference = model.get_inference(test_loader)

    print("\n Model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")
