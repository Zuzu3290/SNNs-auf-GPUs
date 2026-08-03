import torch
import torch.nn as nn
import sinabs.layers as sl

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import clear_hidden_spikes
from learning.utilities import build_optimizer, build_loss


def build_sinabs_layer(layer_name: str, cfg: Settings, **kwargs) -> nn.Module:
    """
    Build a Sinabs spiking neuron for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.sinabs):
      lif — sinabs.layers.LIF  (exponential membrane leak). Default.
      iaf — sinabs.layers.IAF  (no leak — integrate-and-fire).
    """
    neuron_type = cfg.NEURON_TYPES.get("sinabs", {}).get(layer_name, "lif")
    fw_cfg      = cfg.FRAMEWORK_CFG["sinabs"]
    threshold   = torch.as_tensor(fw_cfg["threshold"])

    if neuron_type == "iaf":
        return sl.IAF(spike_threshold=threshold, **kwargs)
    return sl.LIF(tau_mem=fw_cfg["tau_mem"], spike_threshold=threshold, **kwargs)


class SNN_SINABS(ModelInterface, nn.Module):
    """
    Sinabs is DVS-first: every spiking layer is batch-first (B, T, ...) rather
    than the time-first (T, B, ...) layout SNNTorch/Norse/SpikingJelly use, and
    it ships a real export path to SynSense's Speck chip — relevant for an
    automotive edge-deployment story built on the same event-camera data this
    pipeline already produces.
    """

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["sinabs"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.flatten_t = sl.FlattenTime()
        self.conv1     = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.pool1     = nn.MaxPool2d(cfg.POOL_KERNEL)
        self.conv2     = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.pool2     = nn.MaxPool2d(cfg.POOL_KERNEL)
        self.flat      = nn.Flatten()
        self.fc        = nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES)

        self.lif1    = build_sinabs_layer("lif1",    cfg)
        self.lif2    = build_sinabs_layer("lif2",    cfg)
        self.lif_out = build_sinabs_layer("lif_out", cfg)

        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="sinabs")

        # No register_activity_hooks() here: activity_reg.py's hooks assume the
        # hooked layer is called once PER TIMESTEP (true for Norse/SNNTorch/
        # SpikingJelly's per-timestep loop). Sinabs' LIF layers are called once
        # per forward() with the whole (B,T,...) tensor — hooking them would
        # record a single T=1 "timestep" containing all T internally, which
        # silently breaks stdp_regularization's per-timestep trace math
        # (confirmed: `ema_kernel (1x1) @ post_t` size mismatch).

    def tensor_format(self) -> str:
        """Sinabs LIF/IAF layers expect (Batch, Time, ...) — the trainer transposes for us."""
        return "BT"

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [B, T, C, H, W]
        returns: [T, B, num_classes]  — transposed back to time-first so
                 aggregate_spike_output() and the STDP regulariser (which expect
                 [T, B, ...]) work unmodified.

        Conv2d/MaxPool2d only understand 4D (N, C, H, W), so the batch and time
        dims are flattened before each conv stage and unflattened before each
        LIF stage, which is the layout Sinabs' stateful layers require.
        """
        clear_hidden_spikes(self)
        for layer in (self.lif1, self.lif2, self.lif_out):
            layer.reset_states()

        B, T = data.size(0), data.size(1)

        x = self.flatten_t(data)           # [B*T, C, H, W]
        x = self.conv1(x)
        x = x.unflatten(0, (B, T))         # [B, T, C', H', W'] — LIF needs batch-first
        x = self.lif1(x)
        x = x.flatten(0, 1)                # [B*T, ...] for the next conv/pool stage
        x = self.pool1(x)

        x = self.conv2(x)
        x = x.unflatten(0, (B, T))
        x = self.lif2(x)
        x = x.flatten(0, 1)
        x = self.pool2(x)

        x = self.flat(x)                   # [B*T, FC_IN]
        x = self.fc(x)                     # [B*T, num_classes]
        x = x.unflatten(0, (B, T))         # [B, T, num_classes]
        x = self.lif_out(x)

        return x.transpose(0, 1)           # [T, B, num_classes]

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

    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_SINABS(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n Sinabs model ready.")
    print(f"  - Device : {model.device}")
    print(f"  - FC_IN  : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
