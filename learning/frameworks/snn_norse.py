import torch
import torch.nn as nn
import norse.torch as norse

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.utilities import build_optimizer, build_loss, ActivityMonitor


def build_norse_layer(layer_name: str, cfg: Settings) -> nn.Module:
    """
    Build a Norse LIF neuron for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.norse):
      lif_cell       — norse.LIFCell  (standard LIF). Default.
      lif_rec_cell   — norse.LIFRecurrentCell  (adds recurrent self-connection).
    """
    neuron_type = cfg.NEURON_TYPES.get("norse", {}).get(layer_name, "lif_cell")
    fw_cfg      = cfg.FRAMEWORK_CFG["norse"]

    lif_params = norse.LIFParameters(
        tau_mem_inv = torch.as_tensor(fw_cfg["tau_mem_inv"], dtype=torch.float32),
        v_th        = torch.as_tensor(fw_cfg["threshold"],   dtype=torch.float32),
    )

    if neuron_type == "lif_rec_cell":
        raise NotImplementedError(
            f"lif_rec_cell for layer '{layer_name}' requires input_size and hidden_size. "
            "Subclass SNN_NORSE and override the layer construction for recurrent cells."
        )
    return norse.LIFCell(p=lif_params)


class SNN_NORSE(ModelInterface, nn.Module):

    def __init__(self, cfg: Settings):
        super().__init__()

        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)

        # tau_mem_inv is read directly from frameworks.norse in YAML.
        # Do NOT convert from SNNTorch beta: different coordinate systems.
        # SNNTorch: V_eq = 20*I  (20x amplification).  Norse: V_eq = I (no amplification).
        fw_cfg = {
            **cfg.FRAMEWORK_CFG["norse"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.conv1   = nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, cfg.CONV1_KERNEL)
        self.lif1    = build_norse_layer("lif1",    cfg)
        self.pool1   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.conv2   = nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, cfg.CONV2_KERNEL)
        self.lif2    = build_norse_layer("lif2",    cfg)
        self.pool2   = nn.MaxPool2d(cfg.POOL_KERNEL)

        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(cfg.FC_IN, cfg.NUM_CLASSES)
        self.lif_out = build_norse_layer("lif_out", cfg)

        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), fw_cfg)
        self.loss_fn   = build_loss(fw_cfg, framework="norse")

        self.activity = ActivityMonitor({'lif1': self.lif1, 'lif2': self.lif2})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: [T, B, C, H, W]
        returns: [T, B, num_classes]

        Unlike SNNTorch's Leaky (which hides state internally), norse.LIFCell returns
        (spikes, new_state) and expects the previous state as input. States are
        initialised to None on the first timestep; Norse auto-creates zero tensors.
        """
        self.activity.clear()
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

    print("\n Norse model ready.")
    print(f"  - Device : {model.device}")
    print(f"  - FC_IN  : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes: {cfg.NUM_CLASSES}")
