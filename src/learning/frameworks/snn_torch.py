import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch
import torch.nn as nn
from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface
from learning.frameworks.activity_reg import register_activity_hooks, clear_hidden_spikes


class SNN_TORCH(ModelInterface, nn.Module):

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

        # snn.Alpha requires alpha > beta: alpha = membrane decay (slow), beta = synaptic decay (fast)
        alpha       = cfg.BETA                  # membrane decay  (e.g. 0.95)
        beta        = max(0.5, cfg.BETA - 0.1)  # synaptic decay, must be < alpha
        num_classes = cfg.NUM_CLASSES

        self.net = nn.Sequential(
            nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT, self.CONV1_KERNEL),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            snn.Alpha(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(self.POOL_KERNEL),
            nn.Conv2d(self.CONV1_OUT, self.CONV2_OUT, self.CONV2_KERNEL),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            snn.Alpha(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(self.POOL_KERNEL),
            nn.Flatten(),
            nn.Linear(self.FC_IN, num_classes),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
            snn.Alpha(alpha=alpha, beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.WEIGHT_DECAY,
        )
        self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

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
 
    print("\n\u2713 Model ready.")
    print(f"  - Device    : {model.device}")
    print(f"  - FC_IN     : {SNN_TORCH.FC_IN}  (verify matches your sensor resolution)")
    print(f"  - Classes   : {cfg.NUM_CLASSES}")



    """
Automatic Mixed Precision (AMP) keeps master weights in FP32 but runs the forward pass and loss computation in FP16. 
This halves memory bandwidth requirements and enables Tensor Core acceleration on modern NVIDIA GPUs, 
often yielding a 1.5–2x throughput improvement with negligible accuracy impact.

We use torch.autocast for the forward pass and torch.amp.GradScaler for loss scaling. 
A subtlety: we create the GradScaler with enabled=config.use_amp. 
When disabled, the scaler becomes a no-op — same code path, zero overhead, no branching.
"""