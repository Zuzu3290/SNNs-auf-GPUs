import functools

import torch
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import spyx.nn as snn

from skeleton.snn_config import Settings
from learning.frameworks.model_interface import ModelInterface

# hk.max_pool needs a window/stride tuple with the same rank as the tensor —
# (batch, height, width, channel) — not just the 2D pooling extent.
_POOL_RANK4 = lambda k: (1, k, k, 1)


def _build_spyx_layer(layer_name: str, shape: tuple, cfg: Settings):
    """
    Build a Spyx (Haiku RNNCore) spiking neuron for the given layer slot.

    Neuron types (set per layer in network_architecture.yaml → neuron_types.spyx):
      lif  — spyx.nn.LIF   (single-compartment leak). Default.
      alif — spyx.nn.ALIF  (adaptive threshold — gamma controls adaptation rate).
    """
    neuron_type = cfg.NEURON_TYPES.get("spyx", {}).get(layer_name, "lif")
    fw_cfg      = cfg.FRAMEWORK_CFG["spyx"]

    if neuron_type == "alif":
        return snn.ALIF(shape, beta=fw_cfg["beta"], gamma=fw_cfg["gamma"], threshold=fw_cfg["threshold"])
    return snn.LIF(shape, beta=fw_cfg["beta"], threshold=fw_cfg["threshold"])


def _build_optax_optimizer(fw_cfg: dict):
    """optax equivalent of learning.build_optimizer — same names, JAX backend."""
    lr  = fw_cfg["learning_rate"]
    wd  = fw_cfg["weight_decay"]
    opt = fw_cfg.get("optimizer", "adam").lower()

    if opt == "adamw":
        return optax.adamw(lr, weight_decay=wd)
    if opt == "sgd":
        return optax.sgd(lr, momentum=0.9)
    return optax.adam(lr)


class _SpykingCore(hk.RNNCore):
    """One conv1→lif1→pool1→conv2→lif2→pool2→fc→lif_out timestep, threaded through
    hk.dynamic_unroll. Mirrors the architecture used by the other 3 backends —
    Conv2d/MaxPool2d params come from network_architecture.yaml via cfg."""

    def __init__(self, cfg: Settings, name="SpykingCore"):
        super().__init__(name=name)
        h1  = cfg.SENSOR_H - cfg.CONV1_KERNEL + 1
        hp1 = h1 // cfg.POOL_KERNEL
        h2  = hp1 - cfg.CONV2_KERNEL + 1
        hp2 = h2 // cfg.POOL_KERNEL
        self._pool_k = cfg.POOL_KERNEL

        self.conv1   = hk.Conv2D(cfg.CONV1_OUT, cfg.CONV1_KERNEL, padding="VALID", data_format="NHWC")
        self.lif1    = _build_spyx_layer("lif1", (h1, h1, cfg.CONV1_OUT), cfg)
        self.conv2   = hk.Conv2D(cfg.CONV2_OUT, cfg.CONV2_KERNEL, padding="VALID", data_format="NHWC")
        self.lif2    = _build_spyx_layer("lif2", (h2, h2, cfg.CONV2_OUT), cfg)
        self.fc      = hk.Linear(cfg.NUM_CLASSES)
        self.lif_out = _build_spyx_layer("lif_out", (cfg.NUM_CLASSES,), cfg)

    def __call__(self, x, state):
        v1, v2, v3 = state

        x = self.conv1(x)
        spk1, v1 = self.lif1(x, v1)
        x = hk.max_pool(spk1, window_shape=_POOL_RANK4(self._pool_k), strides=_POOL_RANK4(self._pool_k), padding="VALID")

        x = self.conv2(x)
        spk2, v2 = self.lif2(x, v2)
        x = hk.max_pool(spk2, window_shape=_POOL_RANK4(self._pool_k), strides=_POOL_RANK4(self._pool_k), padding="VALID")

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        spk_out, v3 = self.lif_out(x, v3)

        return spk_out, (v1, v2, v3)

    def initial_state(self, batch_size):
        return (
            self.lif1.initial_state(batch_size),
            self.lif2.initial_state(batch_size),
            self.lif_out.initial_state(batch_size),
        )


def _net_fn(cfg: Settings, x):
    """Single SpykingCore instance, unrolled over the time axis (x is [T, B, H, W, C])."""
    core  = _SpykingCore(cfg)
    state = core.initial_state(x.shape[1])
    spikes, _ = hk.dynamic_unroll(core, x, state, time_major=True)
    return spikes


class SNN_SPYX(ModelInterface):
    """
    Spyx is JAX/Haiku, not PyTorch — the only backend in this project that isn't.
    Two consequences follow directly from that:

    1. PyTorch tensors cross the boundary via DLPack (zero-copy on a shared CUDA
       device; falls back to a CPU copy here since JAX has no Windows CUDA wheels —
       `jax.devices()` reports CpuDevice-only on this machine. Run under the
       `container` branch's Linux Docker image with `pip install jax[cuda12]` for
       real GPU execution).
    2. PyTorch autograd cannot trace through JAX/XLA. backward_pass() is a no-op —
       gradients are computed with jax.value_and_grad and applied via optax inside
       loss_fn(), the one place that has access to both the model output and the
       batch's targets (ModelInterface.forward() only receives `data`).
       See docs/frameworks/README.md → "JAX backend" for the full pattern.

    Known limitation: the optax update fires on every micro-batch, so
    GRAD_ACCUM_STEPS > 1 is not honoured for this backend (unlike the 3 PyTorch
    backends, which accumulate via backward_pass(do_step=False)).
    """

    def __init__(self, cfg: Settings):
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)
        self._is_training = True

        fw_cfg = {
            **cfg.FRAMEWORK_CFG["spyx"],
            "learning_rate": cfg.LEARNING_RATE,
            "weight_decay":  cfg.WEIGHT_DECAY,
        }

        self.rng = jax.random.PRNGKey(0)
        self.net = hk.transform(functools.partial(_net_fn, cfg))

        dummy = jnp.zeros((cfg.TIMESTEPS, cfg.BATCH_SIZE, cfg.SENSOR_H, cfg.SENSOR_W, cfg.IN_CHANNELS))
        self.params = self.net.init(self.rng, dummy)

        self.optimizer = _build_optax_optimizer(fw_cfg)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_fn   = self._jax_loss_and_update

        self._last_jax_input = None

    def tensor_format(self) -> str:
        return "TB"

    def is_differentiable(self) -> bool:
        return False

    def _to_jax_nhwc(self, data: torch.Tensor):
        """[T, B, C, H, W] torch → [T, B, H, W, C] jax, via DLPack."""
        x = data.permute(0, 1, 3, 4, 2).contiguous()
        if jax.default_backend() != "gpu":
            x = x.cpu()
        return jax.dlpack.from_dlpack(x)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x = self._to_jax_nhwc(data)
        self._last_jax_input = x
        spk = self.net.apply(self.params, self.rng, x)   # [T, B, num_classes]
        return torch.from_dlpack(spk).to(self.device)

    def _jax_loss_and_update(self, spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Recomputes the forward pass fused with its gradient (jax.value_and_grad
        cannot reuse spk_rec — that tensor already crossed back into PyTorch).
        Applies the optax update as a side effect when self._is_training is True,
        then returns a plain scalar tensor so the trainer's loss.item() logging
        keeps working — backward_pass() never actually calls .backward() on it.
        """
        x = self._last_jax_input
        y = jnp.asarray(targets.detach().cpu().numpy())

        def loss_fn(params):
            spk = self.net.apply(params, self.rng, x)
            logits = spk.sum(axis=0)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        loss_val, grads = jax.value_and_grad(loss_fn)(self.params)

        if self._is_training:
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)

        return torch.tensor(float(loss_val), device=self.device)

    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        pass  # weights already updated inside _jax_loss_and_update()

    def zero_grad(self) -> None:
        pass  # optax carries its own state; nothing to zero between steps

    def train_mode(self) -> None:
        self._is_training = True

    def eval_mode(self) -> None:
        self._is_training = False

    def get_lr(self) -> float:
        return self.cfg.LEARNING_RATE

    def get_state(self) -> dict:
        to_numpy = lambda tree: jax.tree_util.tree_map(jax.device_get, tree)
        return {
            "model_state_dict":     to_numpy(self.params),
            "optimizer_state_dict": to_numpy(self.opt_state),
        }


if __name__ == "__main__":
    from event_data_workflow import NeuromorphicEncoder

    cfg     = Settings()
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()

    model     = SNN_SPYX(cfg)
    trainer   = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)

    print("\n Spyx model ready.")
    print(f"  - JAX devices : {jax.devices()}")
    print(f"  - FC_IN       : {cfg.FC_IN}  (auto-computed from network_architecture.yaml)")
    print(f"  - Classes     : {cfg.NUM_CLASSES}")
