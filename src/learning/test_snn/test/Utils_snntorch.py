"""

Provides interface registries for:
  - Loss functions     (supervised & unsupervised)
  - Neuron models      (LIF variants available in snntorch)
  - Learning modes     (supervised | unsupervised)

These registries are consumed by the SNNLayer builder and the
training loop so that swapping any component is a one-line config change.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF


class LearningMode:
    """
    Sentinel strings that gate which training control-flow branch executes.
    Pass one of these constants to the training loop via config.
    """
    SUPERVISED   = "supervised"
    UNSUPERVISED = "unsupervised"

SURROGATE_REGISTRY = {
    # name           : callable that returns a surrogate gradient function
    "fast_sigmoid"   : surrogate.fast_sigmoid,
    "atan"           : surrogate.atan,
    "straight_through_estimator": surrogate.straight_through_estimator,
}

def get_surrogate(name: str = "fast_sigmoid"):
    """Return an instantiated surrogate gradient function by name."""
    if name not in SURROGATE_REGISTRY:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Available: {list(SURROGATE_REGISTRY.keys())}"
        )
    return SURROGATE_REGISTRY[name]()


# ──────────────────────────────────────────────────────────────────────────────
# NEURON MODEL REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
# Each entry is a factory function: (in_features, out_features, beta, cfg) -> snn neuron
# beta and threshold may be scalar or per-neuron tensors.

def _make_leaky(in_f, out_f, beta, cfg):
    """Standard Leaky Integrate-and-Fire neuron."""
    spike_grad = get_surrogate(cfg.get("surrogate", "fast_sigmoid"))
    return snn.Leaky(
        beta=beta,
        threshold=cfg.get("threshold", 1.0),
        learn_beta=cfg.get("learn_beta", True),
        learn_threshold=cfg.get("learn_threshold", False),
        spike_grad=spike_grad,
        reset_mechanism=cfg.get("reset_mechanism", "subtract"),
    )

def _make_leaky_integrator(in_f, out_f, beta, cfg):
    """
    Leaky integrator output neuron:  reset_mechanism='none'.
    Does NOT spike; outputs raw membrane potential.
    Useful as the final regression layer.
    """
    spike_grad = get_surrogate(cfg.get("surrogate", "fast_sigmoid"))
    return snn.Leaky(
        beta=beta,
        threshold=cfg.get("threshold", 1.0),
        learn_beta=cfg.get("learn_beta", True),
        spike_grad=spike_grad,
        reset_mechanism="none",
    )

def _make_synaptic(in_f, out_f, beta, cfg):
    """
    Synaptic (second-order) LIF with both alpha (synaptic) and beta (membrane) decay.
    Captures synaptic current dynamics in addition to membrane decay.
    """
    spike_grad = get_surrogate(cfg.get("surrogate", "fast_sigmoid"))
    alpha = cfg.get("alpha", 0.9)
    return snn.Synaptic(
        alpha=alpha,
        beta=beta,
        spike_grad=spike_grad,
        learn_alpha=cfg.get("learn_alpha", True),
        learn_beta=cfg.get("learn_beta", True),
    )

def _make_rleaky(in_f, out_f, beta, cfg):
    """
    Recurrent Leaky LIF:  output spikes feed back to the same layer.
    Suited for sequence / temporal-pattern tasks.
    """
    spike_grad = get_surrogate(cfg.get("surrogate", "fast_sigmoid"))
    return snn.RLeaky(
        beta=beta,
        spike_grad=spike_grad,
        learn_beta=cfg.get("learn_beta", True),
        all_to_all=cfg.get("all_to_all", True),
        linear_features=in_f,        # required when all_to_all=True
    )

def _make_lapicque(in_f, out_f, beta, cfg):
    """
    Lapicque (RC circuit) LIF — biophysically motivated, tunable R and C.
    """
    spike_grad = get_surrogate(cfg.get("surrogate", "fast_sigmoid"))
    return snn.Lapicque(
        beta=beta,
        spike_grad=spike_grad,
        learn_beta=cfg.get("learn_beta", True),
    )


NEURON_REGISTRY = {
    "leaky"            : _make_leaky,
    "leaky_integrator" : _make_leaky_integrator,
    "synaptic"         : _make_synaptic,
    "rleaky"           : _make_rleaky,
    "lapicque"         : _make_lapicque,
}

def get_neuron(name: str, in_features: int, out_features: int,
               beta=None, cfg: dict = None):
    """
    Instantiate a spiking neuron layer by registry name.

    Parameters
    ----------
    name         : key in NEURON_REGISTRY
    in_features  : fan-in (needed by recurrent models)
    out_features : fan-out / layer width
    beta         : decay rate — scalar float or per-neuron tensor.
                   If None, a random per-neuron tensor is created.
    cfg          : optional dict of extra keyword args (threshold, surrogate, …)
    """
    if name not in NEURON_REGISTRY:
        raise ValueError(
            f"Unknown neuron type '{name}'. "
            f"Available: {list(NEURON_REGISTRY.keys())}"
        )
    cfg = cfg or {}
    if beta is None:
        beta = torch.rand(out_features)
    return NEURON_REGISTRY[name](in_features, out_features, beta, cfg)


# ──────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION REGISTRY
# ──────────────────────────────────────────────────────────────────────────────
# Supervised losses operate on spike counts / membrane potentials against labels.
# Unsupervised losses operate without labels (STDP trace-based or reconstruction).

def _mse_count(cfg):
    """
    MSE on spike counts.  Encourages correct neuron to fire `correct_rate`
    of total time-steps, others to fire `incorrect_rate`.
    Best for: rate-coded classification tasks.
    """
    return SF.mse_count_loss(
        correct_rate=cfg.get("correct_rate", 0.8),
        incorrect_rate=cfg.get("incorrect_rate", 0.2),
    )

def _ce_count(cfg):
    """
    Cross-entropy on accumulated spike counts.
    Best for: multi-class classification with rate coding.
    """
    return SF.ce_count_loss()

def _ce_rate(cfg):
    """
    Cross-entropy applied at every time step to the membrane potential
    (softmax over output neurons at each step).
    Best for: maximising per-timestep class separation.
    """
    return SF.ce_rate_loss()

def _mse_membrane(cfg):
    """
    Standard PyTorch MSE on raw membrane potential output.
    Best for: regression tasks (non-spiking output layer).
    """
    return nn.MSELoss()

def _l1_membrane(cfg):
    """L1 / Mean Absolute Error on membrane potential."""
    return nn.L1Loss()

def _stdp_trace(cfg):
    """
    Unsupervised STDP-inspired loss placeholder.

    True STDP updates weights locally using pre/post-synaptic spike traces
    rather than a global gradient signal.  In a software (PyTorch) setting
    we approximate this with a Hebbian correlation loss:

        L = -mean( pre_trace * post_spike )

    where pre_trace decays exponentially between spikes.  The negative sign
    turns gradient *descent* into *potentiation* (weights increase when
    pre fires before post).

    Note: this loss is used with `learning_mode = LearningMode.UNSUPERVISED`.
    The training loop must supply (pre_spikes, post_spikes) rather than
    (output, label) pairs.
    """
    tau_plus  = cfg.get("tau_plus",  20.0)   # trace decay time constant (ms)
    A_plus    = cfg.get("A_plus",    0.01)    # LTP amplitude
    A_minus   = cfg.get("A_minus",   0.0105)  # LTD amplitude

    class STDPTraceLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.tau_plus = tau_plus
            self.A_plus   = A_plus
            self.A_minus  = A_minus

        def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
            """
            pre_spikes  : [timesteps, batch, pre_neurons]
            post_spikes : [timesteps, batch, post_neurons]

            Returns a scalar loss that, when minimised, approximates STDP
            potentiation for causal spike pairs.
            """
            # Build a decaying pre-synaptic trace
            T   = pre_spikes.shape[0]
            trace = torch.zeros_like(pre_spikes[0])   # [batch, pre_neurons]
            ltp_total = torch.tensor(0.0, requires_grad=False)

            for t in range(T):
                # Decay trace
                trace = trace * (1.0 - 1.0 / self.tau_plus) + pre_spikes[t]
                # LTP: post fires → reward correlated pre traces
                # broadcast: [batch, post] x [batch, pre] → mean
                ltp = (post_spikes[t].sum(dim=-1, keepdim=True) * trace).mean()
                ltp_total = ltp_total + ltp

            # Negative: gradient descent → potentiation
            return -self.A_plus * ltp_total / T

    return STDPTraceLoss()


# Public supervised registry
SUPERVISED_LOSS_REGISTRY = {
    "mse_count"    : _mse_count,
    "ce_count"     : _ce_count,
    "ce_rate"      : _ce_rate,
    "mse_membrane" : _mse_membrane,
    "l1_membrane"  : _l1_membrane,
}

# Public unsupervised registry
UNSUPERVISED_LOSS_REGISTRY = {
    "stdp_trace"   : _stdp_trace,
}


def get_loss_function(name: str, mode: str = LearningMode.SUPERVISED,
                      cfg: dict = None):
    """
    Return a loss function instance by name and learning mode.

    Parameters
    ----------
    name : key in SUPERVISED_LOSS_REGISTRY or UNSUPERVISED_LOSS_REGISTRY
    mode : LearningMode.SUPERVISED | LearningMode.UNSUPERVISED
    cfg  : optional dict of loss hyperparameters
    """
    cfg = cfg or {}
    if mode == LearningMode.SUPERVISED:
        if name not in SUPERVISED_LOSS_REGISTRY:
            raise ValueError(
                f"Unknown supervised loss '{name}'. "
                f"Available: {list(SUPERVISED_LOSS_REGISTRY.keys())}"
            )
        return SUPERVISED_LOSS_REGISTRY[name](cfg)

    elif mode == LearningMode.UNSUPERVISED:
        if name not in UNSUPERVISED_LOSS_REGISTRY:
            raise ValueError(
                f"Unknown unsupervised loss '{name}'. "
                f"Available: {list(UNSUPERVISED_LOSS_REGISTRY.keys())}"
            )
        return UNSUPERVISED_LOSS_REGISTRY[name](cfg)

    else:
        raise ValueError(f"Unknown learning mode '{mode}'.")


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE DEFAULTS  (backward-compat with original Utils_snntorch)
# ──────────────────────────────────────────────────────────────────────────────

# These mimic the two lines present in the original file.
MSE           = nn.MSELoss()
loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
