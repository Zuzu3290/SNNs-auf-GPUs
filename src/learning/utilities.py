"""
Shared helper factories used by all three SNN framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss, reset_mode_guard
    from learning.utilities import build_lif_layer   # SNNTorch only
    from learning.utilities import build_sj_layer    # SpikingJelly only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_optimizer(params, fw_cfg: dict) -> torch.optim.Optimizer:
    """
    Factory that reads optimizer name + lr + wd from fw_cfg.

    fw_cfg must contain: optimizer, learning_rate, weight_decay.
    Supported names: adam (default), adamw, sgd.
    """
    lr  = fw_cfg["learning_rate"]
    wd  = fw_cfg["weight_decay"]
    opt = fw_cfg.get("optimizer", "adam").lower()

    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


def build_loss(fw_cfg: dict, framework: str = "norse"):
    """
    Factory that reads loss_fn name from fw_cfg and returns a callable.

    Supported loss names:
      cross_entropy — standard classification loss.
          Norse/SNNTorch: spk_rec is [T, B, C]; sums over T before loss.
          SpikingJelly:   forward already sums T, returns [B, C]; uses nn.CrossEntropyLoss.
      mse_count     — SNNTorch mse_count_loss (requires snntorch installed).

    Args:
        fw_cfg    : dict from cfg.FRAMEWORK_CFG[<framework>] merged with lr/wd
        framework : "norse" | "torch" | "spikingjelly"
    """
    loss_name = fw_cfg.get("loss_fn", "cross_entropy")

    if loss_name == "cross_entropy":
        if framework == "spikingjelly":
            return nn.CrossEntropyLoss()
        return lambda spk_rec, targets: F.cross_entropy(spk_rec.float().sum(0), targets)

    if loss_name == "mse_count":
        from snntorch import functional as SF
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    raise NotImplementedError(
        f"loss_fn='{loss_name}' not supported for framework='{framework}'. "
        "Supported: cross_entropy, mse_count."
    )


def reset_mode_guard(fw_cfg: dict, framework: str = "norse") -> None:
    """
    Raises if the requested reset_mode is incompatible with the framework.
    Norse only supports hard reset ('zero'). Call this in __init__ when
    reset_mode is exposed in the YAML.
    """
    mode = fw_cfg.get("reset_mode", "zero")
    if framework == "norse" and mode == "subtract":
        raise NotImplementedError(
            "Norse only supports reset_mode='zero' (hard reset). "
            "Set reset_mode: zero in the norse section of SNN_module.yaml."
        )


def build_norse_layer(layer_name: str, cfg) -> nn.Module:
    """
    Build a Norse neuron, selecting the type from cfg.NEURON_TYPES.

    Neuron types (set per layer in network_architecture.yaml under neuron_types.norse):
      lif_cell       — norse.LIFCell  (standard leaky integrate-and-fire). Default.
      lif_rec_cell   — norse.LIFRecurrentCell  (adds recurrent self-connection).

    Uses tau_mem_inv and threshold from cfg.FRAMEWORK_CFG["norse"].
    """
    import norse.torch as norse
    import torch

    # layer_name: "lif1" | "lif2" | "lif_out"
    neuron_type = cfg.NEURON_TYPES.get("norse", {}).get(layer_name, "lif_cell")
    tau_mem_inv = cfg.FRAMEWORK_CFG["norse"]["tau_mem_inv"]
    threshold   = cfg.FRAMEWORK_CFG["norse"]["threshold"]

    lif_params = norse.LIFParameters(
        tau_mem_inv = torch.as_tensor(tau_mem_inv, dtype=torch.float32),
        v_th        = torch.as_tensor(threshold,   dtype=torch.float32),
    )

    if neuron_type == "lif_rec_cell":
        return norse.LIFRecurrentCell(p=lif_params)
    return norse.LIFCell(p=lif_params)


def build_lif_layer(layer_name: str, cfg, spike_grad, **kwargs) -> nn.Module:
    """
    Build an SNNTorch LIF neuron, selecting the type from cfg.NEURON_TYPES.

    Neuron types:
      alpha  — snn.Alpha  (two-compartment: membrane + synaptic decay). Default.
      leaky  — snn.Leaky  (single-compartment).

    Uses threshold and beta from cfg.FRAMEWORK_CFG["snntorch"].
    """
    import snntorch as snn

    # layer_name: "lif1" | "lif2" | "lif_out"
    neuron_type = cfg.NEURON_TYPES.get("snntorch", {}).get(layer_name, "alpha")
    beta        = cfg.FRAMEWORK_CFG["snntorch"]["beta"]
    threshold   = cfg.FRAMEWORK_CFG["snntorch"]["threshold"]

    if neuron_type == "alpha":
        alpha_val = beta
        beta_syn  = max(0.5, beta - 0.1)
        return snn.Alpha(
            alpha=alpha_val, beta=beta_syn,
            threshold=threshold, spike_grad=spike_grad,
            **kwargs,
        )
    return snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad, **kwargs)


def build_sj_layer(layer_name: str, cfg, spike_grad, **kwargs) -> nn.Module:
    """
    Build a SpikingJelly neuron, selecting the type from cfg.NEURON_TYPES.

    Neuron types:
      izhikevich — neuron.IzhikevichNode. Default.
      lif        — neuron.LIFNode.

    Uses threshold and tau from cfg.FRAMEWORK_CFG["spikingjelly"].
    """
    from spikingjelly.activation_based import neuron

    # layer_name: "lif1" | "lif2" | "lif_out"
    neuron_type = cfg.NEURON_TYPES.get("spikingjelly", {}).get(layer_name, "izhikevich")
    tau         = cfg.FRAMEWORK_CFG["spikingjelly"]["tau"]
    threshold   = cfg.FRAMEWORK_CFG["spikingjelly"]["threshold"]

    if neuron_type == "izhikevich":
        return neuron.IzhikevichNode(
            tau=tau, v_threshold=threshold,
            surrogate_function=spike_grad,
            **kwargs,
        )
    return neuron.LIFNode(
        tau=tau, v_threshold=threshold,
        surrogate_function=spike_grad,
        **kwargs,
    )
