"""
Shared helper factories used by all three SNN framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss
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
      cross_entropy   — standard classification loss.
          Norse/SNNTorch: spk_rec is [T, B, C]; sums over T before loss.
          SpikingJelly:   forward already sums T, returns [B, C]; uses nn.CrossEntropyLoss.
      mse_count       — SNNTorch mse_count_loss (requires snntorch installed).
      mse_regression  — Phase B regression models (personal/snn_*_regression.py).
          Norse/SNNTorch/Sinabs: readout is [T, B, output_dim]; averages over T before loss
          (continuous analog readout, not a spike count — there's nothing to sum).
          SpikingJelly: forward already averages T, returns [B, output_dim]; uses nn.MSELoss.

    Args:
        fw_cfg    : dict from cfg.FRAMEWORK_CFG[<framework>] merged with lr/wd
        framework : "norse" | "torch" | "spikingjelly" | "sinabs"
    """
    loss_name = fw_cfg.get("loss_fn", "cross_entropy")

    if loss_name == "cross_entropy":
        if framework == "spikingjelly":
            return nn.CrossEntropyLoss()
        return lambda spk_rec, targets: F.cross_entropy(spk_rec.float().sum(0), targets)

    if loss_name == "mse_count":
        from snntorch import functional as SF
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    if loss_name == "mse_regression":
        if framework == "spikingjelly":
            return nn.MSELoss()
        return lambda readout, targets: F.mse_loss(readout.float().mean(0), targets.float())

    raise NotImplementedError(
        f"loss_fn='{loss_name}' not supported for framework='{framework}'. "
        "Supported: cross_entropy, mse_count, mse_regression."
    )


