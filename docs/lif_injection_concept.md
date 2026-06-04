# Design Consideration — Injectable LIF Kernel Primitive

Status: Open — not implemented. Review before proceeding.

---

## Current behaviour

When `kernel: ON`, `SNNTrainer.forward_pass()` replaces `model.forward()` entirely:

```
data → reshape to [B, N, T] → kernel.warp_oriented_forward() → spikes → training loop
                                      ↑
                        ModelInterface.forward() is never called
```

The model's internal architecture — hidden layers, custom neurons, attention, any JAX computation — is bypassed. The kernel acts as the whole model, not as a layer inside it.

This works for models where the kernel IS the correct LIF implementation for every layer. It breaks down when:
- The model has multiple LIF layers with different parameters per layer
- Hidden layers use custom neuron types (adaptive threshold, conductance-based, etc.)
- The model is a JAX backend with its own computation graph for non-LIF operations

---

## Proposed behaviour — kernel as injectable layer

The kernel becomes a compute primitive that a model calls internally for its LIF operations, rather than a full model replacement:

```
data → model.forward()
           ↓
       linear layer
           ↓
       kernel.lif(x, voltage, cfg)   ← kernel called here by the model
           ↓
       linear layer
           ↓
       kernel.lif(x, voltage, cfg)   ← called again for each spiking layer
           ↓
       output
```

The model owns the architecture. The kernel owns the LIF compute.

---

## Interface change required

`ModelInterface` would need an optional hook:

```python
class ModelInterface(ABC):

    def kernel_lif(
        self,
        x:       torch.Tensor,   # [B, N, T] input current
        voltage: torch.Tensor,   # [B, N]    membrane voltage (mutated in place)
        v_th:    float,
        tau_inv: float,
        kernel_module,            # snn_forward module, passed by trainer
        mode:    str = "warp_oriented",
    ) -> torch.Tensor:
        """
        Default: calls the appropriate kernel function for this mode.
        Models override this to skip the kernel for layers where it does not apply.
        """
        if mode == "warp_oriented":
            spikes, _, _ = kernel_module.warp_oriented_forward(x, voltage, v_th, tau_inv)
        elif mode == "temporal":
            spikes = kernel_module.temporal_forward(x, voltage, v_th, tau_inv)
        else:
            spikes = kernel_module.forward(x, voltage, v_th, tau_inv)
        return spikes
```

Each model's `forward()` then calls `self.kernel_lif(...)` at each spiking layer instead of its own LIF implementation.

---

## Compatibility

| Model type | Compatible | Notes |
|---|---|---|
| `SNN_TORCH` — standard LIF layers | Yes — replace each LIF call with `kernel_lif()` | |
| `SNN_NORSE` — Norse functional LIF | Yes — Norse uses functional API, easy to swap | |
| `SNN_SJ` — SpikingJelly | Partial — SpikingJelly wraps LIF in nn.Module; requires unwrapping | |
| JAX backend | Yes — `forward()` calls `kernel_lif()` via PyTorch wrapper at LIF layers | |
| RNN / CNN / Transformer | No — no LIF layer to inject into | kernel_lif() never called, no change in behaviour |

---

## What changes

- `ModelInterface` — add default `kernel_lif()` method
- `SNNTrainer.forward_pass()` — pass `self.kernel_module` and `self.kernel_mode` to `self.model(data)` (or as a context that models read)
- Each framework model (`snn_torch.py`, `snn_norse.py`, `snn_sj.py`) — replace internal LIF forward with `self.kernel_lif()` calls
- Compatibility report — updated to check per-layer compatibility, not whole-model

## What stays the same

- Kernel code (C++/CUDA) — no changes
- Training loop — no changes
- Backward pass — no changes (kernel outputs are differentiable tensors)
- Config — no changes

---

## Open questions before implementation

1. Voltage buffer ownership — currently the trainer holds `_voltage_buf`. With injection, each LIF layer needs its own voltage buffer. Does the model own them or does the trainer?
2. Multi-layer parameter variation — if layer 1 has `v_th=1.0` and layer 2 has `v_th=0.8`, each call to `kernel_lif()` passes different params. The current kernel accepts these as call-time arguments — no change needed in C++.
3. SpikingJelly wrapper — SpikingJelly's LIF is a stateful `nn.Module`. Unwrapping it to extract `[B, N, T]` tensors for the kernel requires accessing internal state that SpikingJelly manages.
