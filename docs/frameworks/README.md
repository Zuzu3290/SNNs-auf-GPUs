# Framework Integration Guide

This document explains how different ML frameworks cooperate with the training
pipeline and what a developer needs to know before building a custom backend.

---

## The contract

Every model passed to `SNNTrainer` or `SNNTester` must satisfy one rule:

> `forward(data: torch.Tensor) → torch.Tensor`

Input is a PyTorch tensor on the configured device. Output is a PyTorch tensor
of spikes. What happens in between is entirely up to the framework.

The `ModelInterface` base class (see `src/learning/frameworks/`) formalises this
into eight methods that the trainer calls and nothing else.

---

## PyTorch backends (SNNTorch, Norse, SpikingJelly)

These work natively. PyTorch autograd traces through the model, `loss.backward()`
flows gradients back to every parameter, and `optimizer.step()` updates weights.
Adversarial robustness (FGSM, PGD) works fully because gradients propagate from
the loss all the way to the input tensor.

---

## JAX backend

### How it cooperates with the pipeline

JAX runs on the same CUDA device as PyTorch. The two share GPU memory via
**DLPack** — a zero-copy tensor exchange protocol:

```python
# PyTorch tensor → JAX array (zero-copy)
import jax
jax_array = jax.dlpack.from_dlpack(torch.to_dlpack(pt_tensor))

# JAX array → PyTorch tensor (zero-copy)
pt_tensor = torch.from_dlpack(jax.dlpack.to_dlpack(jax_array))
```

The JAX model adapter:
1. Receives a PyTorch tensor from the trainer
2. Converts it to a JAX array via DLPack
3. Runs the forward pass through XLA (compiled, fused, on GPU)
4. Converts the output spikes back to a PyTorch tensor via DLPack
5. Returns to the trainer as if nothing unusual happened

### Training — who owns the backward pass

This is the critical difference. PyTorch cannot differentiate through XLA
operations, so `loss.backward()` stops at the DLPack boundary.

**The JAX adapter must compute its own gradients before returning.**

The standard pattern using `jax.value_and_grad` and `optax`:

```python
import jax
import optax

def forward_and_update(params, opt_state, x, targets):
    def loss_fn(params):
        spikes = model.apply(params, x)
        return cross_entropy(spikes, targets)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, spikes
```

`jax.jit(forward_and_update)` compiles this entire block — forward pass,
loss, gradient, and weight update — into a single XLA program. One GPU
kernel launch covers what PyTorch does in many separate steps.

The trainer's `loss.backward()` and `optimizer.step()` calls become no-ops
for JAX models. The `ModelInterface.optimizer_step()` method is where the
JAX adapter signals that weights have already been updated internally.

### Adversarial robustness limitation

FGSM and PGD require `loss.backward()` to flow gradients back to the **input**
tensor. Since PyTorch cannot trace through XLA, this is not possible for JAX
models. The adversarial evaluator skips the attack and runs clean evaluation
only when it detects a non-differentiable boundary.

A workaround exists — `jax.custom_vjp` can expose a hand-defined backward pass
to PyTorch via DLPack — but this is advanced and only needed if adversarial
robustness under JAX is a hard requirement.

### What JAX gives you that PyTorch does not

- `jax.jit` fuses the entire temporal LIF loop into one XLA kernel — no per-timestep launch overhead
- `jax.vmap` vectorises across neurons or seeds without writing CUDA
- `jax.hessian` and higher-order derivatives for research experiments
- Functional transforms compose cleanly: `jit(grad(vmap(loss_fn)))`

---

## TensorFlow backend

Works the same way as JAX via DLPack:

```python
import tensorflow as tf

tf_tensor = tf.experimental.dlpack.from_dlpack(torch.to_dlpack(pt_tensor))
pt_tensor = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(tf_tensor))
```

`tf.experimental.dlpack` is stable from TensorFlow 2.5 onwards.

Gradient flow has the same limitation as JAX — TensorFlow's autograd cannot
be traced by PyTorch. The adapter manages its own `tf.GradientTape` and weight
updates internally before returning the output tensor.

---

## Custom / from-scratch backend

If someone builds a model without any framework — pure NumPy, custom C, or
their own autograd — the only requirement is:

1. Implement `ModelInterface`
2. Return a `torch.Tensor` from `forward()`
3. Handle your own weight updates inside `optimizer_step()`

A NumPy model pays a CPU → GPU copy cost on the tensor conversion. Everything
else in the pipeline — caching, logging, checkpointing, inference metrics —
works without modification.

---

## Summary

| Backend | Trains via | Backward pass | Adversarial eval |
|---------|-----------|---------------|-----------------|
| PyTorch | PyTorch autograd | Full | Full |
| JAX | `jax.value_and_grad` + optax | Internal only | Clean eval only |
| TensorFlow | `tf.GradientTape` | Internal only | Clean eval only |
| Custom | Your implementation | Your implementation | Clean eval only |
