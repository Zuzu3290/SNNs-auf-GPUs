# Framework Integration Guide

This document explains how the four SNN backends cooperate with the training
pipeline and what a developer needs to know before adding another one.

---

## The contract

Every model passed to `SNNTrainer`/`SNNTester` implements `ModelInterface`
(`frameworks/model_interface.py`) — PyTorch only. Every backend trains via
standard PyTorch autograd (`loss.backward()` + `optimizer.step()`); no other
backend (JAX, TensorFlow) is supported or accommodated. That flexibility was
explored and dropped rather than kept as unused surface area — see
[`additional_frameworks.md`](additional_frameworks.md) for what was tried
and why it didn't stick.

Core methods the trainer calls: `forward(data) -> torch.Tensor`,
`backward_pass(loss, scaler, do_step)`, `zero_grad()`, `train_mode()`/
`eval_mode()`, `get_lr()`, `get_state()`. `tensor_format()` declares the
tensor layout a model expects (`"TB"` time-first by default, `"BT"` for
Sinabs) — the trainer transposes automatically.

---

## The four backends

| Backend | Neuron model | Notes |
|---|---|---|
| Norse | `LIFCell` | Explicit per-timestep state-tuple handling |
| SNNTorch | `snn.Alpha` | Second-order neuron: coupled synaptic-current + membrane-potential decay |
| SpikingJelly | `IzhikevichNode` | Two coupled ODEs (`v`, `u`) — richer firing dynamics than LIF |
| Sinabs | LIF, batch-first | DVS-first; `tensor_format() == "BT"`; ships a real export path to SynSense's Speck chip |

All four trace PyTorch autograd end-to-end, so adversarial robustness (FGSM,
PGD, TRADES) works fully — gradients flow from the loss all the way to the
input tensor for every backend.

---

## Adding a new backend

1. Implement `ModelInterface` in `frameworks/`.
2. `forward()` takes and returns a `torch.Tensor`.
3. Override `tensor_format()` only if the model needs batch-first input.
4. Wire it into `MODELS` in `learning/main.py` and `training.framework` in
   `configuration/SNN_module.yaml`.
