# frameworks

The model side of the pipeline: one network, written once, in which only the LIF neuron changes
between frameworks.

| File | Role |
|---|---|
| `model_interface.py` | `ModelInterface`, the contract the trainer and tester use; they never see a specific framework |
| `snn_model.py` | `SNNModel`, the one implementation of `ModelInterface` for all four frameworks |
| `snn_torch.py`, `snn_spikingjelly.py`, `snn_norse.py`, `snn_sinabs.py` | Thin classes that select the framework for `SNNModel` |
| `spiking_net.py` | The shared network (conv + LIF blocks, readout); plain PyTorch layers with a LIF slot after each |
| `adapters/` | One LIF adapter per framework under a common `BaseLIF`, built from `network_architecture.yaml` |
| `reference_lif.py` | The LIF equation computed directly (exact and Euler), independent of any framework |
| `snn_lif.py` | A hand-written PyTorch LIF network; not wired into `learning/main.py` |
