# frameworks

The model side of the pipeline: one network, written once, in which only the LIF neuron changes
between frameworks.

| File | Role |
|---|---|
| `snn_model.py` | `SNNModel`: the shared network plus optimizer and loss; `training.framework` picks the neuron |
| `spiking_net.py` | The shared network (conv + LIF blocks, readout); plain PyTorch layers with a LIF after each |
| `adapters/` | One LIF adapter per framework under a common `BaseLIF`, built from `network_architecture.yaml`; only the chosen framework's library is imported |
| `reference_lif.py` | The LIF equation computed directly (exact and Euler), independent of any framework |
