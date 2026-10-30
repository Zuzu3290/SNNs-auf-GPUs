# configuration

Every setting a run uses is in these three files. A missing key raises an error rather than
falling back to a default.

| File | Controls | Read by |
|---|---|---|
| `SNN_module.yaml` | Framework, device, epochs, batch-size calibration, optimiser, learning rate, output paths, checkpoint | `skeleton/config.py` |
| `network_architecture.yaml` | Convolution blocks, BatchNorm, hidden layers, and each framework's LIF neuron parameters | `skeleton/config.py` |
| `data_workflow.yaml` | Time bins, denoising, cache strategy, DataLoader workers, VRAM target for the batch size | `skeleton/config.py` |

