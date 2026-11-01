# configuration

Every setting a run uses is in these three files. A missing key raises an error rather than
falling back to a default.

| File | Controls | Read by |
|---|---|---|
| `SNN_module.yaml` | Framework, training device, inference device, epochs, batch-size calibration, optimiser, learning rate, output paths, checkpoint | `skeleton/config.py` |
| `network_architecture.yaml` | Convolution blocks, BatchNorm, hidden layers, and each framework's LIF neuron parameters | `skeleton/config.py` |
| `data_workflow.yaml` | Time bins, denoising, cache strategy, augmentation, DataLoader workers, VRAM target for the batch size | `skeleton/config.py` |

## Changing the network or adding a framework

`network_architecture.yaml` is the file to edit when designing your own network. Under
`convolution:`, each entry in `blocks` is one convolution → neuron → pooling stage (`out` channels,
`kernel` size). Add or remove entries to make the network deeper or shallower, list widths in
`fc_hidden` (e.g. `[256]`) for hidden layers before the output, and set `pool_kernel` (how much each
stage shrinks the picture), `padding` (zeros added around the edges so the convolution keeps the
size) and `batch_norm`. The input size and number of outputs come from the dataset, and an
impossible combination stops the run at startup. Under `neuron:`, each framework has its own block,
read only by that framework's adapter in `frameworks/`. Change the values to tune a neuron,
or add a block for a new framework with whatever parameters its library needs. Other frameworks are
unaffected, and a key the adapter expects but the block lacks stops the run with an error naming
it. Write a surrogate gradient as a block (`type:` plus its parameters), as the existing frameworks
do. Adding the framework itself (its file in `frameworks/`, plus one line each in `skeleton/config.py`
and `skeleton/results_collection.py`) is described step by step in `frameworks/README.md`.

