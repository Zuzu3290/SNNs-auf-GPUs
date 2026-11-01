# SNNs-auf-GPUs

A pipeline for training and evaluating spiking neural networks (SNNs) on event-camera data on a GPU.
Four PyTorch-based SNN frameworks — snnTorch, SpikingJelly, Norse and Sinabs — share one network,
one data path and one measurement layer, so that the neuron implementation is the only thing that
changes between runs.

## Project layout

```
configuration/        The three YAML files every run reads (see configuration/README.md)
event_data_workflow/  Event data: dataset registry, denoising, framing, caching, DataLoaders, resource monitor
frameworks/           The shared spiking network and one LIF adapter per framework
learning/             Entry point, training loop, testing, metrics
skeleton/             Settings, config loading and validation, seeding, results files
docs/                 Background notes, design records and earlier reports
```

## Running

```
python learning/main.py
```

`learning/main.py` asks for the dataset and the inference device, prints the configuration and the network layer by layer, calibrates the batch size to the GPU, trains, tests and
writes the results. Installation, options and the Colab setup are in [HOW_TO_RUN.md](HOW_TO_RUN.md).

## Datasets

Three classification datasets train end to end: N-MNIST, N-Caltech101 and DVS128 Gesture. The
registry in `event_data_workflow/dataset_registry.py` also holds loaders for two regression datasets,
DAVIS Camera Pose and Eye Tracking, which the pipeline does not train. A new classification dataset
only needs a registry entry with its tonic class, sensor size and number of classes; input size,
output layer and batch size follow automatically.
Events must be available as `(x, y, t, p)`: pixel column, pixel row, timestamp, polarity.

## Frameworks

| Framework | Selected with | Neuron adapter |
|---|---|---|
| snnTorch | `torch` | `frameworks/snntorch_lif.py` |
| SpikingJelly | `sj` | `frameworks/spikingjelly_lif.py` |
| Norse | `norse` | `frameworks/norse_lif.py` |
| Sinabs | `sinabs` | `frameworks/sinabs_lif.py` |

The framework is set by `training.framework` in `configuration/SNN_module.yaml` or `--framework`.
Each framework's neuron parameters live in `configuration/network_architecture.yaml`.

## Adding a framework

Only the spiking neuron changes between frameworks; the network, training, data path and measurements are
shared. To add an SNN library, a different neuron from a library already here, or a neuron you write
yourself, add one file with a class `Neuron` to `frameworks/`, give it a name in the config, and run it
with `--framework <name>`. The step-by-step guide, with a template, is in
[frameworks/README.md](frameworks/README.md).

## What is measured

Every run records test accuracy, spike rates (training, test and per layer),
single-sample latency, throughput, training forward/backward time, peak GPU memory, GPU energy from
NVML, and the synaptic operations (SynOps) per inference.
The synaptic-operation energy columns are an estimate for neuromorphic hardware, not a measurement
on a chip: SynOps × 0.9 pJ, the energy of one addition at 45 nm (Horowitz, ISSCC 2014;
`SYNOPS_ENERGY_PJ` in `learning/utilities.py`).
