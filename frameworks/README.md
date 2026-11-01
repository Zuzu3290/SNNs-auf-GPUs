# frameworks

The model side of the pipeline: one network, written once, in which only the spiking neuron changes
between frameworks.

| File | Role |
|---|---|
| `snn_model.py` | `BaseNeuron`, the interface every framework's neuron implements (plus spike counting), and `SNNModel`, the shared network: plain PyTorch layers in an `nn.Sequential` with a neuron after each conv and linear layer. The optimizer and loss live in `learning/training.py` |
| `snntorch_lif.py`, `norse_lif.py`, `spikingjelly_lif.py`, `sinabs_lif.py` | One class `Neuron` per framework, built from `network_architecture.yaml`. `main.py` imports only the file named by `training.framework`, so a run loads a single SNN library |
| `reference_lif.py` | The LIF equation written out directly (exact and Euler), independent of any framework; for studying the neuron itself, not imported by the pipeline |

## Adding a framework or your own neuron

Every convolution, pooling and linear layer is plain PyTorch and shared, so a new framework is one
neuron layer plus its name in the config.

**1. Write the file** `frameworks/<key>_lif.py`, where `<key>` is the name you give it under `neuron:`
in step 3. It holds one class called `Neuron`, a subclass of `BaseNeuron`, with three methods:

- `__init__` reads its block from the config and builds the neuron as `self.neuron`;
- `forward` takes one timestep, passes it through `self.neuron`, calls `self.record(spikes)` and returns the spikes;
- `reset` clears the neuron's state; the network calls it before every batch.

```python
import torch
import mylib                                   # the SNN library
from frameworks.snn_model import BaseNeuron


class Neuron(BaseNeuron):
    """mylib's LIF behind BaseNeuron."""

    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.NEURON["mylib"]                # this framework's block in network_architecture.yaml
        self.neuron = mylib.LIF(tau=n["tau"], threshold=n["threshold"])
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes, self.state = self.neuron(x, self.state)
        self.record(spikes)                    # counts spikes for the spike rates and SynOps
        return spikes

    def reset(self) -> None:
        self.state = None
```

Start from whichever of the four existing files is closest to your library. If the library takes the
whole time sequence in one call, return `True` from `consumes_sequence()`, as `spikingjelly_lif.py`
does; `forward` then receives `[T, batch, ...]`.

**Your own neuron** needs no library: write the update directly in `forward`, in plain PyTorch, as
`reference_lif.py` does for the LIF. It does not have to be a LIF either; any neuron model (adaptive
threshold, Izhikevich, ...) works, as long as `forward` returns one spike value per neuron per
timestep, shaped like its input. Spikes must be computed with a surrogate gradient to be trainable.

**2. Give it a name.** In `skeleton/config.py`, add `"mylib": "mylib"` to `FW_TO_CFG_KEY`
(the `--framework` name → the `<key>` of step 1). For a library, also add `"mylib": "<pip package>"`
to `FRAMEWORK_PACKAGES` in `skeleton/results_collection.py`, so the results record its version, and the
package to `requirements.txt`.

**3. Add its parameters** under `neuron:` in `configuration/network_architecture.yaml`:

```yaml
neuron:
  mylib:
    tau: 10.0
    threshold: 1.0
```

**4. Run it:** `python learning/main.py --framework mylib`. The printed weight fingerprint should match
the other frameworks for the same seed, which confirms the neuron is the only difference. If it does
not, the library registers its own trainable parameters; `sinabs_lif.py` shows how to freeze them.

To compare neurons fairly, match their behaviour, not only their parameter names: libraries differ in
input scaling, reset (to zero or by subtraction, at once or one step later) and whether a neuron may
fire more than once per step. The docstrings of the four files note which defaults were overridden.
