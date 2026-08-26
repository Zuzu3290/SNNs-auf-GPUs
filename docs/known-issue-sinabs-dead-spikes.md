# Known issue: Sinabs produces zero output spikes (dead network)

**Status: open, not fixed.** Found during the diagnostic-mode verification run of
`run_benchmark.py` on N-MNIST (1 epoch, 20-iteration subset, 10 test batches), before
the full 5-epoch benchmark run.

## Symptom

Every batch in the Sinabs test run reports `Spikes: 0` (`Total Spikes Activated: 0`
across all 1280 test samples). Training shows `Spike Rate: 0.0000`, `Firing Rate: 0.00 Hz`.
`framework_ratio` (input activity / output spikes) is undefined (`N/A`) because the
denominator is zero.

The reported test accuracy (76.56%) is not real: with the output layer never spiking,
the classifier's logits are effectively constant/zero, so predictions collapse to a
single class. In this diagnostic run's 1280-sample slice, ground-truth class 0 happens
to make up 980/1280 samples, so "always predict class 0" alone produces ~76% accuracy —
consistent with the label-distribution dump (`Class 0 | GT: 980 Pred: 1280`, every other
class predicted 0 times).

## Where

`learning/frameworks/snn_sinabs.py` — `build_sinabs_layer()` builds `sl.LIF(tau_mem=20.0,
spike_threshold=0.5, ...)` per `configuration/SNN_module.yaml`'s `sinabs:` block. These
threshold/tau_mem values are nominally comparable to snnTorch's (`beta=0.95,
threshold=0.5`), but Sinabs's `sl.LIF` internally scales/leaks membrane potential
differently, so the same nominal threshold does not guarantee the same firing behavior —
same class of problem as the documented Norse `threshold=1.0` dead-neuron finding
(EXP-004; see [Haseeb-open-items.md](Haseeb-open-items.md)), just not yet root-caused for
Sinabs specifically.

## Not yet investigated

- Whether `spike_threshold=0.5` is simply too high given Sinabs's actual membrane-update
  scale (untested: lowering the threshold, or an explicit input gain akin to Haseeb's
  Norse `input_scale`).
- Whether `sl.LIF`'s default surrogate gradient function is even installed/active here —
  `build_sinabs_layer()` passes no explicit surrogate function, unlike snnTorch/Norse/
  SpikingJelly which all take one.
- Whether this is scale-dependent (only 1 diagnostic epoch on a 20-iteration subset) or a
  hard structural dead-neuron state that would persist through the full 5-epoch run too.

## Impact on the full benchmark run

Sinabs's numbers in the full run (`--full`) should be treated as unreliable — accuracy,
spike-rate, energy-per-spike, and `framework_ratio` are all downstream of real spiking
activity, none of which is happening — until this is root-caused and fixed. Norse,
snnTorch (torch), and SpikingJelly (sj) all showed real, nonzero spiking activity in the
same verification run and are not affected by this issue.
