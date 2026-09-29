# Participation Ratio / Spike Entropy — sample-size ceiling

## TL;DR

At the batch size ex6 just locked in (64), **Participation Ratio (PR) cannot exceed
63** regardless of true neuron count, and **spike entropy tracks `log2(neuron count)`
closely for ordinary firing patterns** — for our conv layers (3k–115k neurons), both
risk mostly re-measuring layer size rather than anything new. Need a decision on one
of three fixes below before ex7 runs, since the probe-batch size is frozen for the
whole study once chosen.

## The two metrics, as specified

**Participation Ratio** (Scalability.pdf, verbatim):

```
PR = (Σ λᵢ)² / Σ λᵢ²
```

λᵢ = eigenvalues of the layer's neuron-covariance matrix, estimated from a batch of B
samples' per-neuron firing rates.

**Spike-train entropy** (Shannon entropy over the population code):

```
H = − Σ pᵢ log₂ pᵢ
```

pᵢ = neuron *i*'s mean firing rate, normalized across neurons into a distribution.

## The limitation

**PR — a hard sample-size ceiling.** With only B samples, the covariance estimate has
at most `B−1` nonzero eigenvalues (one degree of freedom lost to centering). So
`PR ≤ B−1` **by construction**, independent of the layer's real neuron count. At
`batch_size=64`, that ceiling is **63**.

Measured on ex6's real layers (all comfortably above that ceiling in neuron count):

| layer | neurons | PR ceiling at B=64 |
|---|---|---|
| lif1 | 10,800 – 115,200 | 63 |
| lif2 | 3,872 – 21,632 | 63 |

**Entropy — not a sample-size artifact, an inherent property.** For roughly uniform
firing across neurons (the normal case for a healthy, non-degenerate layer), `H`
converges toward `log2(N)` regardless of how many samples are used. More samples give
a more *precise* estimate that H ≈ log2(N) — they don't change the fact that it does.

## Which fix addresses which metric

| option | fixes PR? | fixes entropy? |
|---|---|---|
| 1. Accept, document the caveat | no | no |
| 2. Accumulate multiple probe batches (e.g. 4–8, ~256–512 samples) | yes — raises ceiling to ~255–511 | **no** — larger sample confirms H≈log2(N) more precisely, doesn't remove it |
| 3. Normalize (`PR / ceiling`, `H / log2(N)`) | yes | yes |

## Three options

1. **Accept and document.** Zero extra work. Risk: numbers may be uninformative for
   the layers the width ladder cares about most.
2. **Bigger probe sample.** Small code change, no effect on training or ex6's
   settings. Only fixes PR; entropy still tracks `log2(N)`.
3. **Normalize both metrics against their own maximum.** Cheap, fixes both. Departs
   from the un-normalized formulas as quoted from the cited literature (Rigotti et al.,
   Gao et al. use raw PR) — no longer a direct apples-to-apples comparison against
   those papers' numbers.

## Decision needed

Which option (or a different one) should the scalability study use for PR and spike
entropy, given ex6's batch size of 64 is locked for the whole study?
