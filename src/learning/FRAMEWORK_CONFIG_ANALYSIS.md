# Framework Config Parameter Analysis — SNNTorch vs Norse vs SpikingJelly

A thorough breakdown of which `SNN_module.yaml` parameters must be framework-specific, which can safely be shared across all three frameworks, and which look shared but actually mean different things.

For each parameter we distinguish:
- **What each library considers optimal for itself** (max performance per framework)
- **What we need for fair cross-framework comparison** (level playing field)

---

## Parameter classification matrix

| Parameter | SNNTorch | Norse | SpikingJelly | Verdict |
|---|---|---|---|---|
| `architecture.*` (conv, pool, FC) | ✓ | ✓ | ✓ | **SHARED** — must be identical for fair comparison |
| `architecture.threshold` | `snn.Leaky(threshold=)` | `LIFParameters(v_th=)` | `LIFNode(v_threshold=)` | **NAME shared, MEANING differs** — see below |
| `architecture.reset_mode` | `reset_mechanism=` (config-driven) | only "zero" supported (crashes on "subtract") | `v_reset=0.0` or `None` (config-driven) | **Norse only supports hard reset — others fully configurable** |
| `training.timesteps` | T loop | T loop | T loop | **SHARED** |
| `training.batch_size` | ✓ | ✓ | ✓ | **SHARED** |
| `training.learning_rate` | ✓ | ✓ | ✓ | **Usually shared, framework-sensitive** |
| `training.weight_decay` | ✓ | ✓ | ✓ | **SHARED** |
| `training.num_workers`, `grad_accum_steps`, `lr_scheduler` | ✓ | ✓ | ✓ | **SHARED** |
| `training.use_amp` | works | **risky** (silent gradient underflow) | works | **MUST be false for Norse; verify for others** |
| `training.loss_fn` | `cross_entropy` or `mse_count` (config-driven) | `cross_entropy` only (crashes on others) | `cross_entropy` only (crashes on others) | **`mse_count` is SNNTorch-only; misconfig crashes loudly** |
| `training.optimizer` | ✓ | ✓ | ✓ | **SHARED** |
| `training.surrogate` (YAML inactive) | hardcoded `atan` | hardcoded `SuperSpike` (Norse default) | hardcoded `ATan` | **YAML param inactive — surrogate hardcoded per framework, each prints what it uses at startup** |
| `frameworks.snntorch.beta` (0–1) | β decay coeff | — | — | **SNNTorch only** |
| `frameworks.norse.tau_mem_inv` (Hz) | — | inverse τ_mem | — | **Norse only** |
| `frameworks.spikingjelly.tau` (dimensionless) | — | — | τ ratio | **SpikingJelly only** |

---

## 1. Threshold — the headline portability issue

The hypothesis was: *"maybe 1.0 works for Norse but not SNNTorch."* The math says the opposite. Here's why.

All three frameworks expose `threshold` as a single voltage value, but the **scale of the membrane voltage V relative to input I is wildly different**. Working through the update equations:

### SNNTorch — `snn.Leaky(beta=0.95)`

```
V[t+1] = β · V[t] + I[t]
       = 0.95·V[t] + I[t]
```

For sustained input I, equilibrium: `V_eq = I / (1 − β) = I / 0.05 = 20·I`

→ **Input is 20× AMPLIFIED** at the membrane.

### Norse — `LIFCell(tau_mem_inv=50.0)` (default dt = 1 ms)

```
V[t+1] = V[t] + dt · τ_mem_inv · (v_leak − V[t] + I[t])
       = V[t] + 0.001 · 50 · (0 − V[t] + I[t])
       = 0.95·V[t] + 0.05·I[t]
```

Equilibrium: `V_eq = I` → **no amplification.**

### SpikingJelly — `LIFNode(tau=2.0)`

```
V[t+1] = V[t] + (X[t] − V[t]) / τ
       = 0.5·V[t] + 0.5·X[t]
```

Equilibrium: `V_eq = X` → **no amplification.**

### What this means for `threshold` portability

For the same effective input I, how easily does V reach threshold?

| Threshold | SNNTorch needs sustained I > | Norse needs sustained I > | SpikingJelly needs sustained I > |
|---|---|---|---|
| 0.5 | **0.025** | 0.5 | 0.5 |
| 1.0 | **0.05** | 1.0 | 1.0 |

**Norse with threshold=1.0 is the hardest to spike of the three configurations.** The hypothesis has it reversed: SNNTorch tolerates higher thresholds (it amplifies 20×), while Norse and SpikingJelly need lower thresholds — or, equivalently, much higher initial weights — to fire at all.

This aligns with the earlier accuracy gap finding and the Norse-only-9%-first-epoch issue: **Norse is undershooting on activity, not overshooting.**

### Recommendation

- The current `threshold: 0.5` is **conservative-friendly for all three.**
- For SNNTorch, you could go higher (0.8–1.0) and still spike easily.
- For Norse, **going to 1.0 would make the spiking problem worse**. If anything, you'd want to:
  - Keep `threshold` at 0.5 (or even 0.3), OR
  - Raise `tau_mem_inv` from 50 → 100 (faster integration, larger per-step ΔV)
  - The two knobs are coupled — both effectively make spikes easier.

Keep `threshold` shared at 0.5 for the comparison study (it's a fair common ground), but document in the final MD that the operating point each framework lands on is *not equivalent in terms of spike rate*. For a truly fair comparison, you'd need to calibrate per framework to match a target spike rate (e.g. ~5% activity), then report that calibration.

---

## 2. Surrogate gradients — library defaults vs our hardcoded choice

### Current implementation: hardcoded per framework, YAML param inactive

After review, the YAML `surrogate:` param is **commented out** because it could only ever be honored by 2 of 3 frameworks (Norse has no easy way to expose it). To avoid silent inconsistency, each framework now hardcodes its surrogate explicitly:

| Framework | Hardcoded surrogate | Why this one |
|---|---|---|
| SNNTorch | `surrogate.atan()` | Matches SpikingJelly's `ATan` for fair comparison |
| Norse | `SuperSpike` (Norse library default, set via omission of `method=`) | Norse is tuned around this; no easy alternative |
| SpikingJelly | `surrogate.ATan()` | SpikingJelly's library default; mathematically same as SNNTorch's `atan` |

Each framework prints its surrogate choice at model init (e.g. `[SNNTorch] Surrogate gradient is hardcoded to 'atan' (YAML 'surrogate' inactive)`) so it's always visible what's actually running.

**To re-enable as a config param later:** wire `cfg.SURROGATE` back into [snn_torch.py](src/learning/frameworks/snn_torch.py) + [snn_spikingjelly.py](src/learning/frameworks/snn_spikingjelly.py), and add `method=` plumbing into [snn_norse.py](src/learning/frameworks/snn_norse.py) `LIFParameters`.

### Background: is SuperSpike a Norse requirement?

Neither hardcoded by us nor required by Norse — it's the Norse **library default** when `method=` isn't passed. So it's the Norse-recommended choice, not something we forced.

**Best surrogate per framework, judged by each library's own benchmarks/papers:**

| Framework | Library default | Library's own benchmark papers use | Available alternatives |
|---|---|---|---|
| **SNNTorch** | `fast_sigmoid(slope=25)` | `atan` and `fast_sigmoid` both used heavily | `atan`, `fast_sigmoid`, `straight_through_estimator`, `sigmoid`, `spike_rate_escape` |
| **Norse** | `super` (SuperSpike) | `super` exclusively in all Zenke-lineage papers (Norse's primary lineage) | `super`, `tanh`, `triangle`, `circ`, `logistic`, `heavi` |
| **SpikingJelly** | `ATan(alpha=2.0)` | `ATan` in their canonical Fang et al. 2021 paper | `ATan`, `Sigmoid`, `PiecewiseQuadratic`, `PiecewiseExp`, `S2NN`, `QPseudoSpike`, ~10 more |

### SNNTorch vs Norse contrast

- **SNNTorch's `atan`** has a *broad* gradient — it returns a non-zero gradient even when V is far from threshold. This means gradient signal flows to many neurons, including those not near firing. → Stable but slower convergence.
- **Norse's `SuperSpike`** has a *sharp* gradient that peaks tightly at V=V_th and decays fast. → Stronger updates near the decision boundary, but if LR is too high → unstable.
- SpikingJelly's `ATan` is mathematically the same as SNNTorch's `atan`, just a different implementation.

### Optimal per framework (per library's own recommendations)

- SNNTorch → `atan` for stability, `fast_sigmoid` for speed
- Norse → `super` (don't change — it's what Norse is tuned for)
- SpikingJelly → `ATan`

### For fair comparison

The closest you can get across all three is `atan` for SNNTorch + SpikingJelly, but Norse will keep using SuperSpike (ignored YAML). True parity isn't achievable without wiring `method` into Norse explicitly. **Document this as a known limitation.**

---

## 3. AMP (Automatic Mixed Precision) — am I sure?

Honest answer: **mostly yes, with one caveat to flag.**

| Framework | Library recommendation | Our finding (EXP-003) | Best setting for max performance | For fair comparison |
|---|---|---|---|---|
| **SNNTorch** | AMP supported, recommended for speed | Works fine | `use_amp: true` (faster, no accuracy loss) | `false` |
| **Norse** | No official AMP guidance; community reports gradient underflow | EXP-003: shallow 3-LIF net tolerated AMP, but deep nets fail silently | `use_amp: false` (safe default — silent failures are the worst kind) | `false` |
| **SpikingJelly** | Explicitly supports AMP, used in their own benchmarks | (not yet tested in our project) | `use_amp: true` | `false` |

### SNNTorch vs Norse contrast

- SNNTorch's `atan` produces bounded gradients that don't underflow easily in float16
- Norse's `SuperSpike` produces very small gradients far from threshold → float16 rounds these to zero → silent learning failure

### The caveat

"SNNTorch can use AMP=true" is **probably** correct for our shallow Conv-SNN, but we haven't actually verified it. EXP-003 only confirmed AMP behavior on Norse. Before flipping AMP on for SNNTorch in production, run a quick AMP-on vs AMP-off comparison on SNNTorch with our exact net.

### For fair comparison

Keep `use_amp: false` across all three — removes numerical precision as a confounding variable.

---

## 4. Loss function — playground freedom + crash-loudly on misconfig

### Current implementation

All three frameworks now read `cfg.LOSS_FN` via a `_loss(cfg)` helper. Unsupported values raise `NotImplementedError` at model init (not during training — fail fast):

- SNNTorch: handles `cross_entropy` + `mse_count`; raises on anything else
- Norse: handles `cross_entropy` only; raises on `mse_count` and other values
- SpikingJelly: handles `cross_entropy` only; raises on `mse_count` and other values

**Rationale for no silent fallback:** if you set `loss_fn: mse_count` with framework=norse, you'd want to know *immediately* that the intended loss isn't being used — not after wasting an epoch on a different loss than expected. Crash-loudly preserves the playground philosophy without hiding configuration mistakes.

### Best loss per framework (per library)

| Framework | Canonical loss from library docs/papers | Why |
|---|---|---|
| **SNNTorch** | `mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)` | Used in original SNNTorch tutorials; directly trains for target spike rates per class |
| **Norse** | Cross-entropy on summed spike counts (rate decoding) | Standard approach in Norse examples |
| **SpikingJelly** | Cross-entropy on summed spike counts | Used in Fang et al. 2021 + their tutorials |

### SNNTorch vs Norse contrast

- SNNTorch's `mse_count_loss` explicitly tells each output neuron "fire at 80% of timesteps if you're the correct class, 20% otherwise." Very direct, biologically motivated.
- Norse's cross-entropy treats spike-sum as a logit — softmax over classes, then NLL. Standard deep-learning style, framework-agnostic.

### For fair comparison

`cross_entropy` on summed spikes — the only loss available in all three with comparable semantics. The current YAML default is already correct.

---

## 5. Reset behavior — what happens to V after a spike

When V crosses the threshold and emits a spike, the voltage needs to be reset so the neuron can charge again. There are **two reset styles**:

### A) Hard reset (zero reset) — Norse + SpikingJelly default

```
After spike: V → v_reset (typically 0)
```

"Wipe the slate clean. Whatever extra voltage you had above threshold — gone."

### B) Soft reset (subtract reset) — SNNTorch default

```
After spike: V → V − threshold
```

"Only subtract the threshold; keep any overshoot voltage."

### Concrete example — threshold=1.0, V just reached 2.5

| Step | V before spike | V after spike (hard reset) | V after spike (soft reset) |
|---|---|---|---|
| t=5 | 2.5 → SPIKE | 0.0 | 1.5 |
| t=6 (assume no input) | starts at 0 | doesn't fire | **1.5 → SPIKE again!** |

### SNNTorch vs Norse contrast

- **SNNTorch (soft reset):** Large input → multiple rapid spikes encoding magnitude. The neuron acts like an "integrate-and-fire counter." Information about input strength is preserved.
- **Norse (hard reset):** Large input → still just one spike per timestep, then full reset. Information about *how much* over-threshold is **lost**.

This is **another reason `threshold` doesn't transfer 1:1** between SNNTorch and Norse — even if they had the same V-scale, the soft-vs-hard reset means SNNTorch fires more spikes per neuron per timestep for strong inputs.

### Can each library configure it?

| Framework | API | Hard reset | Soft reset |
|---|---|---|---|
| SNNTorch | `snn.Leaky(reset_mechanism="zero" or "subtract" or "none")` | ✓ | ✓ (default) |
| Norse | `LIFParameters(v_reset=value)` | ✓ (default v_reset=0) | ✗ (would need custom cell or different cell type) |
| SpikingJelly | `LIFNode(v_reset=0.0 or None)` | ✓ (default) | ✓ (pass `v_reset=None`) |

**Norse cannot easily do soft reset.** This is a hard library limitation.

### Current implementation: `reset_mode` is now a YAML param

Added to `architecture:` block in [SNN_module.yaml](SNN_module.yaml), default `zero`:

```yaml
architecture:
  threshold: 0.5
  reset_mode: zero    # zero (hard: V→0) | subtract (soft: V−=threshold); Norse only supports "zero"
```

Wiring per framework:

| Framework | Behavior |
|---|---|
| SNNTorch | Reads `cfg.RESET_MODE`, passes `reset_mechanism="zero"` or `"subtract"` to all `snn.Leaky` calls. Crashes on anything else. |
| Norse | Reads `cfg.RESET_MODE`, sets `v_reset=0.0` explicitly in `LIFParameters`. **Crashes with `NotImplementedError`** if user requests `subtract` (library limitation). |
| SpikingJelly | Reads `cfg.RESET_MODE`, maps `zero` → `v_reset=0.0`, `subtract` → `v_reset=None` (SpikingJelly's soft-reset signal). Crashes on anything else. |

Each framework prints its reset mode at model init.

| Setting | What it does | Behavior |
|---|---|---|
| `reset_mode: zero` (default) | Hard reset across all three | Works in all three — **fair comparison setting** |
| `reset_mode: subtract` | Soft reset in SNNTorch + SpikingJelly | Norse **crashes** at init with clear message — by design |

### Best per framework (max performance)

- SNNTorch → `subtract` (its native default — designed around this)
- Norse → `zero` (only option)
- SpikingJelly → `zero` (their default in benchmarks)

---

## 6. Learning rate — surrogate gradient shape interaction

The relationship between surrogate gradient shape and learning rate.

### The mechanism

- **SuperSpike (Norse):** gradient = `1 / (α·|V − V_th| + 1)²`, with α default ≈ 100. This peaks **sharply** at V=V_th. Near the threshold, gradients are large; far from it, they're tiny.
- **atan (SNNTorch / SpikingJelly):** gradient is **broader**, with smaller peak but non-zero over a wider V range.

### Effect on training

| Framework | Surrogate | Gradient magnitude near V_th | Recommended LR |
|---|---|---|---|
| SNNTorch | atan | Moderate, broad | `1e-3` (standard) |
| Norse | SuperSpike | **High, narrow** | `1e-3` may oscillate; try `5e-4` or `1e-4` |
| SpikingJelly | ATan | Moderate, broad | `1e-3` (standard) |

### SNNTorch vs Norse contrast

- SNNTorch with `atan` + `lr=1e-3` is a battle-tested combination from the official tutorials. Stable.
- Norse with `SuperSpike` + `lr=1e-3` is sometimes too aggressive — the gradient near threshold is so large that Adam takes oversized steps, causing oscillation or divergence. Many Norse examples in the wild use `lr=2e-4` or `lr=5e-4`.

### Practical recommendation

- For fair comparison: keep `learning_rate: 0.001` across all three (current setting).
- For best individual performance: if Norse undershoots at lr=1e-3 (which matches our current observation of ~9% first epoch), try a sweep at `5e-4` and `1e-4` for Norse only. Document the per-framework optimal in the final comparison MD.

---

## Summary table

| Truly shared (set once) | Framework-sensitive (review per framework) | Framework-exclusive (each in its own section) |
|---|---|---|
| All `architecture.*` (geometry) | `threshold` (same name, different scale) | `beta` (SNNTorch) |
| `timesteps`, `batch_size` | `learning_rate` (Norse may need lower) | `tau_mem_inv` (Norse) |
| `weight_decay`, `optimizer` | `use_amp` (must be false for Norse) | `tau` (SpikingJelly) |
| `num_classes`, `device`, `num_workers` | `surrogate` (Norse ignores it) | |
| `grad_accum_steps`, `lr_scheduler` | `loss_fn` (`mse_count` is SNNTorch-only) | |
| | `reset_mode` (Norse only supports hard reset) | |

The current YAML structure (per-framework sections under `frameworks:`) handles the third column correctly. The middle column is where the comparison-fairness debate lives.

---

## Quick-reference equivalence table

If you ever want to roughly match dynamics across frameworks (NOT a substitute for tuning, but a starting point):

| What it does | SNNTorch | Norse | SpikingJelly |
|---|---|---|---|
| Per-step decay factor for V | `beta = 0.95` | `1 − dt·τ_mem_inv = 0.95` (dt=1ms, τ_inv=50) | `1 − 1/τ = 0.5` (τ=2) |
| Per-step input gain | `1.0` (full input added) | `dt·τ_mem_inv = 0.05` (input attenuated 20×) | `1/τ = 0.5` (input attenuated 2×) |
| Effective V_eq for sustained input I | `20·I` | `I` | `I` |
| Default reset | subtract threshold (soft) | hard reset to 0 | hard reset to 0 |
| Default surrogate | `fast_sigmoid(slope=25)` | `super` (SuperSpike) | `ATan(alpha=2.0)` |
| Surrogate (current YAML `atan`) | atan ✓ | **ignored** (SuperSpike) | ATan ✓ |
| AMP-safe in float16 | ✓ (atan gradients bounded) | ✗ (SuperSpike underflows) | ✓ |
| Canonical loss | `mse_count_loss` | cross-entropy on spike sum | cross-entropy on spike sum |
| Recommended LR (with default surrogate) | `1e-3` | `2e-4` to `5e-4` | `1e-3` |

### Conclusion

SNNTorch's `beta` and Norse's `tau_mem_inv·dt` happen to produce similar V-decay coefficients (~0.95) with the current YAML — but the input-gain term differs by 20×, which is what breaks threshold portability. SpikingJelly's `tau=2.0` is in a different regime altogether (fast integration, low retention).

Beyond threshold, the key portability gotchas are:
1. **Reset behavior** differs (SNNTorch soft vs Norse/SpikingJelly hard) — Norse can't even do soft.
2. **Surrogate gradient** is silently ignored by Norse (always uses SuperSpike).
3. **AMP** is unsafe with Norse's default SuperSpike (silent gradient underflow).
4. **Loss `mse_count`** only works in SNNTorch.
5. **Learning rate** sensitivity differs because surrogate gradient shape differs.

For the final comparison MD: pick settings that work in **all three** as the "fair comparison" baseline, then separately report each framework's best-individual-performance setting in a per-framework appendix.

---

## Final reference: optimal per framework + fair comparison baseline

### Table A — Best parameters for each framework's individual peak performance

These are the settings each library is designed and benchmarked around. Use these when you want **each framework to perform at its best** (not for cross-framework comparison).

| Parameter | SNNTorch (best for itself) | Norse (best for itself) | SpikingJelly (best for itself) |
|---|---|---|---|
| `threshold` | `0.5–1.0` (tolerates higher due to 20× V amplification) | `0.3–0.5` (lower needed — no input amplification) | `0.5–1.0` (no amplification but soft-reset-capable) |
| Neuron decay param | `beta: 0.95` | `tau_mem_inv: 50–100` (raise to 100 if undershooting) | `tau: 2.0` |
| `surrogate` (YAML inactive) | hardcoded `atan` | hardcoded `SuperSpike` (Norse default) | hardcoded `ATan` |
| `loss_fn` | `mse_count` (SNNTorch's canonical loss — config-driven) | `cross_entropy` (only option — crashes on others) | `cross_entropy` (only option — crashes on others) |
| `reset_mode` | `subtract` (soft — SNNTorch's native default) | `zero` (only option — library limitation) | `zero` (their benchmark default) |
| `learning_rate` | `1e-3` (battle-tested with `atan`) | `2e-4` to `5e-4` (lower needed for SuperSpike's sharp gradient) | `1e-3` |
| `use_amp` | `true` (faster, no accuracy loss — but verify on your net first) | `false` (silent gradient underflow with SuperSpike in float16) | `true` (library-supported, used in their benchmarks) |
| `optimizer` | `adam` | `adam` | `adam` |
| `weight_decay` | `1e-4` | `1e-4` | `1e-4` |

### Table B — Fair comparison baseline (level playing field for all three)

These are the settings where all three frameworks behave as similarly as possible. Use these when you want to **compare frameworks against each other** on equal terms.

| Parameter | Fair comparison value | Why this value | Limitations / caveats |
|---|---|---|---|
| `threshold` | `0.5` | Conservative-friendly across all three; reachable by all without huge weight inflation | **Spike rates will differ** at the same threshold (SNNTorch fires more easily due to 20× V amplification + soft reset). True parity needs per-framework calibration to a target spike rate (~5% activity). |
| Neuron decay param | per-framework defaults (`beta=0.95`, `tau_mem_inv=50`, `tau=2.0`) | Each framework's "stock" tuning so we don't favor one | These produce **different dynamics** — coordinate systems are not convertible. Document the difference explicitly. |
| `surrogate` (hardcoded) | `atan` in SNNTorch + SpikingJelly; `SuperSpike` in Norse | Each framework explicitly prints what it's using; YAML param inactive | **Norse uses SuperSpike, not atan** — not fully equal across all three. True parity needs `method=` wired into Norse's `LIFParameters`. |
| `loss_fn` | `cross_entropy` | Only loss available with comparable semantics in all three | SNNTorch could perform better with `mse_count`, but that's not available in Norse/SpikingJelly. Trade fairness for SNNTorch peak performance. Setting `mse_count` on Norse/SpikingJelly **crashes at init** (by design). |
| `reset_mode` | `zero` (hard reset) | Only mode supported by all three | SNNTorch is forced off its native `subtract` mode — slight performance loss for SNNTorch. **Norse limitation:** can't do soft reset at all (crashes on `subtract`). |
| `learning_rate` | `1e-3` | Standard for all three with their default optimizers | Norse may oscillate or undershoot — if so, document as a known cross-framework difference rather than tuning per-framework. |
| `use_amp` | `false` | Removes numerical precision as a confounding variable | Slower training for SNNTorch + SpikingJelly (~1.5–2× slower than AMP=true). Accept the speed cost for fairness. |
| `optimizer` | `adam` | Default in all three; well-understood behavior | None — true shared parameter. |
| `weight_decay` | `1e-4` | Mild regularization, framework-agnostic | None — true shared parameter. |
| `epochs`, `timesteps`, `batch_size` | identical across runs | Trivially comparable | None. |
| `architecture.*` (conv/pool/FC) | identical across runs | Same network topology = the only way comparison makes sense | None. |

### Known limitations that cannot be fully equalized

These are **hard library limitations** that prevent perfect framework parity, no matter what YAML setting you choose:

| Limitation | Affected framework | Impact on comparison | Mitigation |
|---|---|---|---|
| `surrogate` YAML param removed (hardcoded per framework) | **All three** | Surrogate is no longer configurable via YAML — SNNTorch + SpikingJelly are hardcoded to `atan`/`ATan`, Norse to `SuperSpike`. Decision made to avoid silent inconsistency (was: SNNTorch + SpikingJelly read YAML, Norse ignored it). | To restore as a YAML param, wire `cfg.SURROGATE` back into all three framework files AND add `method=` plumbing into Norse `LIFParameters`. |
| Soft reset not supported | **Norse** | Can't replicate SNNTorch's spike-counter behavior for strong inputs. Hard reset everywhere is the only fair option. | None — accept the limitation. Use `zero` reset everywhere for fairness. |
| `mse_count_loss` not available | **Norse, SpikingJelly** | SNNTorch's canonical loss is unavailable elsewhere. Fair comparison must use `cross_entropy`. | None — accept the limitation. |
| AMP gradient underflow with SuperSpike | **Norse** | Can't speed up Norse training with mixed precision. Forces AMP=false everywhere for fairness. | None known. Could try float32 master weights with float16 forward, but not wired up. |
| Neuron decay parameter coordinate systems | **All three** | `beta`, `tau_mem_inv`, `tau` are not directly convertible — same numerical value means different dynamics. | Tune each independently; report all three values; never claim they're "equivalent." |
| Default reset values differ even within "hard reset" | All three | Norse `v_reset=0`, SpikingJelly `v_reset=0`, SNNTorch `subtract` (different by default) | Set explicitly in YAML rather than relying on defaults. |

### How to present this in the final comparison MD

Suggested structure for your end-deliverable:

1. **Header section:** "Fair comparison baseline" — show Table B settings, explain why each was chosen, and call out the known limitations table.
2. **Results section:** Train all three frameworks with Table B settings, report accuracy/loss/spike-rate side by side.
3. **Per-framework appendix:** For each of SNNTorch / Norse / SpikingJelly, show:
   - Table A's row for that framework
   - Re-train with those settings
   - Report the "best individual" accuracy
   - Calculate the gap between "fair comparison" and "best individual" — this tells the reader how much each framework was held back by the fairness constraint.

This separation lets readers see both (a) which framework is fundamentally best for this task and (b) which framework loses the least when forced into a common configuration.
