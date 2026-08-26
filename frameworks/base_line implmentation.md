“If the frameworks already expose the relevant LIF parameters and implementation details through their APIs, can we use those exposed mechanisms to construct an equivalent baseline LIF configuration across the four frameworks and then determine, through a controlled diagnostic experiment, whether any observable differences remain?"

Yes. If this is meant to become an **executable research task**, I would constrain it very tightly. The objective should not be to investigate the frameworks generally, reverse-engineer their internals, or evaluate their hardware behavior. It should be a controlled **LIF equivalence experiment**.

## Task: Cross-framework LIF equivalence under a common physical baseline

### 1. Objective

Determine whether the four framework implementations produce equivalent LIF dynamics when configured to represent the **same physical baseline**:

* SpikingJelly
* Norse
* snnTorch
* Sinabs

The experiment should answer only:

> **When the four frameworks are given an equivalent LIF configuration and identical input conditions, how closely do their membrane-voltage and spike dynamics agree, and what implementation-level differences explain any remaining divergence?**

Nothing beyond this question should be investigated.

---

## 2. Fix the baseline first

Use one fixed physical reference:

[
\tau_{mem}=10,ms
]

[
\Delta t=1,ms
]

Then define the remaining parameters explicitly and keep them identical wherever the frameworks permit:

* (V_{reset})
* (V_{th})
* initial membrane potential
* input current
* simulation duration
* numerical precision
* batch/input shape.

**Do not optimize these parameters separately for each framework.**

The purpose is equivalence, not finding the best-performing configuration for each library.

---

## 3. Translate the baseline into each framework

Use the framework's **documented native parameters** rather than attempting to reconstruct or modify its implementation.

The initial mapping is:

| Framework    | Native parameterization                |
| ------------ | -------------------------------------- |
| SpikingJelly | `tau = 10.0`                           |
| Norse        | `tau_mem_inv = 100`, `dt = 0.001`      |
| snnTorch     | (\beta=e^{-\Delta t/\tau}\approx0.905) |
| Sinabs       | `tau_mem = 10.0`                       |

These values should be **verified against the exact versions of the libraries used in the experiment** before implementation.

The important principle is:

> **Physical baseline → documented framework parameter**

not:

> Framework A → manually altered parameters → Framework B.

---

# 4. Resolve the Norse asymmetry before running the comparison

This is the **one structural issue that must be explicitly controlled**.

Norse's default LIF implementation contains:

[
i(t) \rightarrow v(t)
]

with a separate synaptic-current state, whereas the other implementations can operate with direct input-to-membrane integration.

Therefore, you must choose **one** of these two experimental conditions **before implementation**:

### Condition A — Strict equation equivalence

Configure Norse so that its synaptic-current contribution approximates direct input injection.

This makes the comparison:

> **same intended LIF equation across frameworks**

### Condition B — Native implementation comparison

Leave Norse's additional current state intact.

Then the experiment explicitly asks:

> **How does the native implementation of an otherwise similarly parameterized LIF differ across frameworks?**

For the primary experiment, I would use **Condition A** if the goal is equivalence.

If you later want to document the effect of Norse's extra state, that can be a **secondary controlled test**, but it should not be mixed into the primary experiment.

---

# 5. Use one identical input

Do not begin with a neural network.

Start with **one LIF neuron**.

Feed every implementation exactly the same deterministic input sequence:

[
I[0],I[1],...,I[T]
]

For example, use a controlled constant or step current that produces several spikes during the simulation.

The input must be identical in:

* magnitude
* duration
* timestep
* sequence
* datatype.

This eliminates network architecture and training as confounding variables.

---

# 6. Record only the quantities relevant to LIF equivalence

For every timestep, record:

### Primary measurements

[
V(t)
]

Membrane potential.

[
S(t)
]

Spike output.

### Derived measurements

* spike times
* spike count
* firing rate
* maximum voltage deviation
* voltage-trace error relative to the reference.

Do **not** start measuring:

* classification accuracy
* training performance
* energy consumption
* GPU performance
* neuromorphic hardware performance
* surrogate-gradient quality
* network-level accuracy
* framework speed.

Those are outside the scope of this task.

---

# 7. Establish a mathematical reference

Do not use one framework as the “ground truth.”

Implement the canonical LIF equation independently:

[
\tau_{mem}\frac{dV}{dt}
=======================

-(V-V_{reset})+I
]

with the same:

[
\tau_{mem}=10,ms,\qquad \Delta t=1,ms
]

and the same reset and threshold rules.

This gives you:

**Reference LIF → expected trajectory**

Then compare:

**Reference → SpikingJelly**
**Reference → Norse**
**Reference → snnTorch**
**Reference → Sinabs**

This is important because otherwise you are only determining whether the frameworks agree with one another, rather than whether they agree with the specified LIF model.

---

# 8. Compare the trajectories

For each framework, calculate the deviation from the reference.

For example:

[
E_V =
\frac{1}{T}
\sum_{t=1}^{T}
|V_{framework}(t)-V_{reference}(t)|
]

You can also calculate:

* maximum absolute voltage error
* RMS voltage error
* spike-time difference
* spike-count difference.

Then create one direct comparison table:

| Framework    | Voltage error | Max error | Spike-count difference | Spike-time difference |
| ------------ | ------------: | --------: | ---------------------: | --------------------: |
| SpikingJelly |               |           |                        |                       |
| Norse        |               |           |                        |                       |
| snnTorch     |               |           |                        |                       |
| Sinabs       |               |           |                        |                       |

---

# 9. Only after observing divergence, diagnose it

This is an important boundary.

Do **not** begin by searching for every possible difference between the frameworks.

First establish:

> **Is there actually a measurable divergence?**

If there is, then investigate only the specific implementation property capable of explaining that observed divergence.

For example:

### Observed voltage divergence

Investigate:

* discretization equation
* decay formulation
* timestep treatment.

### Observed spike-timing divergence

Investigate:

* threshold application
* reset ordering
* timestep discretization.

### Norse-specific divergence

Investigate:

* separate synaptic-current state.

This keeps the investigation causal rather than turning it into a general framework survey.

---

# 10. The final result should have only three levels

Your experiment should ultimately produce:

### Level 1 — Parameter equivalence

> Can the same physical LIF specification be represented in all four frameworks?

### Level 2 — Behavioral equivalence

> When configured equivalently, do they produce the same (V(t)) and spike output?

### Level 3 — Residual explanation

> If they do not, which documented implementation difference accounts for the observed deviation?

That is the entire research task.

---

## What is explicitly **out of scope**

To keep the project from expanding, **do not investigate**:

* NIR as a research topic in itself
* neuromorphic hardware
* hardware deployment
* training algorithms
* surrogate gradients unless they directly affect the measured inference dynamics
* network architectures
* learning performance
* energy efficiency
* GPU/CPU benchmarking
* framework popularity
* general framework comparisons
* biological realism beyond the chosen LIF equation
* discovering undocumented/internal mechanisms
* rewriting framework implementations
* inventing new conversion equations where the documented API already provides the required parameterization.

### The guiding rule

> **If a factor does not affect whether the four implementations reproduce the same specified LIF dynamics under the controlled experiment, it is not part of this study.**

That boundary is what makes this executable rather than turning into a broad investigation of four SNN frameworks.
