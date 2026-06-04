# Config Wiring

All framework parameters, neuron types, and architecture dimensions flow from two YAML files through a single `Settings` object. No Python file hardcodes a threshold, beta, or kernel size — every tunable value is declared in YAML and read once at startup.

---

## YAML Files

| File | Purpose |
|---|---|
| `configuration/SNN_module.yaml` | Training params, per-framework neuron/optimizer/loss params, dataset and output paths |
| `configuration/network_architecture.yaml` | Conv dimensions, neuron type selection per layer per framework |

Both are loaded by `skeleton/snn_config.py` at import time.

---

## How `snn_config.py` Wires Everything

`Settings.__init__()` reads both YAMLs and exposes everything as flat attributes (`cfg.EPOCHS`, `cfg.CONV1_OUT`, etc.) plus two structured dicts:

### `FRAMEWORK_CFG`

Holds the per-framework param block for all three frameworks, always populated regardless of which one is active:

```python
cfg.FRAMEWORK_CFG = {
    "snntorch":     { "beta": 0.95, "threshold": 0.5, "optimizer": "adam", "loss_fn": "mse_count" },
    "norse":        { "tau_mem_inv": 100.0, "threshold": 0.5, "optimizer": "adam", "loss_fn": "cross_entropy" },
    "spikingjelly": { "tau": 2.0, "threshold": 0.5, "optimizer": "adam", "loss_fn": "cross_entropy" },
}
```

### `FW_TO_CFG_KEY`

`training.framework` in YAML uses short keys (`torch`, `norse`, `sj`), but `FRAMEWORK_CFG` and `NEURON_TYPES` use full library names. The module-level mapping resolves this:

```python
FW_TO_CFG_KEY = { "torch": "snntorch", "norse": "norse", "sj": "spikingjelly" }
```

### `active_fw_cfg` property

Returns `FRAMEWORK_CFG[FW_TO_CFG_KEY[self.FRAMEWORK]]` — the config dict for whichever framework is currently selected in the YAML. Framework files and the trainer always use this instead of indexing `FRAMEWORK_CFG` directly.

### `NEURON_TYPES`

Loaded from `network_architecture.yaml`. Keyed by full library name, then by layer slot (`lif1`, `lif2`, `lif_out`):

```yaml
neuron_types:
  norse:
    lif1:    lif_cell
    lif2:    lif_cell
    lif_out: lif_cell
```

Each framework's layer builder reads from `cfg.NEURON_TYPES.get("<lib_name>", {}).get("<layer>", "<default>")`.

---

## How Framework Files Consume the Config

Every framework `__init__` builds a local `fw_cfg` dict by merging the framework block with the shared training params:

```python
fw_cfg = {
    **cfg.FRAMEWORK_CFG["norse"],   # tau_mem_inv, threshold, optimizer, loss_fn
    "learning_rate": cfg.LEARNING_RATE,
    "weight_decay":  cfg.WEIGHT_DECAY,
}
self.optimizer = build_optimizer(self.parameters(), fw_cfg)
self.loss_fn   = build_loss(fw_cfg, framework="norse")
```

Layer builders (`build_norse_layer`, `build_lif_layer`, `build_sj_layer`) live in their respective framework files and read directly from `cfg.FRAMEWORK_CFG["<lib_name>"]` and `cfg.NEURON_TYPES`.

---

## Switching Frameworks

Set `training.framework: norse` (or `torch` / `sj`) in `configuration/SNN_module.yaml`. `main.py` reads `cfg.FRAMEWORK` and selects the model class — no CLI flags, no Python edits needed.

---

## FC_IN Auto-Computation

The flattened size after both conv+pool stages is computed from conv dimensions, never hardcoded:

```python
h = (SENSOR_H - CONV1_KERNEL + 1) // POOL_KERNEL
h = (h - CONV2_KERNEL + 1) // POOL_KERNEL
FC_IN = CONV2_OUT * h * h   # → 800 for N-MNIST defaults
```

Any change to sensor size or kernel size in `network_architecture.yaml` automatically propagates to the linear layer.
