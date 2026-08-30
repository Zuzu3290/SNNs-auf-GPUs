# Configuration

All tunable parameters for the SNN project live here. No Python file contains hardcoded training values — everything is declared in these three files and read once at startup by `skeleton/snn_config.py` and `event_data_workflow/workflow_config.py`.

---

## Files

| File | Controls | Read by |
|---|---|---|
| `SNN_module.yaml` | Framework selector, training loop, regularization, per-framework neuron params, dataset and output paths | `skeleton/snn_config.py` → `Settings` |
| `network_architecture.yaml` | Conv-SNN layer dimensions, neuron type per layer per framework | `skeleton/snn_config.py` → `Settings` |
| `data_workflow.yaml` | Event binning mode, temporal slicing, cache strategy | `event_data_workflow/workflow_config.py` → `WorkflowSettings` |

---

## Typical Workflow

1. Set `training.framework` in `SNN_module.yaml` to `norse`, `torch`, or `sj`.
2. Adjust the matching `frameworks.<name>` block (threshold, tau, beta).
3. If needed, change neuron types per layer in `network_architecture.yaml`.
4. Set binning mode in `data_workflow.yaml` (`n_time_bins` for training, `time_window` for analysis).
5. Run via `launch.sh` (Linux/Colab) or `launch.bat` (Windows) — no CLI flags needed.

See individual README files in this folder for full parameter documentation.
