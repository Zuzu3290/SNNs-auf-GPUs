# skeleton

Settings, seeding and the files each run writes.

| File | Role |
|---|---|
| `config.py` | `Settings`: reads the three files in `configuration/`, the `--framework` flag, and the neuron-parameter helpers the LIF adapters use |
| `seeding.py` | Seeds weights and data order so every framework starts from the same weights |
| `results_collection.py` | Build and write `training_runs.csv`, `inference_runs.csv`, `epochs.csv`, `layers.csv` and the per-run JSON |
