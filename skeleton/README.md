# skeleton

Settings, seeding and the files each run writes.

| File | Role |
|---|---|
| `config.py` | `Settings`: reads the three files in `configuration/` (plus an optional `--config` overlay), the command-line flags (`--config`, `--framework`, `--seed`, `--cache-root`), and the neuron-parameter helpers the LIF adapters use |
| `seeding.py` | Seeds weights and data order so every framework starts from the same weights |
| `results_collection.py` | Build and write `runs.csv`, `epochs.csv`, `layers.csv` and the per-run JSON |
| `snn_logging.py` | Logging setup |
