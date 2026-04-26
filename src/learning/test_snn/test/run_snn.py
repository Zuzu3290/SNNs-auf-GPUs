"""

Entry point to train and evaluate the SNN on the MNIST CSV dataset.

"""

from snn_config import Settings
from snn_torch_module import build_and_run
from Utils_snntorch import LearningMode

# ── Config ────────────────────────────────────────────────────────────────────
cfg = Settings()

# ── Path to your MNIST CSV file ───────────────────────────────────────────────
CSV_PATH = "mnist_train_small.csv"   # put this file in the same folder

# ── Run supervised training ───────────────────────────────────────────────────
# net, metrics = build_and_run(
#     cfg,
#     csv_path=CSV_PATH,
#     mode=LearningMode.SUPERVISED,
#     loss_name="mse_count",     # options: mse_count | ce_count | ce_rate | mse_membrane | l1_membrane
#     neuron_type="leaky",       # options: leaky | synaptic | rleaky | lapicque
#     encoding="rate",           # options: rate | latency
# )

#── Run unsupervised training (STDP) — uncomment to use ──────────────────────
net, metrics = build_and_run(
    cfg,
    csv_path=CSV_PATH,
    mode=LearningMode.UNSUPERVISED,
    loss_name="stdp_trace",
    neuron_type="leaky",
    encoding="rate",
)
