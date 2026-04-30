# =============================================================
# config.py — Central configuration for the Norse SNN system
# =============================================================
# This is the ONLY file you need to change between experiments.
# Every other module reads from this dictionary.

CONFIG = {
    # ── Neuron type ───────────────────────────────────────────
    # Options:
    #   "LIF"          Basic Leaky Integrate-and-Fire
    #   "LIFAdEx"      Adaptive Exponential LIF (threshold adapts after firing)
    #   "LIFRecurrent" LIF with within-layer recurrent connections
    "neuron_type": "LIF",

    # ── Neuron parameters ─────────────────────────────────────
    # tau_mem: membrane time constant — how fast voltage leaks
    #   higher = slower leak, neuron remembers input longer
    #   lower  = faster leak, neuron forgets quickly
    "tau_mem": 0.02,

    # threshold: voltage level required to fire a spike
    #   higher = harder to fire, fewer spikes (more selective)
    #   lower  = fires easily, many spikes (more active)
    "threshold": 1.0,

    # ── Surrogate gradient function ───────────────────────────
    # Used during backprop to approximate gradient through spikes.
    # Options:
    #   "super"      SuperSpike — smooth sigmoid-based (default)
    #   "heaviside"  Step function approximation
    #   "boxcar"     Box-shaped approximation
    "surrogate": "super",

    # ── Architecture ─────────────────────────────────────────
    "num_layers":  1,     # number of hidden LIF layers
    "hidden_size": 256,   # neurons per hidden layer
    "output_size": 10,    # one per digit class (fixed for N-MNIST)

    # ── Spike decoding ────────────────────────────────────────
    # How output spikes are converted into a final prediction.
    # Options:
    #   "rate"        sum all spikes over T → highest count wins
    #   "max"         did the neuron fire at least once? (0 or 1)
    #   "first_spike" which neuron fired first = most confident
    "decoding": "rate",

    # ── Temporal ─────────────────────────────────────────────
    # T = how many time windows to bin events into
    # Higher T = finer time resolution but slower training
    "timesteps": 16,

    # ── Regularization ───────────────────────────────────────
    "dropout_p":    0.0,    # dropout probability (0.0 = off)
    "weight_decay": 1e-4,   # L2 penalty on weights

    # ── Training ─────────────────────────────────────────────
    "epochs":     5,
    "batch_size": 64,
    "lr":         1e-3,

    # optimizer options: "adam", "sgd", "adamw"
    "optimizer": "adam",

    # ── Dataset / Paths ──────────────────────────────────────
    "dataset":     "nmnist",
    "data_dir":    "./data",
    "results_dir": "src/Learning/snn_norse/results",
}
