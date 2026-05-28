# Skeleton

The `skeleton/` folder provides the basic configuration and utility layer for the SNN project.

It acts as a bridge between the model configuration, GPU runtime diagnostics, and internal logging setup. These files help prepare the system for training, monitoring, and debugging.

## Contents

| File | Description |
|---|---|
| `snn_config.py` | Loads the SNN configuration from a YAML file and prepares model, training, dataset, compiler, and output settings. |
| `gpu_stats.py` | Tracks GPU utilization and memory usage during training. |
| `snn_logging.py` | Provides a simple logging setup for debugging, training updates, and runtime messages. |

## Purpose

The purpose of this folder is to keep the core setup utilities in one place.
It supports:

- SNN model configuration
- Training parameter setup
- Dataset path configuration
- Compiler setting control
- GPU usage monitoring
- Runtime performance tracking
- Internal logging and debugging

## Role in the Project

The `skeleton/` folder helps connect the configuration files with the learning and runtime workflow.
It provides the necessary setup layer before model training begins and helps monitor how the system behaves during execution.