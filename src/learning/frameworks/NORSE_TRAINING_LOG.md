# Norse SNN — Training Performance Log

Dataset: N-MNIST | Sensor: 34×34×2 | Framework: Norse v1.1.0

---

## Run 001

**Date:** 2026-05-14
**Loss function:** `SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)` (SNNTorch)
**Config:**
- Epochs: 10 | Iterations/epoch: 100 | Timesteps: 25
- Batch size: 128 | LR: 0.001 | Weight decay: 0.0001
- Beta: 0.95 → tau_mem_inv ≈ 51.3 Hz | Threshold (v_th): 0.5
- Device: CUDA | Network structure: S → [50, 50, 50, 50, 50, 50, 10]

**Notes:** First run. Accuracy peaks at epoch 4 then degrades — loss landscape is unstable with mse_count_loss. Network structure [50,50,...] is the MLP config from YAML, not the CNN used by the model (CNN is hardcoded in _NorseNet).

| Epoch | Train Loss | Train Accuracy | Duration (s) | Checkpoint |
|-------|-----------|----------------|--------------|------------|
| 1     | 12.2486   | 22.28%         | 671.65       | ✓ (best)   |
| 2     | 8.4436    | 53.27%         | 609.76       | ✓ (best)   |
| 3     | 8.0034    | 61.15%         | 559.85       | ✓ (best)   |
| 4     | 7.9639    | 61.98%         | 517.71       | ✓ (best)   |
| 5     | 8.0668    | 58.90%         | 488.94       |            |
| 6     | 8.1657    | 57.10%         | 459.53       |            |
| 7     | 8.2237    | 57.16%         | 441.76       |            |
| 8     | 11.7746   | 37.09%         | 426.87       |            |
| 9     | 10.2137   | 43.95%         | 434.94       |            |
| 10    | 8.1598    | 58.59%         | 448.43       |            |

**Best accuracy:** 61.98% (epoch 4)
**Final accuracy:** 58.59%
**Observation:** Loss spikes at epoch 8 (11.77) suggesting instability. mse_count_loss enforces fixed spike rate targets (80%/20%) which constrains the gradient and appears to cause oscillation after epoch 4.

---

## Run 002

**Date:**
**Loss function:** `CrossEntropyLoss` on summed spikes — `F.cross_entropy(spk_rec.sum(0), targets)`
**Config:**
- Epochs: | Iterations/epoch: | Timesteps:
- Batch size: | LR: | Weight decay:
- Beta: → tau_mem_inv Hz | Threshold (v_th):
- Device: | Network structure:

**Notes:**

| Epoch | Train Loss | Train Accuracy | Duration (s) | Checkpoint |
|-------|-----------|----------------|--------------|------------|
|       |           |                |              |            |

**Best accuracy:**
**Final accuracy:**
**Observation:**

---

<!-- Copy the block below for each new run -->
<!--
## Run 00X

**Date:**
**Loss function:**
**Config:**
- Epochs: | Iterations/epoch: | Timesteps:
- Batch size: | LR: | Weight decay:
- Beta: → tau_mem_inv Hz | Threshold (v_th):
- Device: | Network structure:

**Notes:**

| Epoch | Train Loss | Train Accuracy | Duration (s) | Checkpoint |
|-------|-----------|----------------|--------------|------------|
|       |           |                |              |            |

**Best accuracy:**
**Final accuracy:**
**Observation:**
-->
