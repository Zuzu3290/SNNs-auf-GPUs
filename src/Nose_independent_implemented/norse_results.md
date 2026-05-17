# Norse SNN — MNIST Results

## Model Architecture

```
Input Layer      784 neurons      one per pixel (28×28 flattened)
      ↓
Linear (fc1)     784 → 256
      ↓
LIF Layer 1      256 hidden spiking neurons
      ↓
Linear (fc2)     256 → 10
      ↓
LIF Layer 2      10 output spiking neurons  (one per digit class)
      ↓
Sum spikes over T=16 timesteps → argmax → predicted digit
```

## Configuration

| Parameter | Value |
|---|---|
| Framework | Norse 1.1.0 |
| Backend | PyTorch 2.11.0 |
| Device | CPU |
| Timesteps (T) | 16 |
| Input encoding | Poisson spike trains |
| Hidden neurons | 256 |
| Output neurons | 10 |
| Batch size | 64 |
| Epochs | 3 |
| Learning rate | 0.001 |
| Optimiser | Adam |
| Loss function | Cross-Entropy |
| Gradient method | Surrogate gradients (Norse built-in) |

## Training Performance

| Epoch | Avg Loss | Train Accuracy |
|---|---|---|
| 1 | 0.3778 | 88.70% |
| 2 | 0.1877 | 94.20% |
| 3 | 0.1360 | 95.79% |

## Test Performance

| Metric | Value |
|---|---|
| Test Accuracy | **95.87%** |
| Test samples | 10,000 |
| Training samples | 60,000 |

## Spike Raster Observation (digit `7`)

| Layer | Observation |
|---|---|
| Hidden (256 neurons) | Irregular sparse firing across all 16 timesteps |
| Output neuron 7 | Fired most frequently — correct prediction |
| Output neuron 3 | Fired occasionally — closest distractor |
| Output neurons 0,1,2,4,5,6,8,9 | Silent — confidently rejected |

## Notes

- Accuracy improves rapidly: **+5.5% from epoch 1 to 2**, then **+1.6% from epoch 2 to 3**
- Loss more than halved each epoch, indicating healthy gradient flow through surrogate gradients
- No GPU used — training ran entirely on CPU
- A 4th epoch would likely push accuracy past **96%**
