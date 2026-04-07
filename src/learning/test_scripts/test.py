import torch
import torch.nn as nn
import snn_cuda.snn_forward as snn_cuda
from learning.snn_torch_module import SNNLayer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

layer = SNNLayer(in_features=10, out_features=5, v_th=1.0, tau=10.0).to(device)

x = torch.randn(2, 10, 20, device=device, requires_grad=True)  # [B, C, T]

spikes = layer(x)

print("Input shape:", x.shape)
print("Spike shape:", spikes.shape)
print("Mean spikes:", spikes.mean().item())