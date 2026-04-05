import torch
import torch.nn as nn
import snn_cuda.snn_forward as snn_cuda

class SNNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, v_th: float = 1.0, tau: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.v_th = v_th
        self.tau = tau
        self.register_buffer("v", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  ; assume C is in_features
        B, C, T = x.shape
        out_features = self.weight.size(0)

        # Linear transform to spike input
        inp = torch.einsum("bct,on->bnt", x, self.weight)  # [B, out_features, T]

        # Ensure CUDA
        if not inp.is_cuda:
            inp = inp.cuda()
            self.v = self.v.cuda()

        # cuda kernel works on [B, N, T] and current v [B, N]
        v_batch = self.v.unsqueeze(0).expand(B, -1)  # [B, N]

        # Run the CUDA SNN forward pass
        spikes = snn_cuda.snn_forward_cuda(
            inp,
            v_batch,
            v_th=self.v_th,
            tau_inv=1.0 / self.tau,
        )

        # For training, you usually want to detach the state update
        # and keep v as a module buffer
        with torch.no_grad():
            self.v.data = v_batch[0]  # just keep first sample for module state

        return spikes  # [B, N, T]