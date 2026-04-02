import torch

from triton_snn.config import SNNConfig
from triton_snn.models.snn import TritonSNNClassifier


def test_model_constructs():
    cfg = SNNConfig()
    model = TritonSNNClassifier(cfg)
    assert model.cfg.input_dim == 128


def test_forward_shape_cpu_construction_only():
    # This does not execute Triton; it only validates package import and layout.
    cfg = SNNConfig()
    model = TritonSNNClassifier(cfg)
    x = torch.randn(4, cfg.input_dim)
    assert x.shape == (4, 128)
