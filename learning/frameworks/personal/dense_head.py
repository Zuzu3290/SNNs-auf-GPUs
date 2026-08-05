"""
Shared dense (per-pixel) output head for regression models — DSEC's optical-flow
target is (H, W, 2) [+1 validity channel, not predicted], not the flat pose-like
vector the original MVSEC/TUM-VIE regression heads were built for (those two are
removed; DSEC is the only regression dataset left, so the flat-vector head has no
consumer anymore — this replaces it).

Pure PyTorch (nn.ConvTranspose2d/Upsample/Conv2d) — framework-agnostic, shared
across all four snn_*_regression.py variants. Only the LIF layers themselves
differ per framework; the conv/decode plumbing around them doesn't.
"""
import torch.nn as nn
import torch.nn.functional as F
from skeleton.snn_config import Settings


class DenseDecoder(nn.Module):
    """Maps the conv backbone's final spatial feature map (post conv2+pool2, NOT
    flattened) back to a dense per-pixel prediction at the original sensor
    resolution. Mirrors the encoder's two conv+pool stages with two
    deconv+upsample stages, then a 1x1 conv to out_channels. Conv/pool/upsample
    arithmetic doesn't invert to an exact size, so the last step is an explicit
    resize to (cfg.SENSOR_H, cfg.SENSOR_W) rather than fiddly output-padding math.
    """

    def __init__(self, cfg: Settings, out_channels: int = 2):
        super().__init__()
        self.out_h, self.out_w = cfg.SENSOR_H, cfg.SENSOR_W
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cfg.CONV2_OUT, cfg.CONV1_OUT, cfg.CONV2_KERNEL),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.POOL_KERNEL, mode="nearest"),
            nn.ConvTranspose2d(cfg.CONV1_OUT, cfg.CONV1_OUT, cfg.CONV1_KERNEL),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=cfg.POOL_KERNEL, mode="nearest"),
            nn.Conv2d(cfg.CONV1_OUT, out_channels, kernel_size=1),
        )

    def forward(self, features):
        """features: [B, CONV2_OUT, h', w'] -> [B, out_channels, SENSOR_H, SENSOR_W]"""
        x = self.net(features)
        return F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
