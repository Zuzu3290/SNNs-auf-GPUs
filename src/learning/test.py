import torch

from skeleton import Settings, build_model

cfg = Settings("config/snn_config.yaml")
model = build_model(cfg)

x = torch.rand(
    cfg.BATCH_SIZE,
    cfg.IMAGE_CHANNELS,
    cfg.IMAGE_HEIGHT,
    cfg.IMAGE_WIDTH
)

spk_rec, mem_rec = model(x)

print("Input:", x.shape)
print("Spike recording:", spk_rec.shape)
print("Membrane recording:", mem_rec.shape)