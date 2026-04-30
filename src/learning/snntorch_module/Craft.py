import torch
import torch.nn as nn

from src.learning.snntorch_module.library import SNNLibrary


class CSNN(nn.Module):
    """
    Convolutional Spiking Neural Network.

    Craft.py should only define the network architecture flow.
    The snnTorch-related details are handled inside library.py.
    """

    def __init__(self, config):
        super().__init__()

        self.cfg = config
        self.snn = SNNLibrary(config)

        self.num_steps = config.TIMESTEPS
        self.num_classes = config.NUM_CLASSES

        self.conv1 = self.snn.create_conv2d(
            in_channels=config.IMAGE_CHANNELS,
            out_channels=12,
            kernel_size=5,
            padding=2
        )

        self.lif1 = self.snn.create_lif()
        self.pool1 = self.snn.create_maxpool2d(kernel_size=2)

        self.conv2 = self.snn.create_conv2d(
            in_channels=12,
            out_channels=64,
            kernel_size=5,
            padding=2
        )

        self.lif2 = self.snn.create_lif()
        self.pool2 = self.snn.create_maxpool2d(kernel_size=2)

        flattened_size = self.snn.calculate_flattened_size()

        self.fc1 = self.snn.create_linear(
            in_features=flattened_size,
            out_features=self.num_classes
        )

        self.lif3 = self.snn.create_lif()

    def forward(self, x):
        mem1 = self.snn.init_memory(self.lif1)
        mem2 = self.snn.init_memory(self.lif2)
        mem3 = self.snn.init_memory(self.lif3)

        spike_recording = []
        membrane_recording = []

        for _ in range(self.num_steps):
            spk1, mem1 = self.snn.conv_lif_pool_step(
                x=x,
                conv_layer=self.conv1,
                lif_layer=self.lif1,
                pool_layer=self.pool1,
                mem=mem1
            )

            spk2, mem2 = self.snn.conv_lif_pool_step(
                x=spk1,
                conv_layer=self.conv2,
                lif_layer=self.lif2,
                pool_layer=self.pool2,
                mem=mem2
            )

            flattened = spk2.flatten(1)

            spk3, mem3 = self.snn.linear_lif_step(
                x=flattened,
                linear_layer=self.fc1,
                lif_layer=self.lif3,
                mem=mem3
            )

            spike_recording.append(spk3)
            membrane_recording.append(mem3)

        return (
            self.snn.stack_recordings(spike_recording),
            self.snn.stack_recordings(membrane_recording)
        )


def build_model(config):
    if config.MODEL_TYPE.upper() == "CSNN":
        return CSNN(config)

    raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")