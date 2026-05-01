import torch
import torch.nn as nn

from src.learning.snntorch_module.library import SNNLibrary

    # Spiking Data to rate 
    def rate_encoding():
        spike_data = spikegen.rate(data_it, num_steps=num_steps)
        return print(spike_data.size()) ## Defining input vector size orientation with num_steps x batch_size x input dimensions

    ## snntorch.spikeplot is used to generate a visualization of the spiking activity in the data batcch being sent as neuron input.
    def latency_encoding():
        spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01) # Higehr tau lower firing rate, lower tau higher firing rate. Threshold is the voltage threshold for spike generation, which determines when a neuron will fire based on its membrane potential. A lower threshold will result in more frequent spiking, while a higher threshold will lead to less frequent spiking.
        return print(spike_data.size()) ## Defining input vector size orientation with num_steps x batch_size x input dimensions    

    def convert_to_time(data, tau=5, threshold=0.01):   # threshold isdefined to resapocate voltage threshold and the tau for the time constant of the neuron that is of teh RC circuit.
        spike_time = tau * torch.log(data / (data - threshold))
        return spike_time 

    def delta_encoding():
        # A configuration can be set to mediate between on_spike + off_spike if needed 
        if True :
            spike_data = spikegen.delta(data, threshold=4, off_spike=True)
        else: 
            spike_data = spikegen.delta(data_it, num_steps=100, threshold=0.01)
        return print(spike_data.size()) ## Defining input vector size orientation with num_steps x batch_size x input dimensions
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