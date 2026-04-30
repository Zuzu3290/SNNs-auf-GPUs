import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNLibrary:
    """
    Wrapper class around common snnTorch features.

    This class stores the main reusable properties for building SNN models:
    - beta/leak behavior
    - surrogate gradient
    - LIF neurons
    - Conv2d blocks
    - Linear blocks
    - spike recording
    - membrane recording
    - rate-coded prediction
    """

    def __init__(self, config):
        self.cfg = config

        self.beta = config.BETA
        self.num_steps = config.TIMESTEPS
        self.num_classes = config.NUM_CLASSES

        self.spike_grad = surrogate.fast_sigmoid(slope=25)

    def create_lif(self):
        """
        Creates a Leaky Integrate-and-Fire neuron using snnTorch.
        """
        return snn.Leaky(
            beta=self.beta,
            spike_grad=self.spike_grad
        )

    def create_conv2d(self, in_channels, out_channels, kernel_size=5, padding=2):
        """
        Creates a normal PyTorch Conv2d layer.
        """
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def create_maxpool2d(self, kernel_size=2):
        """
        Creates a MaxPool2d layer.
        """
        return nn.MaxPool2d(kernel_size)

    def create_linear(self, in_features, out_features):
        """
        Creates a normal PyTorch Linear layer.
        """
        return nn.Linear(
            in_features=in_features,
            out_features=out_features
        )

    def init_memory(self, lif_layer):
        """
        Initializes the membrane potential for a given LIF layer.
        """
        return lif_layer.init_leaky()

    def conv_lif_pool_step(self, x, conv_layer, lif_layer, pool_layer, mem):
        """
        One full SNN convolutional step:

        input
          -> Conv2d
          -> MaxPool2d
          -> LIF neuron
          -> spike output + updated membrane
        """
        current = pool_layer(conv_layer(x))
        spike, mem = lif_layer(current, mem)

        return spike, mem

    def linear_lif_step(self, x, linear_layer, lif_layer, mem):
        """
        One full SNN fully-connected step:

        input
          -> Linear
          -> LIF neuron
          -> spike output + updated membrane
        """
        current = linear_layer(x)
        spike, mem = lif_layer(current, mem)

        return spike, mem

    def stack_recordings(self, recordings):
        """
        Converts a list of time-step recordings into one tensor.

        Output shape:
        [timesteps, batch_size, features]
        """
        return torch.stack(recordings, dim=0)

    def rate_code_prediction(self, spike_recording):
        """
        Rate-coded prediction.

        The output neuron with the highest number of spikes wins.

        spike_recording shape:
        [timesteps, batch_size, num_classes]
        """
        return spike_recording.sum(dim=0).argmax(dim=1)

    def count_parameters(self, model):
        """
        Counts trainable model parameters.
        """
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    def calculate_flattened_size(self):
        """
        Calculates flattened size after two 2x2 pooling operations.

        Example MNIST:
        input: 28x28
        after pool1: 14x14
        after pool2: 7x7

        final: 64 * 7 * 7
        """
        return 64 * (self.cfg.IMAGE_HEIGHT // 4) * (self.cfg.IMAGE_WIDTH // 4)