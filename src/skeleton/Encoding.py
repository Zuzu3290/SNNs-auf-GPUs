import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import tonic
    import tonic.transforms as tonic_transforms
except ImportError:
    tonic = None


class DigitalDatasetEncoder:
    def __init__(self, config):
        self.cfg = config

    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.IMAGE_HEIGHT, self.cfg.IMAGE_WIDTH)),
            transforms.Grayscale(num_output_channels=self.cfg.IMAGE_CHANNELS),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

    def load_dataset(self):
        transform = self.get_transforms()

        if self.cfg.DATASET_NAME.upper() == "MNIST":
            train_dataset = datasets.MNIST(
                root=self.cfg.DATA_PATH,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.MNIST(
                root=self.cfg.DATA_PATH,
                train=False,
                download=True,
                transform=transform
            )

        elif self.cfg.DATASET_NAME.upper() == "FASHIONMNIST":
            train_dataset = datasets.FashionMNIST(
                root=self.cfg.DATA_PATH,
                train=True,
                download=True,
                transform=transform
            )

            test_dataset = datasets.FashionMNIST(
                root=self.cfg.DATA_PATH,
                train=False,
                download=True,
                transform=transform
            )

        else:
            raise ValueError(f"Unsupported digital dataset: {self.cfg.DATASET_NAME}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            drop_last=True
        )

        return train_loader, test_loader


class NeuromorphicDatasetEncoder:
    def __init__(self, config):
        self.cfg = config

        if tonic is None:
            raise ImportError("Tonic is not installed. Install it using: pip install tonic")

    def load_dataset(self):
        sensor_size = tonic.datasets.NMNIST.sensor_size

        transform = tonic_transforms.Compose([
            tonic_transforms.ToFrame(
                sensor_size=sensor_size,
                n_time_bins=self.cfg.TIMESTEPS
            )
        ])

        if self.cfg.DATASET_NAME.upper() == "NMNIST":
            train_dataset = tonic.datasets.NMNIST(
                save_to=self.cfg.DATA_PATH,
                train=True,
                transform=transform
            )

            test_dataset = tonic.datasets.NMNIST(
                save_to=self.cfg.DATA_PATH,
                train=False,
                transform=transform
            )

        else:
            raise ValueError(f"Unsupported neuromorphic dataset: {self.cfg.DATASET_NAME}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            drop_last=True
        )

        return train_loader, test_loader


def build_dataloaders(config):
    if config.DATASET_TYPE.lower() == "digital":
        encoder = DigitalDatasetEncoder(config)

    elif config.DATASET_TYPE.lower() == "neuromorphic":
        encoder = NeuromorphicDatasetEncoder(config)

    else:
        raise ValueError("DATASET_TYPE must be either 'digital' or 'neuromorphic'")

    return encoder.load_dataset()

train_loader, test_loader = build_dataloaders(config)

#Apply data_subset to reduce the dataset by the factor defined in subset
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
num_steps = 100 
data = iter(train_loader)
data_it, targets_it = next(data)

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


# Receive raw events 
# (x,y,t,p)
# Group them into a short time window.
# Convert them into one of these:
# event voxel grid,
# event frame,
# time surface,
# direct spike input.
# Feed that into the SNN.
#Resume this study at https://www.perplexity.ai/search/76c72946-8644-487d-8c15-fd6d6bb00123