import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
import os 
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import statistics
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate
import snntorch.functional as SF # common arithmetic operations on spikes, e.g., loss, regularization etc.
from snntorch import utils
import snntorch.spikegen as spikegen #a library for spike generation and data conversion
import snntorch.spikeplot as SP # a library for visualizing spiking activity and network dynamics

# Dataset usse if we are to encode spikes from images
batch_size = 128
data_path = '/tmp/data/mnist'
num_classes = 10 # MInst data set has 10 classes (digits 0-9) 

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

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