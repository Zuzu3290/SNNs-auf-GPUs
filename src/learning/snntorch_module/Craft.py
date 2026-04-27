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

# # network initialization
# state = torch.zeros(1, output_size) # state
# w = torch.zeros(input_size, output_size) # weights
# spike_grad = surrogate.fast_sigmoid() # surrogate gradient
# y = torch.zeros(1, output_size) # target output placeholder
# y_predicted = torch.zeros(1, output_size) # predicted output placeholder

# class SNNLayer(nn.Module):