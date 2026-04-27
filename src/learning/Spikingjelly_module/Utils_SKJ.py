import torch
import torch.nn as nn
import spikingjelly as sj
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import os 
import time
from spikingjelly.activation_based import neuron, functional, layer, monitor, surrogate
from spikingjelly.activation_based import layer, learning
import zlib
import torch.nn.functional as F
from torch.optim import SGD, Adam
from spikingjelly.activation_based.cuda_kernel import tensor_cache


import matplotlib.pyplot as plt
import numpy as np
import itertools
import statistics
import tqdm

## Define the varies loss function and also the variation in the types of neuron models 
## All accessable using the env and reimbursed to the model structural dyamics and the training loop.
MSE = nn.MSELoss()
