import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import statistics
import tqdm

## Define the varies loss function and also the variation in the types of neuron models 
## All accessable using the env and reimbursed to the model structural dyamics and the training loop.
MSE = nn.MSELoss()
##for snn
loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
#Inlcusion of performance matrix and the confusion matrix for the evaluation of the model performance
#Inclusion of recoridng performance matrix data for later resulta nd evaulatzion of perfromace.
def plot_confusion_matrix():
    pass