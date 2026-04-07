import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F
import os 
import time
import snn_cuda.snn_forward as snn_cuda
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


#interface
input_size = 0
output_size = 0
hidden_layer_width = 0
hidden_layer_height = 0
# time_interval = 0
threshold = 0
beta = 0.5  # neuron decay rate, this controls membrane decay.
nap_time = 0
batch_size = 1 # only one sample to learn, daataloader returns as an iterator divided up into mini-batches of size
neuron_capacity = 0
spike_grad = surrogate.fast_sigmoid() # surrogate gradient
num_steps = 25 # SNNs process data over time. implement for operation varibale : timesetps
num_inter = 0
y = torch.zeros(1, output_size) # target output placeholder
y_predicted = torch.zeros(1, output_size) # predicted output placeholder


##apparently there is  a stance regaridng SNN which should be updated with states 

##



# network initialization
state = torch.zeros(1, output_size) # state
w = torch.zeros(input_size, output_size) # weights

# Data Stream : DataLoader will serve it up in batches and loads data to memeory prior. 
class Data_stream():
    def __init__(self):
        self.input_data = torch.randn(1, input_size)
        #scaling the data input to be in the range of 0-1 for better training stability
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

    def __iter__(self):
        yield self.input_data # In practice, this would yield batches of data over time

    def train_data(self):
        # This method can be used to update the data stream with new data during training
        pass

    def test_data(self):
        # This method can be used to update the data stream with new data during testing
        pass

class SNNLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold, hidden, timesteps):
        super(SNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.threshold = threshold

        self.timesteps = timesteps # number of time steps to simulate the network
        self.hidden = hidden # number of hidden neurons 
        spike_grad = surrogate.fast_sigmoid() # surrogate gradient function
        
        # randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # layer 1
        self.fc_in = torch.nn.Linear(in_features=1, out_features=self.hidden)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)
        
        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(self.hidden)
        thr_hidden = torch.rand(self.hidden)

        # layer 2
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(1)
        
        # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
        self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=1)
        self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
        # Initialize weights and state variables on the GPU
        self.weights = torch.randn(input_size, output_size) * 0.1
        self.voltages = torch.zeros(output_size)
        self.spikes = torch.zeros(output_size)

        # Define a transform
        transform = transforms.Compose([
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])

        mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
            nn.Conv2d(8, 16, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 10),
            snn.Leaky(beta=beta, init_hidden=True, spike_grad=spike_grad, output=True)
            )

    def forward(self, input):
        # # Call the CUDA kernel for the forward pass
        # snn_cuda.snn_forward(input, self.voltages, self.spikes, self.weights)
        # return self.spikes
        """Forward pass for several time steps."""

        # Initalize membrane potential
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()

        # Empty lists to record outputs
        mem_3_rec = []

        # Loop over 
        for step in range(self.timesteps):
            x_timestep = x[step, :, :]

            cur_in = self.fc_in(x_timestep)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)
            
            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.li_out(cur_hidden, mem_2)

            cur_out = self.fc_out(spk_hidden)
            _, mem_3 = self.li_out(cur_out, mem_3)

            mem_3_rec.append(mem_3)

        return torch.stack(mem_3_rec)
    
    def reset(self):
        if neuron_capacity > threshold:
            self.voltages.zero_()
            self.spikes.zero_()
            self.time.sleep(nap_time) # simulate nap time for energy efficiency 

    def loss(self, output, target):
        # Example loss function (e.g., mean squared error)
        return F.mse_loss(output, target)
    
    def backward(self, grad_output):
        # Implement backward pass (e.g., using surrogate gradients)
        pass

    def update_weights(self, grad_output):
        # Update weights based on gradients and learning rate
        pass

    def gradient_clipping(self, max_norm):
        # Clip gradients to prevent exploding gradients
        pass

    def heartbeat(self):
        # This function can be called periodically to monitor the health of each neuron during operations 
        pass


spike_recording = [] # record spikes over time

training = Data_stream.train_data()
testing = Data_stream.test_data()

def train_snn(model, data, optimizer, loss_fn, device):
    model.train()
    utils.reset(SNNLayer.net) # reset/initialize hidden states for all neurons
    for epoch in range(num_steps):
        for input in data:
            input = input.to(device)
            output = model.forward(input)
            target = torch.zeros_like(output) # Placeholder target
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            model.gradient_clipping(max_norm=1.0) # Example max norm for clipping
            model.update_weights(optimizer)
            spike, state = net(data_in[step]) # one time step of forward-pass
            spike_recording.append(spike) # record spikes in list
        print(f"Epoch {num_steps+1}/{num_steps}, Loss: {loss.item():.4f}")

    ## Running a single forwar-pass as per tutorial to visualize the output of an untrained network.
    train_batch = iter(dataloader)
        with torch.no_grad():
            for feature, label in train_batch:
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)
                mem = model(feature)

        # plot
        plt.plot(mem[:, 0, 0].cpu(), label="Output")
        plt.plot(label[:, 0, 0].cpu(), '--', label="Target")
        plt.title("Untrained Output Neuron")
        plt.xlabel("Time")
        plt.ylabel("Membrane Potential")
        plt.legend(loc='best')
        plt.show()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

loss_hist = [] # record loss

# training loop
# with tqdm.trange(num_iter) as pbar:
#     for _ in pbar:
#         train_batch = iter(dataloader)
#         minibatch_counter = 0
#         loss_epoch = []
        
#         for feature, label in train_batch:
#             # prepare data
#             feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
#             label = torch.swapaxes(input=label, axis0=0, axis1=1)
#             feature = feature.to(device)
#             label = label.to(device)

#             # forward pass
#             mem = model(feature)
#             loss_val = loss_function(mem, label) # calculate loss
#             optimizer.zero_grad() # zero out gradients
#             loss_val.backward() # calculate gradients
#             optimizer.step() # update weights

#             # store loss
#             loss_hist.append(loss_val.item())
#             loss_epoch.append(loss_val.item())
#             minibatch_counter += 1

#             avg_batch_loss = sum(loss_epoch) / minibatch_counter # calculate average loss p/epoch
#             pbar.set_postfix(loss="%.3e" % avg_batch_loss) # print loss p/batch


def test_snn(model, data, device):
    model.eval()
    with torch.no_grad():
        for input in data:
            input = input.to(device)
            output = model.forward(input)
            print(f"Output: {output.cpu().numpy()}")



device = torch.device("cuda") if torch.cuda.is_available() else print("CUDA not available, need CPU")
model = SNNLayer(input_size, output_size, threshold=threshold, hidden=hidden_layer_width, timesteps=num_steps)
model.to_cuda() # Move model to GPU

#Evaluation 
loss_function = torch.nn.L1Loss() # Use L1 loss instead

 # pause gradient calculation during evaluation
with torch.no_grad():
    model.eval()

    test_batch = iter(dataloader)
    minibatch_counter = 0
    rel_err_lst = []

    # loop over data samples
    for feature, label in test_batch:

        # prepare data
        feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
        label = torch.swapaxes(input=label, axis0=0, axis1=1)
        feature = feature.to(device)
        label = label.to(device)

        # forward-pass
        mem = model(feature)

        # calculate relative error
        rel_err = torch.linalg.norm(
            (mem - label), dim=-1
        ) / torch.linalg.norm(label, dim=-1)
        rel_err = torch.mean(rel_err[1:, :])

        # calculate loss
        loss_val = loss_function(mem, label)

        # store loss
        loss_hist.append(loss_val.item())
        rel_err_lst.append(rel_err.item())
        minibatch_counter += 1

    mean_L1 = statistics.mean(loss_hist)
    mean_rel = statistics.mean(rel_err_lst)

print(f"{'Mean L1-loss:':<{20}}{mean_L1:1.2e}")
print(f"{'Mean rel. err.:':<{20}}{mean_rel:1.2e}")


rlif = snn.RLeaky(beta=beta, all_to_all=False) # initialize RLeaky Neuron
spk, mem = rlif.init_rleaky() # initialize state variables
x = torch.rand(1) # generate random input

spk_recording = []
mem_recording = []

# run simulation
for step in range(num_steps):
  spk, mem = rlif(x, spk, mem)
  spk_recording.append(spk)
  mem_recording.append(mem)

