"""
Simple Spiking Neural Network using Norse
==========================================
Architecture:
    784 input neurons (one per MNIST pixel)
    --> 256 hidden LIF neurons
    --> 10 output LIF neurons (one per digit class)

Training:
    - Poisson spike encoding: pixel brightness -> spike rate
    - T=16 timesteps per image
    - Surrogate gradients for backpropagation through spikes
    - Cross-entropy loss on summed output spike counts
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import norse.torch as norse
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────
T          = 16       # number of timesteps per image
N_IN       = 784      # 28x28 pixels flattened
N_HIDDEN   = 256      # hidden LIF neurons
N_OUT      = 10       # one per digit class
BATCH_SIZE = 64
EPOCHS     = 3
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {DEVICE}")

# ─────────────────────────────────────────
# 2. POISSON SPIKE ENCODER
#    Converts pixel brightness [0,1] to
#    binary spike trains over T timesteps
# ─────────────────────────────────────────
def poisson_encode(x: torch.Tensor, t: int) -> torch.Tensor:
    """
    x: shape (batch, 784)  values in [0, 1]
    returns: shape (T, batch, 784)  binary spikes
    """
    # repeat x across T timesteps, then sample spikes with probability = brightness
    x_expanded = x.unsqueeze(0).repeat(t, 1, 1)        # (T, batch, 784)
    spikes = torch.bernoulli(x_expanded)                 # 1 with prob=brightness
    return spikes                                         # (T, batch, 784)

# ─────────────────────────────────────────
# 3. SNN MODEL
# ─────────────────────────────────────────
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Linear layers (no spiking, just weighted connections)
        self.fc1 = nn.Linear(N_IN, N_HIDDEN)
        self.fc2 = nn.Linear(N_HIDDEN, N_OUT)

        # LIF layers (the spiking neurons)
        # LIFCell processes one timestep at a time and carries state forward
        self.lif1 = norse.LIFCell()   # hidden layer
        self.lif2 = norse.LIFCell()   # output layer

    def forward(self, spike_input: torch.Tensor):
        """
        spike_input: (T, batch, 784) — binary spike trains
        returns:     (batch, 10)     — summed output spikes over all timesteps
        """
        # initialise neuron states to None (Norse creates them on first call)
        state1 = None
        state2 = None

        output_spikes = []

        for t in range(T):
            x = spike_input[t]                  # (batch, 784) — one timestep

            # pass through first linear layer
            x = self.fc1(x)                     # (batch, 256) — weighted input

            # pass through first LIF layer
            # z1 = spikes fired, state1 = membrane state carried to next timestep
            z1, state1 = self.lif1(x, state1)   # z1: (batch, 256)

            # pass through second linear layer
            x = self.fc2(z1)                    # (batch, 10)

            # pass through output LIF layer
            z2, state2 = self.lif2(x, state2)   # z2: (batch, 10)

            output_spikes.append(z2)             # collect spikes at this timestep

        # stack and sum: (T, batch, 10) -> (batch, 10)
        # more spikes at output neuron i = network is more confident it's digit i
        return torch.stack(output_spikes, dim=0).sum(dim=0)


# ─────────────────────────────────────────
# 4. LOAD MNIST
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),               # pixel values -> [0, 1]
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")

# ─────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────
model     = SimpleSNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs   = []

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)   # (batch, 784)
        labels = labels.to(DEVICE)   # (batch,)

        # encode pixels as spike trains
        spikes = poisson_encode(images, T)   # (T, batch, 784)

        # forward pass
        optimizer.zero_grad()
        output = model(spikes)               # (batch, 10) — summed spike counts

        # compute loss and backpropagate
        # Norse surrogate gradients handle the non-differentiable spikes
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # track metrics
        predicted      = output.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = total_loss / len(train_loader)
    epoch_acc  = 100.0 * total_correct / total_samples
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"\nEpoch {epoch+1} complete | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%\n")

# ─────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        spikes = poisson_encode(images, T)
        output = model(spikes)
        predicted      = output.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

test_acc = 100.0 * total_correct / total_samples
print(f"Test Accuracy: {test_acc:.2f}%")

# ─────────────────────────────────────────
# 7. VISUALISE: SPIKE RASTER
#    Show which neurons fired over time
#    for a single example image
# ─────────────────────────────────────────
model.eval()
sample_image, sample_label = test_dataset[0]
sample_image = sample_image.unsqueeze(0).to(DEVICE)   # (1, 784)
spikes       = poisson_encode(sample_image, T)         # (T, 1, 784)

# collect output spikes per timestep
state1 = None
state2 = None
hidden_spikes_over_time = []
output_spikes_over_time = []

with torch.no_grad():
    for t in range(T):
        x        = spikes[t]
        x        = model.fc1(x)
        z1, state1 = model.lif1(x, state1)
        x        = model.fc2(z1)
        z2, state2 = model.lif2(x, state2)
        hidden_spikes_over_time.append(z1.squeeze(0).cpu().numpy())
        output_spikes_over_time.append(z2.squeeze(0).cpu().numpy())

hidden_raster = np.array(hidden_spikes_over_time)   # (T, 256)
output_raster = np.array(output_spikes_over_time)   # (T, 10)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# subplot 1: the input image
axes[0].imshow(sample_image.cpu().view(28, 28).numpy(), cmap="gray")
axes[0].set_title(f"Input Image — True Label: {sample_label}", fontsize=13)
axes[0].axis("off")

# subplot 2: hidden layer spike raster (first 50 neurons for clarity)
axes[1].imshow(hidden_raster[:, :50].T, aspect="auto", cmap="binary",
               interpolation="nearest")
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Hidden Neuron Index (first 50)")
axes[1].set_title("Hidden Layer Spike Raster", fontsize=13)
axes[1].set_xticks(range(T))

# subplot 3: output layer spike raster (all 10 output neurons)
axes[2].imshow(output_raster.T, aspect="auto", cmap="Blues",
               interpolation="nearest")
axes[2].set_xlabel("Timestep")
axes[2].set_ylabel("Output Neuron (digit class)")
axes[2].set_yticks(range(10))
axes[2].set_xticks(range(T))
axes[2].set_title(
    f"Output Layer Spike Raster | Predicted: {output_raster.sum(axis=0).argmax()}",
    fontsize=13
)

plt.tight_layout()
plt.savefig("src/Learning/norse_spike_raster.png", dpi=150)
print("\nSpike raster saved to src/Learning/norse_spike_raster.png")
plt.show()
