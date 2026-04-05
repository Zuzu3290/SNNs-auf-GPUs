import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from feature_engine.encoding import OneHotEncoder
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo
abalone = fetch_ucirepo(id=1)

# data (as pandas dataframes)
X = abalone.data.features
y = abalone.data.targets

# # One Hot Encode Sex
# ohe = OneHotEncoder(variables=['Sex'])
# X = ohe.fit_transform(X)

# # Drop Whole Weight and Length (multicolinearity)
# X.drop(['Whole_weight', 'Length'], axis=1, inplace=True)

# View
df = pd.concat([X,y], axis=1)

df2 = df.query('Height < 0.3 and Rings > 2 and outliers != -1').copy()
X = df2.drop(['Rings', 'outliers'], axis=1)
y = np.log(df2[['Rings']])

# X and Y to Numpy
X = X.to_numpy()
y = y.to_numpy()

# Prepare with TensorData
# TensorData helps us transforming the dataset to Tensor object
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

input_sample, label_sample = dataset[0]
print(f"** Input sample: {input_sample}, \n** Label sample: {label_sample}")

batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class AbaloneModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(in_features=X.shape[1], out_features=128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 32)
    self.linear4 = nn.Linear(32, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = nn.functional.relu(x)
    x = self.linear2(x)
    x = nn.functional.relu(x)
    x = self.linear3(x)
    x = nn.functional.relu(x)
    x = self.linear4(x)
    return x

# Instantiate model
model = AbaloneModel()

criterion = nn.MSELoss()

# Random Search
values = []
best_loss = 999
for idx in range(1000):
  # Randomly sample a learning rate factor between 2 and 4
  factor = np.random.uniform(2,5)
  lr = 10 ** -factor

  # Randomly select a momentum between 0.85 and 0.99
  momentum = np.random.uniform(0.90, 0.99)

  # 1. Get Data
  feature, target = dataset[:]
  # 2. Zero Gradients: Clear old gradients before the backward pass
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  optimizer.zero_grad()
  # 3. Forward Pass: Compute prediction
  y_pred = model(feature)
  # 4. Compute Loss
  loss = criterion(y_pred, target)
  # 4.1 Register best Loss
  if loss < best_loss:
    best_loss = loss
    best_lr = lr
    best_momentum = momentum
    best_idx = idx

  # 5. Backward Pass: Compute gradient of the loss w.r.t W and b'
  loss.backward()
  # 6. Update Parameters: Adjust W and b using the calculated gradients
  optimizer.step()
  values.append([idx, lr, momentum, loss])

print(f'n: {idx},lr: {lr}, momentum: {momentum}, loss: {loss}')

#Once we get the best learning rate and momentum, we can move on
