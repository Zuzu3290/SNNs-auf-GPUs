import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# 1. DOWNLOAD MNIST USING KAGGLEHUB
# ============================================
print("Downloading MNIST dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# ============================================
# 2. CUSTOM DATASET CLASS FOR CSV
# ============================================
class MNISTDataset(Dataset):
    """Custom Dataset class to load MNIST from CSV files"""
    def __init__(self, csv_path, transform=None):
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Separate labels and features
        # First column is label, rest are pixel values (0-255)
        self.labels = df.iloc[:, 0].values.astype('int64')
        self.images = df.iloc[:, 1:].values.astype('float32') / 255.0  # Normalize to [0, 1]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get image and label
        image = self.images[idx]
        label = self.labels[idx]
        
        # Reshape to 28x28 (1 channel for grayscale)
        image = torch.tensor(image, dtype=torch.float32).reshape(1, 28, 28)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

# ============================================
# 3. LOAD TRAINING AND TEST DATA
# ============================================
# Find CSV files in the dataset directory
csv_files = list(Path(dataset_path).glob("*.csv"))
print(f"\nFound CSV files: {csv_files}")

# Load train and test datasets
# Adjust the filenames based on what's in your dataset
train_csv = str(Path(dataset_path) / "mnist_train.csv")
test_csv = str(Path(dataset_path) / "mnist_test.csv")

print("\nLoading datasets...")
train_dataset = MNISTDataset(train_csv)
test_dataset = MNISTDataset(test_csv)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ============================================
# 4. DEFINE NEURAL NETWORK
# ============================================
class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Pooling and activation
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 2 pooling layers: 28 -> 14 -> 7
        # Feature maps: 64, so 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (0-9)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================
# 5. TRAINING SETUP
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ============================================
# 6. TRAINING LOOP
# ============================================
num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    train_accuracy = 100 * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    # Testing phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), 'best_mnist_model.pth')
        print(f"  ✓ Best model saved with accuracy: {best_accuracy:.2f}%")
    
    scheduler.step()

print(f"\n✅ Training complete! Best test accuracy: {best_accuracy:.2f}%")

# ============================================
# 7. MAKE PREDICTIONS ON NEW DATA
# ============================================
model.eval()
with torch.no_grad():
    # Get a batch from test set
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    print(f"\nSample Predictions:")
    print(f"True labels:      {labels[:10].cpu().numpy()}")
    print(f"Predicted labels: {predicted[:10].cpu().numpy()}")