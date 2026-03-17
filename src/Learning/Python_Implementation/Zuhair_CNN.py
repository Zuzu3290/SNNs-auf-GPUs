import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================
# 1. DOWNLOAD MNIST DATASET
# ============================================
print("📥 Downloading MNIST dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
print(f"✓ Dataset path: {dataset_path}")

# ============================================
# 2. CUSTOM DATASET CLASS
# ============================================
class MNISTDataset(Dataset):
    """Load MNIST from CSV files"""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values.astype('int64')
        self.images = df.iloc[:, 1:].values.astype('float32') / 255.0
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).reshape(1, 28, 28)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ============================================
# 3. CNN ARCHITECTURE
# ============================================
class CNNModel(nn.Module):
    """
    Convolutional Neural Network for MNIST
    
    Architecture:
    - Conv Layer 1: 1 -> 32 filters (5x5 kernel)
    - Conv Layer 2: 32 -> 64 filters (5x5 kernel)
    - Conv Layer 3: 64 -> 128 filters (3x3 kernel)
    - Fully Connected: 128*2*2 -> 256 -> 10
    """
    
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 28 -> 14 -> 7 -> 3 (with padding adjustments)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

# ============================================
# 4. LOAD DATA
# ============================================
print("\n📂 Loading datasets...")
csv_files = sorted(Path(dataset_path).rglob("*.csv"))

if len(csv_files) >= 2:
    train_csv = csv_files[0]
    test_csv = csv_files[1]
else:
    raise FileNotFoundError("CSV files not found!")

train_dataset = MNISTDataset(str(train_csv))
test_dataset = MNISTDataset(str(test_csv))

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✓ Train samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Test batches: {len(test_loader)}")

# ============================================
# 5. TRAINING SETUP
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")

model = CNNModel(dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Print model summary
print(f"\n📊 Model Summary:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================
# 6. TRAINING FUNCTION
# ============================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ============================================
# 7. EVALUATION FUNCTION
# ============================================
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ============================================
# 8. TRAINING LOOP
# ============================================
num_epochs = 15
best_accuracy = 0.0
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

print(f"\n🚀 Starting training for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    print(f"{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"{'='*60}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    print(f"\n📈 Results:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(test_acc)
    
    # Save best model
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), 'best_cnn_model.pth')
        print(f"  ✅ Best model saved! Accuracy: {best_accuracy:.2f}%")

print(f"\n{'='*60}")
print(f"✅ Training Complete!")
print(f"Best Test Accuracy: {best_accuracy:.2f}%")
print(f"{'='*60}\n")

# ============================================
# 9. VISUALIZE RESULTS
# ============================================
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['test_loss'], label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', marker='o')
plt.plot(history['test_acc'], label='Test Acc', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
print("📊 Training history saved to 'training_history.png'")
plt.show()

# ============================================
# 10. MAKE PREDICTIONS
# ============================================
print("\n🔮 Making Predictions on Test Set...")
model.load_state_dict(torch.load('best_cnn_model.pth'))
model.eval()

with torch.no_grad():
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    print(f"\nSample Predictions (first 10):")
    print(f"True:      {labels[:10].cpu().numpy()}")
    print(f"Predicted: {predicted[:10].cpu().numpy()}")
    print(f"Match:     {(labels[:10] == predicted[:10]).cpu().numpy()}")

# ============================================
# 11. SAVE MODEL FOR FUTURE USE
# ============================================
print("\n💾 Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': CNNModel,
    'best_accuracy': best_accuracy
}, 'cnn_model_checkpoint.pth')
print("✓ Model saved as 'cnn_model_checkpoint.pth'")
print("✓ Best weights saved as 'best_cnn_model.pth'")