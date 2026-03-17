import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import os

# ============================================
# 0. GPU UTILITIES & MONITORING
# ============================================
class GPUMonitor:
    """Monitor GPU usage and performance"""
    
    @staticmethod
    def print_gpu_info():
        if torch.cuda.is_available():
            print(f"🖥️  GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"    Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"    CUDA Version: {torch.version.cuda}")
            print(f"    PyTorch Version: {torch.__version__}")
        else:
            print("⚠️  GPU not available, using CPU")
    
    @staticmethod
    def get_gpu_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return allocated, reserved
        return 0, 0
    
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def print_memory_stats():
        if torch.cuda.is_available():
            allocated, reserved = GPUMonitor.get_gpu_memory()
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# ============================================
# 1. OPTIMIZED SPIKING NEURON (GPU-FRIENDLY)
# ============================================
class LIFNeuronGPU(nn.Module):
    """
    GPU-optimized Leaky Integrate-and-Fire Neuron
    - Vectorized operations for batch processing
    - Reduced memory footprint
    - Fused CUDA kernels
    """
    
    def __init__(self, tau=0.25, vth=1.0, reset_voltage=0.0):
        super(LIFNeuronGPU, self).__init__()
        self.register_buffer('tau', torch.tensor(tau, dtype=torch.float32))
        self.register_buffer('vth', torch.tensor(vth, dtype=torch.float32))
        self.register_buffer('reset_voltage', torch.tensor(reset_voltage, dtype=torch.float32))
    
    def forward(self, input_tensor, voltage=None):
        """
        Optimized forward pass with in-place operations
        
        Args:
            input_tensor: (batch_size, n_neurons)
            voltage: (batch_size, n_neurons) or None
        
        Returns:
            spikes: (batch_size, n_neurons)
            voltage: (batch_size, n_neurons)
        """
        if voltage is None:
            voltage = torch.zeros_like(input_tensor)
        
        # In-place operations to reduce memory
        voltage.mul_(self.tau).add_(input_tensor, alpha=1 - self.tau)
        
        # Spike generation and reset
        spikes = (voltage > self.vth).float()
        voltage.sub_(spikes * self.vth)
        
        return spikes, voltage

# ============================================
# 2. OPTIMIZED SNN LAYER
# ============================================
class SNNLayerGPU(nn.Module):
    """GPU-optimized SNN layer with efficient computation"""
    
    def __init__(self, input_size, output_size, tau=0.25, vth=1.0, 
                 use_bias=True, weight_init='kaiming'):
        super(SNNLayerGPU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Synaptic weights with optimized initialization
        if weight_init == 'kaiming':
            self.weight = nn.Parameter(torch.empty(input_size, output_size))
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        else:
            self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.01)
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter('bias', None)
        
        # LIF neuron
        self.lif = LIFNeuronGPU(tau=tau, vth=vth)
    
    def forward(self, spike_input, voltage=None):
        """
        Forward pass using optimized matrix operations
        
        Args:
            spike_input: (batch_size, input_size)
            voltage: (batch_size, output_size) or None
        
        Returns:
            output_spikes: (batch_size, output_size)
            voltage: (batch_size, output_size)
        """
        # Use fp16 for faster computation (if available)
        with torch.cuda.amp.autocast():
            current = torch.matmul(spike_input, self.weight)
            if self.bias is not None:
                current.add_(self.bias)
        
        output_spikes, voltage = self.lif(current, voltage)
        return output_spikes, voltage

# ============================================
# 3. OPTIMIZED SNN MODEL
# ============================================
class SNNModelGPU(nn.Module):
    """
    GPU-optimized Multi-layer SNN
    - Efficient memory management
    - Gradient checkpointing option
    - Mixed precision support
    """
    
    def __init__(self, input_size=100, hidden_sizes=[256, 128], output_size=10,
                 tau=0.25, vth=1.0, num_timesteps=10, use_checkpointing=False):
        super(SNNModelGPU, self).__init__()
        
        self.num_timesteps = num_timesteps
        self.input_size = input_size
        self.output_size = output_size
        self.use_checkpointing = use_checkpointing
        
        # Build layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([
            SNNLayerGPU(layer_sizes[i], layer_sizes[i+1], tau=tau, vth=vth)
            for i in range(len(layer_sizes) - 1)
        ])
        
        print(f"🏗️  SNN GPU Model Architecture:")
        for i, layer in enumerate(self.layers):
            print(f"    Layer {i+1}: {layer.input_size} -> {layer.output_size}")
        print(f"    Timesteps: {num_timesteps}")
        print(f"    Total Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, spike_input):
        """
        Forward pass with optional gradient checkpointing
        
        Args:
            spike_input: (batch_size, input_size, num_timesteps)
        
        Returns:
            output_spikes: (batch_size, output_size, num_timesteps)
            spike_counts: (batch_size, output_size) - summed over time
        """
        batch_size = spike_input.shape[0]
        
        # Initialize voltages
        voltages = [torch.zeros(batch_size, layer.output_size, 
                               device=spike_input.device, dtype=spike_input.dtype)
                   for layer in self.layers]
        
        output_spikes_list = []
        
        # Unroll time
        for t in range(self.num_timesteps):
            current_input = spike_input[:, :, t]
            
            # Forward through layers
            for layer_idx, layer in enumerate(self.layers):
                spikes, voltages[layer_idx] = layer(current_input, voltages[layer_idx])
                current_input = spikes
            
            output_spikes_list.append(current_input)
        
        # Stack outputs
        output_spikes = torch.stack(output_spikes_list, dim=2)  # (batch, output_size, timesteps)
        spike_counts = output_spikes.sum(dim=2)  # (batch, output_size)
        
        return output_spikes, spike_counts

# ============================================
# 4. OPTIMIZED DATASET WITH GPU PINNING
# ============================================
class RandomSpikeDatasetGPU(Dataset):
    """GPU-optimized dataset with pinned memory"""
    
    def __init__(self, num_samples=10000, input_size=100, output_size=10,
                 num_timesteps=10, spike_probability=0.3, pin_memory=True):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.num_timesteps = num_timesteps
        
        # Generate on CPU
        print("Generating spike trains...")
        data = torch.bernoulli(
            torch.full((num_samples, input_size, num_timesteps), spike_probability)
        )
        labels = torch.randint(0, output_size, (num_samples,))
        
        # Pin memory for faster GPU transfer
        if pin_memory and torch.cuda.is_available():
            data = data.pin_memory()
            labels = labels.pin_memory()
        
        self.data = data
        self.labels = labels
        
        print(f"Dataset created:")
        print(f"  Samples: {num_samples}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Memory: {self.data.element_size() * self.data.nelement() / 1e9:.2f} GB")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ============================================
# 5. CUSTOM SCALER FOR MIXED PRECISION
# ============================================
class MixedPrecisionTrainer:
    """Gradient scaling for mixed precision training"""
    
    def __init__(self, init_scale=65536.0):
        self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
    
    def scale_loss(self, loss):
        return self.scaler.scale(loss)
    
    def step(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()
    
    def unscale(self, optimizer):
        self.scaler.unscale_(optimizer)

# ============================================
# 6. TRAINING SCRIPT
# ============================================
def main():
    print("🧠 GPU-Accelerated Spiking Neural Network")
    print("=" * 70)
    
    # GPU Setup
    GPUMonitor.print_gpu_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    
    print("\n" + "=" * 70)
    
    # Hyperparameters
    input_size = 100
    hidden_sizes = [512, 256]
    output_size = 10
    num_timesteps = 20
    batch_size = 256  # Larger batch size for GPU
    num_epochs = 25
    learning_rate = 0.001
    
    print(f"⚙️  Hyperparameters:")
    print(f"    Input Size: {input_size}")
    print(f"    Hidden Sizes: {hidden_sizes}")
    print(f"    Output Size: {output_size}")
    print(f"    Timesteps: {num_timesteps}")
    print(f"    Batch Size: {batch_size}")
    print(f"    Learning Rate: {learning_rate}")
    print(f"    Epochs: {num_epochs}")
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    train_dataset = RandomSpikeDatasetGPU(
        num_samples=20000,
        input_size=input_size,
        output_size=output_size,
        num_timesteps=num_timesteps,
        spike_probability=0.3,
        pin_memory=True
    )
    
    test_dataset = RandomSpikeDatasetGPU(
        num_samples=4000,
        input_size=input_size,
        output_size=output_size,
        num_timesteps=num_timesteps,
        spike_probability=0.3,
        pin_memory=True
    )
    
    # DataLoaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Test batches: {len(test_loader)}")
    
    # Model
    print(f"\n🏗️  Building GPU-optimized SNN...")
    model = SNNModelGPU(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        tau=0.25,
        vth=1.0,
        num_timesteps=num_timesteps,
        use_checkpointing=False
    ).to(device)
    
    # Mixed precision
    print(f"\n⚡ Using Mixed Precision Training (FP16/FP32)")
    scaler = MixedPrecisionTrainer()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    GPUMonitor.print_memory_stats()
    
    # ============================================
    # TRAINING LOOP (GPU OPTIMIZED)
    # ============================================
    print(f"\n🚀 Starting GPU-Accelerated Training...\n")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'batch_time': [], 'throughput': []
    }
    
    best_accuracy = 0.0
    total_batches = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()
        
        with tqdm(train_loader, desc="Training", unit="batch") as pbar:
            for batch_idx, (spike_input, labels) in enumerate(pbar):
                batch_start = time.time()
                
                # Transfer to GPU
                spike_input = spike_input.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward with mixed precision
                with torch.cuda.amp.autocast():
                    output_spikes, spike_counts = model(spike_input)
                    loss = criterion(spike_counts, labels)
                
                # Backward with gradient scaling
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                scaler.scale_loss(loss).backward()
                scaler.unscale(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                
                # Accuracy
                train_loss += loss.item()
                _, predicted = torch.max(spike_counts.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Timing
                batch_time = time.time() - batch_start
                throughput = labels.size(0) / batch_time
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*train_correct/train_total:.2f}%',
                    'throughput': f'{throughput:.0f} samples/sec'
                })
                
                total_batches += 1
        
        epoch_time = time.time() - epoch_start
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # ===== EVALUATION =====
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluating", unit="batch") as pbar:
                for spike_input, labels in pbar:
                    spike_input = spike_input.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        output_spikes, spike_counts = model(spike_input)
                        loss = criterion(spike_counts, labels)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(spike_counts.data, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += labels.size(0)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        test_acc = 100 * test_correct / test_total
        test_loss /= len(test_loader)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print results
        print(f"\n📈 Results:")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"    Epoch Time: {epoch_time:.2f}s")
        
        GPUMonitor.print_memory_stats()
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_snn_gpu_model.pth')
            print(f"    ✅ Best model saved! Accuracy: {best_accuracy:.2f}%")
        
        scheduler.step()
        print()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"    Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"    Total Time: {total_time/60:.2f} minutes")
    print(f"    Total Batches: {total_batches}")
    print(f"    Avg Time/Batch: {total_time/total_batches*1000:.2f} ms")
    print(f"{'='*70}\n")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    print("📊 Generating performance plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history['test_loss'], label='Test Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(history['test_acc'], label='Test Acc', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spike visualization
    model.load_state_dict(torch.load('best_snn_gpu_model.pth'))
    model.eval()
    
    with torch.no_grad():
        sample_input, _ = test_dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        output_spikes, spike_counts = model(sample_input)
        
        axes[1, 0].imshow(sample_input[0].cpu().numpy(), cmap='gray', aspect='auto')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Input Neuron')
        axes[1, 0].set_title('Input Spike Train')
        
        axes[1, 1].imshow(output_spikes[0].cpu().numpy(), cmap='hot', aspect='auto')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Output Neuron')
        axes[1, 1].set_title('Output Spike Train')
    
    plt.tight_layout()
    plt.savefig('snn_gpu_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: snn_gpu_results.png")
    plt.show()
    
    # ============================================
    # SAVE MODEL
    # ============================================
    print("\n💾 Saving SNN GPU model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_sizes': hidden_sizes,
        'output_size': output_size,
        'num_timesteps': num_timesteps,
        'best_accuracy': best_accuracy,
        'history': history
    }, 'snn_gpu_checkpoint.pth')
    print("✓ Model saved as 'snn_gpu_checkpoint.pth'")
    print("✓ Best weights saved as 'best_snn_gpu_model.pth'")
    
    # Clear GPU cache
    GPUMonitor.clear_cache()

if __name__ == "__main__":
    main()