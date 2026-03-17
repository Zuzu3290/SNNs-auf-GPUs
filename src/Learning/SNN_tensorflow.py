import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# ============================================
# 0. GPU SETUP & CONFIGURATION
# ============================================
print("🖥️  TensorFlow GPU Configuration")
print("=" * 70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detected: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  {gpu}")
    
    # Enable memory growth to avoid OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Set mixed precision policy for faster training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed Precision Policy: {policy.name}")
else:
    print("⚠️  No GPU detected, using CPU")

print(f"✓ TensorFlow Version: {tf.__version__}")
print(f"✓ Keras Version: {tf.keras.__version__}")
print("=" * 70 + "\n")

# ============================================
# 1. CUSTOM LIF NEURON LAYER
# ============================================
class LIFNeuronLayer(layers.Layer):
    """
    Leaky Integrate-and-Fire (LIF) Neuron Layer
    
    Dynamics:
    V(t) = alpha * V(t-1) + (1-alpha) * I(t)
    S(t) = 1 if V(t) > Vth else 0
    V(t) = V(t) - Vth if S(t) = 1
    """
    
    def __init__(self, units, tau=0.25, vth=1.0, reset_voltage=0.0, 
                 name="lif_neuron", **kwargs):
        super(LIFNeuronLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.tau = tau
        self.vth = vth
        self.reset_voltage = reset_voltage
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.voltage = self.add_weight(
            name='membrane_voltage',
            shape=(1, self.units),
            initializer='zeros',
            trainable=False
        )
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, timesteps, input_features)
        
        Returns:
            spikes: (batch_size, timesteps, units)
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        
        # Initialize voltage for this batch
        voltage = tf.zeros((batch_size, self.units), dtype=inputs.dtype)
        spikes_list = []
        
        # Iterate through timesteps
        for t in range(inputs.shape[1] if inputs.shape[1] is not None else timesteps):
            current = inputs[:, t, :]
            
            # LIF dynamics: V(t) = alpha * V(t-1) + (1-alpha) * I(t)
            voltage = self.tau * voltage + (1.0 - self.tau) * current
            
            # Spike generation: S(t) = 1 if V(t) > Vth
            spikes = tf.cast(voltage > self.vth, inputs.dtype)
            
            # Voltage reset
            voltage = voltage - spikes * self.vth + spikes * self.reset_voltage
            
            spikes_list.append(spikes)
        
        # Stack spikes: (batch_size, timesteps, units)
        output_spikes = tf.stack(spikes_list, axis=1)
        return output_spikes
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'tau': self.tau,
            'vth': self.vth,
            'reset_voltage': self.reset_voltage,
        })
        return config

# ============================================
# 2. SPIKING DENSE LAYER
# ============================================
class SpikingDenseLayer(layers.Layer):
    """Spiking Dense Layer with LIF neuron"""
    
    def __init__(self, units, tau=0.25, vth=1.0, use_bias=True,
                 kernel_initializer='glorot_uniform', name="spiking_dense", **kwargs):
        super(SpikingDenseLayer, self).__init__(name=name, **kwargs)
        self.units = units
        self.tau = tau
        self.vth = vth
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
    
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, input_features)
        self.input_features = input_shape[-1]
        
        # Synaptic weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.input_features, self.units),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.bias = None
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, timesteps, input_features)
        
        Returns:
            spikes: (batch_size, timesteps, units)
            spike_counts: (batch_size, units)
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = inputs.shape[1]
        
        # Initialize voltage
        voltage = tf.zeros((batch_size, self.units), dtype=inputs.dtype)
        spikes_list = []
        
        # Process each timestep
        for t in range(timesteps if timesteps is not None else 10):
            # Linear transformation
            current = tf.matmul(inputs[:, t, :], self.kernel)
            if self.bias is not None:
                current = tf.nn.bias_add(current, self.bias)
            
            # LIF dynamics
            voltage = self.tau * voltage + (1.0 - self.tau) * current
            
            # Spike generation
            spikes = tf.cast(voltage > self.vth, inputs.dtype)
            
            # Voltage reset
            voltage = voltage - spikes * self.vth
            
            spikes_list.append(spikes)
        
        # Stack spikes
        output_spikes = tf.stack(spikes_list, axis=1)
        spike_counts = tf.reduce_sum(output_spikes, axis=1)
        
        return output_spikes, spike_counts
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'tau': self.tau,
            'vth': self.vth,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
        })
        return config

# ============================================
# 3. SNN MODEL
# ============================================
class SNNModelTF(keras.Model):
    """
    Multi-layer Spiking Neural Network in TensorFlow
    
    Architecture:
    - Input: spike trains (batch_size, timesteps, input_size)
    - Hidden Layers: Spiking dense layers
    - Output: Classification layer
    """
    
    def __init__(self, input_size=100, hidden_sizes=[256, 128], output_size=10,
                 tau=0.25, vth=1.0, num_timesteps=10, name="snn_model", **kwargs):
        super(SNNModelTF, self).__init__(name=name, **kwargs)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_timesteps = num_timesteps
        
        # Build layers
        self.spiking_layers = []
        layer_sizes = [input_size] + hidden_sizes
        
        for i, size in enumerate(layer_sizes):
            output_dim = hidden_sizes[i] if i < len(hidden_sizes) else output_size
            layer = SpikingDenseLayer(
                units=output_dim,
                tau=tau,
                vth=vth,
                name=f"spiking_dense_{i+1}"
            )
            self.spiking_layers.append(layer)
        
        # Output classification layer
        self.output_layer = layers.Dense(output_size, activation=None, name='output')
        
        print(f"🏗️  SNN TensorFlow Model Architecture:")
        print(f"    Input Size: {input_size}")
        print(f"    Hidden Sizes: {hidden_sizes}")
        print(f"    Output Size: {output_size}")
        print(f"    Timesteps: {num_timesteps}")
        print(f"    Total Layers: {len(self.spiking_layers) + 1}")
    
    def call(self, inputs, training=None):
        """
        Args:
            inputs: (batch_size, timesteps, input_size)
        
        Returns:
            output: (batch_size, output_size) - class logits
            spike_history: list of spike patterns for visualization
        """
        current_input = inputs
        spike_history = []
        
        # Forward through spiking layers
        for i, spiking_layer in enumerate(self.spiking_layers):
            spikes, spike_counts = spiking_layer(current_input, training=training)
            spike_history.append(spikes)
            
            # Use spike_counts as input to next layer
            # Expand to match timestep dimension
            current_input = tf.expand_dims(spike_counts, axis=1)
            current_input = tf.tile(current_input, [1, self.num_timesteps, 1])
        
        # Final output classification from last spike counts
        _, final_spike_counts = self.spiking_layers[-1](current_input, training=training)
        output = self.output_layer(final_spike_counts)
        
        return output, spike_history
    
    def get_config(self):
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'num_timesteps': self.num_timesteps,
        }

# ============================================
# 4. RANDOM SPIKE DATASET GENERATOR
# ============================================
def create_random_spike_dataset(num_samples=10000, input_size=100, output_size=10,
                                num_timesteps=10, spike_probability=0.3):
    """Generate random spike train dataset"""
    print(f"📊 Generating Random Spike Dataset...")
    
    # Generate random spike trains (binary)
    X = np.random.binomial(1, spike_probability, 
                          size=(num_samples, num_timesteps, input_size)).astype(np.float32)
    
    # Generate random labels
    y = np.random.randint(0, output_size, num_samples)
    
    print(f"    Samples: {num_samples}")
    print(f"    Input Shape: {X.shape}")
    print(f"    Output Shape: {y.shape}")
    print(f"    Memory: {X.nbytes / 1e9:.2f} GB")
    
    return X, y

# ============================================
# 5. CUSTOM TRAINING LOOP (For better control)
# ============================================
class SNNTrainer:
    """Custom trainer with mixed precision support"""
    
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    @tf.function
    def train_step(self, x, y):
        """Single training step with automatic mixed precision"""
        with tf.GradientTape() as tape:
            # Forward pass
            logits, _ = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
            
            # Scale loss for mixed precision
            scaled_loss = loss * tf.cast(
                tf.keras.mixed_precision.global_policy().compute_dtype == 'float16',
                loss.dtype
            ) or loss
        
        # Backward pass
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_weights)
        
        # Unscale and clip gradients
        gradients = scaled_gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(clipped_gradients, self.model.trainable_weights))
        
        return loss
    
    @tf.function
    def test_step(self, x, y):
        """Single test step"""
        logits, _ = self.model(x, training=False)
        loss = self.loss_fn(y, logits)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, axis=1), y), tf.float32)
        )
        return loss, accuracy

# ============================================
# 6. TRAINING SCRIPT
# ============================================
def main():
    print("\n🧠 Spiking Neural Network - TensorFlow Implementation")
    print("=" * 70 + "\n")
    
    # ===== HYPERPARAMETERS =====
    input_size = 100
    hidden_sizes = [256, 128]
    output_size = 10
    num_timesteps = 10
    batch_size = 256
    num_epochs = 25
    learning_rate = 0.001
    
    print(f"⚙️  Hyperparameters:")
    print(f"    Input Size: {input_size}")
    print(f"    Hidden Sizes: {hidden_sizes}")
    print(f"    Output Size: {output_size}")
    print(f"    Timesteps: {num_timesteps}")
    print(f"    Batch Size: {batch_size}")
    print(f"    Learning Rate: {learning_rate}")
    print(f"    Epochs: {num_epochs}\n")
    
    # ===== CREATE DATASETS =====
    print("📂 Creating Datasets...")
    X_train, y_train = create_random_spike_dataset(
        num_samples=20000,
        input_size=input_size,
        output_size=output_size,
        num_timesteps=num_timesteps,
        spike_probability=0.3
    )
    
    X_test, y_test = create_random_spike_dataset(
        num_samples=4000,
        input_size=input_size,
        output_size=output_size,
        num_timesteps=num_timesteps,
        spike_probability=0.3
    )
    
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"    Train batches: {len(list(train_dataset))}")
    print(f"    Test batches: {len(list(test_dataset))}\n")
    
    # ===== BUILD MODEL =====
    print("🏗️  Building SNN Model...")
    model = SNNModelTF(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        tau=0.25,
        vth=1.0,
        num_timesteps=num_timesteps
    )
    
    # Build with dummy input
    dummy_input = tf.random.normal((1, num_timesteps, input_size))
    _ = model(dummy_input)
    
    print(f"\n📊 Model Summary:")
    model.summary()
    print()
    
    # ===== OPTIMIZER & LOSS =====
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    trainer = SNNTrainer(model, optimizer, loss_fn)
    
    # ===== CALLBACKS =====
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'best_snn_tf_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    # ===== TRAINING LOOP =====
    print("🚀 Starting Training...\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_accuracy = 0.0
    start_training_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} | {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*70}")
        
        epoch_start = time.time()
        
        # ===== TRAINING =====
        train_loss = []
        train_accuracy = []
        
        with tf.keras.utils.prog_bar(total=len(list(train_dataset)), stateful_metrics=['loss']) as pbar:
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss = trainer.train_step(x_batch, y_batch)
                train_loss.append(loss.numpy())
                
                # Calculate accuracy
                logits, _ = model(x_batch, training=False)
                acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(logits, axis=1), y_batch), tf.float32)
                )
                train_accuracy.append(acc.numpy())
                
                pbar.update(1, values=[('loss', np.mean(train_loss))])
        
        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_accuracy) * 100
        
        # ===== VALIDATION =====
        val_loss = []
        val_accuracy = []
        
        for x_batch, y_batch in test_dataset:
            loss, acc = trainer.test_step(x_batch, y_batch)
            val_loss.append(loss.numpy())
            val_accuracy.append(acc.numpy())
        
        avg_val_loss = np.mean(val_loss)
        avg_val_acc = np.mean(val_accuracy) * 100
        
        epoch_time = time.time() - epoch_start
        
        # ===== RESULTS =====
        print(f"\n📈 Results:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.2f}%")
        print(f"    Time: {epoch_time:.2f}s")
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        history['epoch_time'].append(epoch_time)
        
        # Save best model
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            model.save('best_snn_tf_model.h5')
            print(f"    ✅ Best model saved! Accuracy: {best_val_accuracy:.2f}%")
        
        print()
    
    total_training_time = time.time() - start_training_time
    
    print(f"\n{'='*70}")
    print(f"✅ Training Complete!")
    print(f"    Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"    Total Training Time: {total_training_time/60:.2f} minutes")
    print(f"    Average Epoch Time: {np.mean(history['epoch_time']):.2f}s")
    print(f"{'='*70}\n")
    
    # ============================================
    # 7. VISUALIZATION
    # ============================================
    print("📊 Generating Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Epoch time
    axes[1, 0].plot(history['epoch_time'], marker='o', color='green', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_title('Training Time per Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Spike visualization
    sample_input = X_test[:1]  # Single sample
    sample_input_tf = tf.constant(sample_input, dtype=tf.float32)
    logits, spike_history = model(sample_input_tf, training=False)
    
    # Visualize input spikes
    axes[1, 1].imshow(sample_input[0].T, cmap='gray', aspect='auto')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('Input Neuron')
    axes[1, 1].set_title('Input Spike Pattern')
    
    plt.tight_layout()
    plt.savefig('snn_tensorflow_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: snn_tensorflow_results.png")
    plt.show()
    
    # ============================================
    # 8. PREDICTIONS & SPIKE RASTER
    # ============================================
    print("\n🔮 Making Predictions...")
    
    logits, spike_history = model(tf.constant(X_test[:10], dtype=tf.float32), training=False)
    predictions = tf.argmax(logits, axis=1).numpy()
    
    print(f"\nSample Predictions (first 10):")
    print(f"True Labels:       {y_test[:10]}")
    print(f"Predicted Labels:  {predictions}")
    print(f"Matches:           {predictions == y_test[:10]}")
    
    # Spike raster plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sample_spikes = spike_history[0][0].numpy()  # First layer, first sample
    
    # Input spike raster
    input_spikes = X_test[0]
    spike_times_in = []
    spike_neurons_in = []
    for neuron in range(input_spikes.shape[1]):
        for time in range(input_spikes.shape[0]):
            if input_spikes[time, neuron] > 0:
                spike_neurons_in.append(neuron)
                spike_times_in.append(time)
    
    axes[0].scatter(spike_times_in, spike_neurons_in, s=10, alpha=0.6, color='blue')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Input Neuron')
    axes[0].set_title('Input Layer - Spike Raster')
    axes[0].grid(True, alpha=0.3)
    
    # Hidden layer spike raster
    spike_times_h = []
    spike_neurons_h = []
    for neuron in range(sample_spikes.shape[1]):
        for time in range(sample_spikes.shape[0]):
            if sample_spikes[time, neuron] > 0:
                spike_neurons_h.append(neuron)
                spike_times_h.append(time)
    
    axes[1].scatter(spike_times_h, spike_neurons_h, s=10, alpha=0.6, color='red')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Hidden Neuron')
    axes[1].set_title('Hidden Layer - Spike Raster')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_tensorflow_spike_raster.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: snn_tensorflow_spike_raster.png")
    plt.show()
    
    # ============================================
    # 9. SAVE MODEL
    # ============================================
    print("\n💾 Saving Models...")
    
    # Save in H5 format
    model.save('snn_tensorflow_final.h5')
    print("✓ Saved: snn_tensorflow_final.h5")
    
    # Save in SavedModel format
    model.save('snn_tensorflow_savedmodel')
    print("✓ Saved: snn_tensorflow_savedmodel/")
    
    # Save training history
    import json
    with open('training_history.json', 'w') as f:
        json.dump({
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'best_accuracy': float(best_val_accuracy)
        }, f, indent=2)
    print("✓ Saved: training_history.json")
    
    print("\n" + "="*70)
    print("✅ SNN TensorFlow Training Complete!")
    print("="*70)

if __name__ == "__main__":
    main()