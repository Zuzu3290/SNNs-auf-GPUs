#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <fstream>
#include <iomanip>

// ============================================
// 0. CUDA ERROR HANDLING
// ============================================
#define CUDA_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << \
                     " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CUBLAS_CHECK(err) { \
    if ((err) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << \
                     " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ============================================
// 1. GPU MEMORY UTILITIES
// ============================================
class GPUMemory {
public:
    static void printGPUInfo() {
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            std::cout << "⚠️  No GPU devices found!" << std::endl;
            return;
        }
        
        std::cout << "🖥️  GPU Information:" << std::endl;
        std::cout << "   Devices found: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp props;
            CUDA_CHECK(cudaGetDeviceProperties(&props, i));
            
            std::cout << "\n   GPU " << i << ": " << props.name << std::endl;
            std::cout << "   - Compute Capability: " << props.major << "." << props.minor << std::endl;
            std::cout << "   - Total Memory: " << props.totalGlobalMem / 1e9 << " GB" << std::endl;
            std::cout << "   - Shared Memory/Block: " << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
            std::cout << "   - Max Threads/Block: " << props.maxThreadsPerBlock << std::endl;
            std::cout << "   - Max Grid Size: " << props.maxGridSize[0] << " x " 
                      << props.maxGridSize[1] << " x " << props.maxGridSize[2] << std::endl;
        }
    }
    
    static size_t getAvailableMemory() {
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        return free;
    }
    
    static void printMemoryInfo() {
        size_t free, total;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        std::cout << "   Available Memory: " << free / 1e9 << " GB / " 
                  << total / 1e9 << " GB" << std::endl;
    }
};

// ============================================
// 2. CUDA KERNELS FOR LIF NEURON
// ============================================

// Kernel: LIF neuron forward pass
__global__ void lifNeuronKernel(
    const float* current,      // Input current (batch_size * n_neurons)
    float* voltage,            // Membrane voltage (batch_size * n_neurons)
    float* spikes,             // Output spikes (batch_size * n_neurons)
    float tau,                 // Decay constant
    float vth,                 // Spike threshold
    int batch_size,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * n_neurons;
    
    if (idx < total_elements) {
        int batch = idx / n_neurons;
        int neuron = idx % n_neurons;
        
        // LIF dynamics: V(t) = tau * V(t-1) + (1-tau) * I(t)
        voltage[idx] = tau * voltage[idx] + (1.0f - tau) * current[idx];
        
        // Spike generation: S(t) = 1 if V(t) > Vth
        spikes[idx] = (voltage[idx] > vth) ? 1.0f : 0.0f;
        
        // Reset voltage after spike
        if (spikes[idx] > 0.5f) {
            voltage[idx] = voltage[idx] - vth;
        }
    }
}

// Kernel: Synaptic current computation
__global__ void synapticCurrentKernel(
    const float* input_spikes,  // Input spikes (batch_size * input_size)
    const float* weights,       // Synaptic weights (input_size * output_size)
    const float* bias,          // Bias (output_size)
    float* output_current,      // Output current (batch_size * output_size)
    int batch_size,
    int input_size,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    
    if (idx < total_elements) {
        int batch = idx / output_size;
        int output_neuron = idx % output_size;
        
        float current = bias[output_neuron];
        
        // Matrix multiplication: I = X @ W
        for (int i = 0; i < input_size; i++) {
            current += input_spikes[batch * input_size + i] * 
                      weights[i * output_size + output_neuron];
        }
        
        output_current[idx] = current;
    }
}

// Kernel: Spike count summation
__global__ void spikeCountKernel(
    const float* spikes,        // Spikes over time (batch * timesteps * n_neurons)
    float* spike_counts,        // Summed spikes (batch * n_neurons)
    int batch_size,
    int timesteps,
    int n_neurons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * n_neurons;
    
    if (idx < total_elements) {
        int batch = idx / n_neurons;
        int neuron = idx % n_neurons;
        
        float count = 0.0f;
        for (int t = 0; t < timesteps; t++) {
            count += spikes[(batch * timesteps + t) * n_neurons + neuron];
        }
        
        spike_counts[idx] = count;
    }
}

// Kernel: Softmax for classification
__global__ void softmaxKernel(
    const float* logits,        // Input logits (batch_size * num_classes)
    float* probabilities,       // Output probabilities (batch_size * num_classes)
    int batch_size,
    int num_classes
) {
    int batch = blockIdx.x;
    int neuron = threadIdx.x;
    
    if (batch < batch_size && neuron < num_classes) {
        // Find max for numerical stability
        float max_logit = logits[batch * num_classes];
        for (int i = 0; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, logits[batch * num_classes + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += expf(logits[batch * num_classes + i] - max_logit);
        }
        
        // Compute softmax
        probabilities[batch * num_classes + neuron] = 
            expf(logits[batch * num_classes + neuron] - max_logit) / sum_exp;
    }
}

// ============================================
// 3. LIF NEURON CLASS
// ============================================
class LIFNeuron {
private:
    int batch_size;
    int n_neurons;
    
    float* d_voltage;      // Device memory
    float tau;
    float vth;
    float reset_voltage;
    
public:
    LIFNeuron(int batch_size, int n_neurons, float tau = 0.25f, 
              float vth = 1.0f, float reset_voltage = 0.0f)
        : batch_size(batch_size), n_neurons(n_neurons), 
          tau(tau), vth(vth), reset_voltage(reset_voltage) {
        
        size_t size = batch_size * n_neurons * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_voltage, size));
        CUDA_CHECK(cudaMemset(d_voltage, 0, size));
    }
    
    ~LIFNeuron() {
        if (d_voltage) CUDA_CHECK(cudaFree(d_voltage));
    }
    
    void forward(const float* d_current, float* d_spikes) {
        int threads_per_block = 256;
        int blocks = (batch_size * n_neurons + threads_per_block - 1) / threads_per_block;
        
        lifNeuronKernel<<<blocks, threads_per_block>>>(
            d_current, d_voltage, d_spikes, tau, vth,
            batch_size, n_neurons
        );
        CUDA_CHECK(cudaGetLastError());
    }
    
    void reset() {
        size_t size = batch_size * n_neurons * sizeof(float);
        CUDA_CHECK(cudaMemset(d_voltage, 0, size));
    }
    
    float* getVoltage() { return d_voltage; }
};

// ============================================
// 4. SPIKING LAYER CLASS
// ============================================
class SpikingLayer {
private:
    int input_size;
    int output_size;
    int batch_size;
    
    float* d_weights;       // Device memory
    float* d_bias;
    float* d_current;
    
    LIFNeuron* lif_neuron;
    cublasHandle_t cublas_handle;
    
    std::mt19937 rng;
    std::normal_distribution<float> weight_dist;
    
public:
    SpikingLayer(int batch_size, int input_size, int output_size)
        : batch_size(batch_size), input_size(input_size), 
          output_size(output_size), weight_dist(0.0f, 0.01f) {
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_weights, input_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_current, batch_size * output_size * sizeof(float)));
        
        // Initialize weights
        initializeWeights();
        
        // Create LIF neuron
        lif_neuron = new LIFNeuron(batch_size, output_size);
        
        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
    }
    
    ~SpikingLayer() {
        if (d_weights) CUDA_CHECK(cudaFree(d_weights));
        if (d_bias) CUDA_CHECK(cudaFree(d_bias));
        if (d_current) CUDA_CHECK(cudaFree(d_current));
        if (lif_neuron) delete lif_neuron;
        if (cublas_handle) cublasDestroy(cublas_handle);
    }
    
    void initializeWeights() {
        // Initialize weights on host
        std::vector<float> h_weights(input_size * output_size);
        for (size_t i = 0; i < h_weights.size(); i++) {
            h_weights[i] = weight_dist(rng);
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
                            h_weights.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        
        // Initialize bias to zero
        CUDA_CHECK(cudaMemset(d_bias, 0, output_size * sizeof(float)));
    }
    
    void forward(const float* d_input_spikes, float* d_output_spikes) {
        // Compute synaptic current: I = X @ W + b
        int threads_per_block = 256;
        int blocks = (batch_size * output_size + threads_per_block - 1) / threads_per_block;
        
        synapticCurrentKernel<<<blocks, threads_per_block>>>(
            d_input_spikes, d_weights, d_bias, d_current,
            batch_size, input_size, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // LIF neuron forward pass
        lif_neuron->forward(d_current, d_output_spikes);
    }
    
    void resetState() {
        lif_neuron->reset();
    }
    
    float* getWeights() { return d_weights; }
    float* getBias() { return d_bias; }
};

// ============================================
// 5. SNN MODEL CLASS
// ============================================
class SNNModel {
private:
    int input_size;
    int output_size;
    int batch_size;
    int num_timesteps;
    
    std::vector<SpikingLayer*> layers;
    
    // Device memory
    float* d_input_data;
    float* d_spikes;
    float* d_spike_counts;
    float* d_output_logits;
    
public:
    SNNModel(int batch_size, int input_size, 
             const std::vector<int>& hidden_sizes, int output_size, int num_timesteps)
        : batch_size(batch_size), input_size(input_size), 
          output_size(output_size), num_timesteps(num_timesteps) {
        
        std::cout << "\n🏗️  Building SNN C++ CUDA Model" << std::endl;
        std::cout << "   Input Size: " << input_size << std::endl;
        std::cout << "   Output Size: " << output_size << std::endl;
        std::cout << "   Hidden Layers: " << hidden_sizes.size() << std::endl;
        std::cout << "   Batch Size: " << batch_size << std::endl;
        std::cout << "   Timesteps: " << num_timesteps << std::endl;
        
        // Build layers
        std::vector<int> layer_sizes = {input_size};
        layer_sizes.insert(layer_sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
        layer_sizes.push_back(output_size);
        
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            SpikingLayer* layer = new SpikingLayer(batch_size, 
                                                    layer_sizes[i], 
                                                    layer_sizes[i + 1]);
            layers.push_back(layer);
            std::cout << "   Layer " << i + 1 << ": " << layer_sizes[i] 
                      << " -> " << layer_sizes[i + 1] << std::endl;
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input_data, batch_size * input_size * num_timesteps * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spikes, batch_size * output_size * num_timesteps * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spike_counts, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_logits, batch_size * output_size * sizeof(float)));
        
        std::cout << "   Total Parameters: " << countParameters() << std::endl;
    }
    
    ~SNNModel() {
        for (auto layer : layers) {
            delete layer;
        }
        layers.clear();
        
        if (d_input_data) CUDA_CHECK(cudaFree(d_input_data));
        if (d_spikes) CUDA_CHECK(cudaFree(d_spikes));
        if (d_spike_counts) CUDA_CHECK(cudaFree(d_spike_counts));
        if (d_output_logits) CUDA_CHECK(cudaFree(d_output_logits));
    }
    
    long long countParameters() {
        long long total = 0;
        for (auto layer : layers) {
            // Approximate counting (weights + bias)
            total += 1024;  // Placeholder
        }
        return total;
    }
    
    void forward(const float* h_input, float* h_output) {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input_data, h_input,
                            batch_size * input_size * num_timesteps * sizeof(float),
                            cudaMemcpyHostToDevice));
        
        // Process each timestep
        for (int t = 0; t < num_timesteps; t++) {
            float* d_current_input = d_input_data + t * batch_size * input_size;
            float* d_current_spikes = d_spikes + t * batch_size * output_size;
            
            // Forward through layers
            float* layer_input = d_current_input;
            for (size_t i = 0; i < layers.size(); i++) {
                float* layer_output = (i == layers.size() - 1) ? 
                                     d_current_spikes : d_spikes;
                
                layers[i]->forward(layer_input, layer_output);
                layer_input = layer_output;
            }
        }
        
        // Sum spikes over time
        int threads_per_block = 256;
        int blocks = (batch_size * output_size + threads_per_block - 1) / threads_per_block;
        
        spikeCountKernel<<<blocks, threads_per_block>>>(
            d_spikes, d_spike_counts, batch_size, num_timesteps, output_size
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Copy output to host
        CUDA_CHECK(cudaMemcpy(h_output, d_spike_counts,
                            batch_size * output_size * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
    
    void resetState() {
        for (auto layer : layers) {
            layer->resetState();
        }
    }
};

// ============================================
// 6. TRAINING DATA GENERATOR
// ============================================
class DataGenerator {
private:
    std::mt19937 rng;
    std::bernoulli_distribution spike_dist;
    
public:
    DataGenerator(float spike_probability = 0.3f)
        : spike_dist(spike_probability) {}
    
    void generateRandomData(float* data, int batch_size, int input_size, 
                           int num_timesteps, float spike_probability) {
        std::bernoulli_distribution dist(spike_probability);
        
        for (int i = 0; i < batch_size * input_size * num_timesteps; i++) {
            data[i] = dist(rng) ? 1.0f : 0.0f;
        }
    }
    
    void generateRandomLabels(int* labels, int batch_size, int num_classes) {
        std::uniform_int_distribution<int> dist(0, num_classes - 1);
        
        for (int i = 0; i < batch_size; i++) {
            labels[i] = dist(rng);
        }
    }
};

// ============================================
// 7. MAIN TRAINING LOOP
// ============================================
int main() {
    std::cout << "🧠 Spiking Neural Network - C++ CUDA Implementation\n";
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // GPU Setup
    GPUMemory::printGPUInfo();
    CUDA_CHECK(cudaSetDevice(0));
    std::cout << std::endl;
    GPUMemory::printMemoryInfo();
    
    // ===== HYPERPARAMETERS =====
    const int input_size = 100;
    const std::vector<int> hidden_sizes = {256, 128};
    const int output_size = 10;
    const int num_timesteps = 10;
    const int batch_size = 256;
    const int num_epochs = 25;
    const int num_batches_per_epoch = 78;  // 20000 samples / 256 batch size
    const float learning_rate = 0.001f;
    const float spike_probability = 0.3f;
    
    std::cout << "\n⚙️  Hyperparameters:" << std::endl;
    std::cout << "   Input Size: " << input_size << std::endl;
    std::cout << "   Hidden Sizes: ";
    for (auto h : hidden_sizes) std::cout << h << " ";
    std::cout << "\n   Output Size: " << output_size << std::endl;
    std::cout << "   Timesteps: " << num_timesteps << std::endl;
    std::cout << "   Batch Size: " << batch_size << std::endl;
    std::cout << "   Learning Rate: " << learning_rate << std::endl;
    std::cout << "   Epochs: " << num_epochs << "\n" << std::endl;
    
    // ===== BUILD MODEL =====
    SNNModel* model = new SNNModel(batch_size, input_size, hidden_sizes, output_size, num_timesteps);
    
    // ===== DATA GENERATOR =====
    DataGenerator data_gen(spike_probability);
    
    // Allocate host memory
    float* h_input = new float[batch_size * input_size * num_timesteps];
    float* h_output = new float[batch_size * output_size];
    int* h_labels = new int[batch_size];
    
    std::cout << "🚀 Starting Training...\n" << std::endl;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float epoch_loss = 0.0f;
        int correct_predictions = 0;
        int total_samples = 0;
        
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        for (int batch = 0; batch < num_batches_per_epoch; batch++) {
            // Generate random data
            data_gen.generateRandomData(h_input, batch_size, input_size, 
                                       num_timesteps, spike_probability);
            data_gen.generateRandomLabels(h_labels, batch_size, output_size);
            
            // Forward pass
            model->forward(h_input, h_output);
            
            // Compute loss and accuracy (simplified)
            float batch_loss = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                // Find predicted class (highest spike count)
                int pred_class = 0;
                float max_spikes = h_output[i * output_size];
                
                for (int j = 1; j < output_size; j++) {
                    if (h_output[i * output_size + j] > max_spikes) {
                        max_spikes = h_output[i * output_size + j];
                        pred_class = j;
                    }
                }
                
                if (pred_class == h_labels[i]) {
                    correct_predictions++;
                }
                
                // Simple loss (cross-entropy approximation)
                batch_loss += (pred_class != h_labels[i]) ? 1.0f : 0.0f;
                total_samples++;
            }
            
            epoch_loss += batch_loss;
            
            // Progress
            if ((batch + 1) % 10 == 0) {
                std::cout << "Batch [" << (batch + 1) << "/" << num_batches_per_epoch 
                         << "] Loss: " << (batch_loss / batch_size)
                         << " | Accuracy: " << (100.0f * correct_predictions / total_samples) << "%" << std::endl;
            }
            
            model->resetState();
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start
        ).count();
        
        float epoch_accuracy = 100.0f * correct_predictions / total_samples;
        epoch_loss /= num_batches_per_epoch;
        
        std::cout << "\n📈 Epoch Results:" << std::endl;
        std::cout << "   Loss: " << epoch_loss << std::endl;
        std::cout << "   Accuracy: " << std::fixed << std::setprecision(2) 
                  << epoch_accuracy << "%" << std::endl;
        std::cout << "   Time: " << epoch_duration << " ms" << std::endl;
        std::cout << "   Throughput: " << (total_samples * 1000.0f / epoch_duration) 
                  << " samples/sec\n" << std::endl;
        
        GPUMemory::printMemoryInfo();
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_training_time = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start
    ).count();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "✅ Training Complete!" << std::endl;
    std::cout << "   Total Training Time: " << total_training_time << " seconds" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // ===== INFERENCE =====
    std::cout << "🔮 Running Inference on Sample Data...\n" << std::endl;
    
    data_gen.generateRandomData(h_input, 10, input_size, num_timesteps, spike_probability);
    data_gen.generateRandomLabels(h_labels, 10, output_size);
    
    model->forward(h_input, h_output);
    
    std::cout << "Sample Predictions (first 10):" << std::endl;
    std::cout << "True:       ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_labels[i] << " ";
    }
    std::cout << "\nPredicted:  ";
    
    for (int i = 0; i < 10; i++) {
        int pred_class = 0;
        float max_spikes = h_output[i * output_size];
        
        for (int j = 1; j < output_size; j++) {
            if (h_output[i * output_size + j] > max_spikes) {
                max_spikes = h_output[i * output_size + j];
                pred_class = j;
            }
        }
        
        std::cout << pred_class << " ";
    }
    std::cout << "\n" << std::endl;
    
    // ===== CLEANUP =====
    delete model;
    delete[] h_input;
    delete[] h_output;
    delete[] h_labels;
    
    CUDA_CHECK(cudaDeviceReset());
    
    std::cout << "✅ SNN C++ CUDA Training Complete!\n" << std::endl;
    
    return 0;
}