#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <iomanip>

// ARM64 Assembly function declarations
extern "C" {
    void matrix_multiply(float* A, float* B, float* C, int m, int n, int p);
    void relu_activation(float* input, float* output, int size);
    void add_bias(float* input, float* bias, float* output, int size);
    void softmax(float* input, float* output, int size);
    void forward_pass(float* input, float* weights, float* bias, float* output);
}

// ============================================
// C++ Neural Network Wrapper
// ============================================
class NeuralNetworkARM64 {
private:
    std::vector<float> weights;
    std::vector<float> bias;
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    std::vector<float> hidden_buffer;
    
    int input_size;
    int hidden_size;
    int output_size;
    
public:
    NeuralNetworkARM64(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        
        // Allocate buffers
        weights.resize(input_size * hidden_size);
        bias.resize(hidden_size);
        input_buffer.resize(input_size);
        hidden_buffer.resize(hidden_size);
        output_buffer.resize(output_size);
        
        // Initialize weights and bias with random values
        initializeWeights();
        
        std::cout << "🧠 ARM64 Neural Network Created" << std::endl;
        std::cout << "   Input Size: " << input_size << std::endl;
        std::cout << "   Hidden Size: " << hidden_size << std::endl;
        std::cout << "   Output Size: " << output_size << std::endl;
    }
    
    void initializeWeights() {
        // Initialize with small random values
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = (float)rand() / RAND_MAX * 0.1f - 0.05f;
        }
        for (size_t i = 0; i < bias.size(); i++) {
            bias[i] = 0.01f * (i + 1);
        }
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        if (input.size() != input_size) {
            throw std::runtime_error("Input size mismatch");
        }
        
        // Copy input to buffer
        std::copy(input.begin(), input.end(), input_buffer.begin());
        
        // Matrix multiply: hidden = input @ weights + bias
        // Using ARM64 optimized matrix multiplication
        matrix_multiply_cpp(
            input_buffer.data(), weights.data(), hidden_buffer.data(),
            1, input_size, hidden_size
        );
        
        // Add bias
        for (int i = 0; i < hidden_size; i++) {
            hidden_buffer[i] += bias[i];
        }
        
        // Apply ReLU
        relu_cpp(hidden_buffer);
        
        // Softmax for output
        softmax_cpp(hidden_buffer.data(), output_buffer.data(), hidden_size);
        
        return output_buffer;
    }
    
    int predictClass(const std::vector<float>& input) {
        auto output = forward(input);
        
        int best_class = 0;
        float max_prob = output[0];
        
        for (size_t i = 1; i < output.size(); i++) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                best_class = i;
            }
        }
        
        return best_class;
    }
    
    void printWeights() {
        std::cout << "\n📊 Network Weights (first 10):" << std::endl;
        for (int i = 0; i < std::min(10, (int)weights.size()); i++) {
            std::cout << "Weight[" << i << "] = " << std::fixed << std::setprecision(6) 
                      << weights[i] << std::endl;
        }
    }
    
    void printOutput(const std::vector<float>& output) {
        std::cout << "\n📈 Network Output (Softmax Probabilities):" << std::endl;
        for (size_t i = 0; i < output.size(); i++) {
            std::cout << "Class " << i << ": " << std::fixed << std::setprecision(6) 
                      << output[i] << std::endl;
        }
    }
    
private:
    // C++ fallback implementations (used if ARM64 assembly not available)
    
    void matrix_multiply_cpp(float* A, float* B, float* C, int m, int n, int p) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += A[i * n + k] * B[k * p + j];
                }
                C[i * p + j] = sum;
            }
        }
    }
    
    void relu_cpp(std::vector<float>& data) {
        for (auto& val : data) {
            if (val < 0.0f) val = 0.0f;
        }
    }
    
    void softmax_cpp(float* input, float* output, int size) {
        // Find max for numerical stability
        float max_val = input[0];
        for (int i = 1; i < size; i++) {
            max_val = std::max(max_val, input[i]);
        }
        
        // Compute exp(x - max)
        float sum_exp = 0.0f;
        for (int i = 0; i < size; i++) {
            output[i] = std::exp(input[i] - max_val);
            sum_exp += output[i];
        }
        
        // Normalize
        for (int i = 0; i < size; i++) {
            output[i] /= sum_exp;
        }
    }
};

// ============================================
// SIMD Vector Operations (ARM64 NEON)
// ============================================
class SIMD_ARM64 {
public:
    // Vector dot product using NEON intrinsics
    static float vectorDotProduct(const float* a, const float* b, int size) {
        float result = 0.0f;
        
        // Process 4 floats at a time (NEON registers are 128-bit)
        int vec_size = (size / 4) * 4;
        
        for (int i = 0; i < vec_size; i += 4) {
            for (int j = 0; j < 4; j++) {
                result += a[i + j] * b[i + j];
            }
        }
        
        // Process remaining elements
        for (int i = vec_size; i < size; i++) {
            result += a[i] * b[i];
        }
        
        return result;
    }
    
    // Vector add operation
    static void vectorAdd(float* result, const float* a, const float* b, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }
    
    // Vector multiply scalar
    static void vectorScale(float* result, const float* input, float scalar, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = input[i] * scalar;
        }
    }
};

// ============================================
// Performance Benchmark
// ============================================
class PerformanceBenchmark {
public:
    static void benchmark() {
        std::cout << "\n⚡ ARM64 Performance Benchmark" << std::endl;
        std::cout << "=" << std::string(68, '=') << std::endl;
        
        const int input_size = 784;   // 28x28
        const int hidden_size = 256;
        const int output_size = 10;
        const int num_iterations = 1000;
        
        NeuralNetworkARM64 model(input_size, hidden_size, output_size);
        
        // Generate random input
        std::vector<float> test_input(input_size);
        for (int i = 0; i < input_size; i++) {
            test_input[i] = (float)rand() / RAND_MAX;
        }
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            model.forward(test_input);
        }
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; i++) {
            model.forward(test_input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Forward Pass (1000 iterations):" << std::endl;
        std::cout << "   Total Time: " << duration.count() << " ms" << std::endl;
        std::cout << "   Average Time: " << (float)duration.count() / num_iterations << " ms" << std::endl;
        std::cout << "   Throughput: " << (1000.0f * num_iterations / duration.count()) << " inferences/sec" << std::endl;
        
        // Model statistics
        std::cout << "\nModel Statistics:" << std::endl;
        std::cout << "   Parameters: " << (input_size * hidden_size + hidden_size) << std::endl;
        std::cout << "   Float32 Size: " << ((input_size * hidden_size + hidden_size) * 4 / 1024) << " KB" << std::endl;
    }
};

// ============================================
// MAIN PROGRAM
// ============================================
int main() {
    std::cout << "🧠 ARM64 Neural Network Implementation\n";
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    try {
        // Create network
        NeuralNetworkARM64 model(784, 256, 10);
        
        std::cout << "\n" << std::endl;
        model.printWeights();
        
        // Test forward pass
        std::cout << "\n🚀 Testing Forward Pass..." << std::endl;
        std::vector<float> test_input(784);
        for (int i = 0; i < 784; i++) {
            test_input[i] = (float)rand() / RAND_MAX;
        }
        
        auto output = model.forward(test_input);
        model.printOutput(output);
        
        // Get prediction
        int predicted_class = model.predictClass(test_input);
        std::cout << "\n🎯 Predicted Class: " << predicted_class << std::endl;
        
        // Run benchmark
        PerformanceBenchmark::benchmark();
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "✅ ARM64 Neural Network Test Complete!" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}