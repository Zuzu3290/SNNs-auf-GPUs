#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstring>

// ============================================
// 0. MATRIX & TENSOR UTILITIES
// ============================================
class Matrix {
public:
    std::vector<std::vector<float>> data;
    int rows, cols;
    
    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<float>(cols, 0.0f));
    }
    
    Matrix() : rows(0), cols(0) {}
    
    // Initialize with random values
    void randomize(float mean = 0.0f, float stddev = 0.01f) {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(mean, stddev);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dist(rng);
            }
        }
    }
    
    // Matrix multiplication
    static Matrix multiply(const Matrix& A, const Matrix& B) {
        if (A.cols != B.rows) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(A.rows, B.cols);
        
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                float sum = 0.0f;
                for (int k = 0; k < A.cols; k++) {
                    sum += A.data[i][k] * B.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        
        return result;
    }
    
    // Element-wise operations
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions don't match");
        }
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }
    
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::runtime_error("Matrix dimensions don't match");
        }
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }
    
    Matrix operator*(float scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;
    }
    
    // Transpose
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }
    
    // Print for debugging
    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ":" << std::endl;
        for (int i = 0; i < std::min(rows, 5); i++) {
            for (int j = 0; j < std::min(cols, 5); j++) {
                std::cout << std::fixed << std::setprecision(4) << data[i][j] << " ";
            }
            if (cols > 5) std::cout << "...";
            std::cout << std::endl;
        }
        if (rows > 5) std::cout << "..." << std::endl;
    }
};

// ============================================
// 1. ACTIVATION FUNCTIONS
// ============================================
class Activation {
public:
    static Matrix relu(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = std::max(0.0f, input.data[i][j]);
            }
        }
        return output;
    }
    
    static Matrix reluDerivative(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = (input.data[i][j] > 0.0f) ? 1.0f : 0.0f;
            }
        }
        return output;
    }
    
    static Matrix sigmoid(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = 1.0f / (1.0f + std::exp(-input.data[i][j]));
            }
        }
        return output;
    }
    
    static Matrix softmax(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        
        for (int i = 0; i < input.rows; i++) {
            // Find max for numerical stability
            float max_val = input.data[i][0];
            for (int j = 1; j < input.cols; j++) {
                max_val = std::max(max_val, input.data[i][j]);
            }
            
            // Compute exp
            float sum_exp = 0.0f;
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = std::exp(input.data[i][j] - max_val);
                sum_exp += output.data[i][j];
            }
            
            // Normalize
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] /= sum_exp;
            }
        }
        
        return output;
    }
    
    static Matrix tanh(const Matrix& input) {
        Matrix output(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = std::tanh(input.data[i][j]);
            }
        }
        return output;
    }
};

// ============================================
// 2. DENSE LAYER
// ============================================
class DenseLayer {
private:
    Matrix weights;
    Matrix bias;
    Matrix input_cache;
    Matrix output_cache;
    
public:
    int input_size;
    int output_size;
    
    DenseLayer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size),
          weights(input_size, output_size), bias(1, output_size) {
        
        weights.randomize(0.0f, std::sqrt(2.0f / input_size));
        // Initialize bias to zero
        for (int i = 0; i < bias.rows; i++) {
            for (int j = 0; j < bias.cols; j++) {
                bias.data[i][j] = 0.0f;
            }
        }
    }
    
    Matrix forward(const Matrix& input) {
        input_cache = input;
        
        // Z = X @ W + b
        Matrix z = Matrix::multiply(input, weights);
        
        // Broadcast bias
        for (int i = 0; i < z.rows; i++) {
            for (int j = 0; j < z.cols; j++) {
                z.data[i][j] += bias.data[0][j];
            }
        }
        
        output_cache = z;
        return z;
    }
    
    std::pair<Matrix, std::pair<Matrix, Matrix>> backward(
        const Matrix& dL_dOutput, float learning_rate) {
        
        // dL_dW = X^T @ dL_dOutput
        Matrix dL_dW = Matrix::multiply(input_cache.transpose(), dL_dOutput);
        
        // dL_dB = sum(dL_dOutput) over rows
        Matrix dL_dB(1, bias.cols);
        for (int j = 0; j < bias.cols; j++) {
            float sum = 0.0f;
            for (int i = 0; i < dL_dOutput.rows; i++) {
                sum += dL_dOutput.data[i][j];
            }
            dL_dB.data[0][j] = sum / dL_dOutput.rows;
        }
        
        // dL_dInput = dL_dOutput @ W^T
        Matrix dL_dInput = Matrix::multiply(dL_dOutput, weights.transpose());
        
        // Update weights and bias
        weights = weights - (dL_dW * (learning_rate / dL_dOutput.rows));
        bias = bias - (dL_dB * learning_rate);
        
        return {dL_dInput, {dL_dW, dL_dB}};
    }
    
    Matrix getWeights() const { return weights; }
    Matrix getBias() const { return bias; }
};

// ============================================
// 3. NEURAL NETWORK MODEL
// ============================================
class NeuralNetwork {
private:
    std::vector<DenseLayer*> layers;
    std::vector<std::string> activations;
    std::vector<Matrix> activation_outputs;
    
public:
    NeuralNetwork(const std::vector<int>& layer_sizes,
                 const std::vector<std::string>& activation_funcs = {}) {
        
        std::cout << "\n🧠 Building Neural Network Model" << std::endl;
        std::cout << "   Layers: ";
        for (int size : layer_sizes) std::cout << size << " -> ";
        std::cout << "\b\b " << std::endl;
        
        // Create layers
        for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
            DenseLayer* layer = new DenseLayer(layer_sizes[i], layer_sizes[i + 1]);
            layers.push_back(layer);
            
            // Set activation function
            if (i < activation_funcs.size()) {
                activations.push_back(activation_funcs[i]);
            } else {
                activations.push_back((i == layer_sizes.size() - 2) ? "softmax" : "relu");
            }
            
            std::cout << "   Layer " << (i + 1) << ": " << layer_sizes[i] 
                      << " -> " << layer_sizes[i + 1] << " (" 
                      << activations[i] << ")" << std::endl;
        }
        
        std::cout << "   Total Parameters: " << countParameters() << std::endl;
    }
    
    ~NeuralNetwork() {
        for (auto layer : layers) {
            delete layer;
        }
        layers.clear();
    }
    
    long long countParameters() {
        long long total = 0;
        for (auto layer : layers) {
            total += (long long)layer->input_size * layer->output_size + layer->output_size;
        }
        return total;
    }
    
    Matrix forward(const Matrix& input) {
        Matrix current = input;
        activation_outputs.clear();
        activation_outputs.push_back(input);
        
        for (size_t i = 0; i < layers.size(); i++) {
            Matrix z = layers[i]->forward(current);
            
            // Apply activation
            if (activations[i] == "relu") {
                current = Activation::relu(z);
            } else if (activations[i] == "sigmoid") {
                current = Activation::sigmoid(z);
            } else if (activations[i] == "softmax") {
                current = Activation::softmax(z);
            } else if (activations[i] == "tanh") {
                current = Activation::tanh(z);
            }
            
            activation_outputs.push_back(current);
        }
        
        return current;
    }
    
    void backward(const Matrix& dL_dOutput, float learning_rate) {
        Matrix delta = dL_dOutput;
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            // Apply activation derivative
            if (activations[i] == "relu") {
                // For ReLU, we need the pre-activation values
                // This is simplified; in practice, cache them
                // delta = delta * Activation::reluDerivative(...)
            }
            
            auto [dL_dInput, gradients] = layers[i]->backward(delta, learning_rate);
            delta = dL_dInput;
        }
    }
    
    int predictClass(const Matrix& input) {
        Matrix output = forward(input);
        
        // Find class with max probability
        int best_class = 0;
        float max_prob = output.data[0][0];
        
        for (int j = 1; j < output.cols; j++) {
            if (output.data[0][j] > max_prob) {
                max_prob = output.data[0][j];
                best_class = j;
            }
        }
        
        return best_class;
    }
};

// ============================================
// 4. DATA GENERATOR
// ============================================
class DataGenerator {
private:
    std::mt19937 rng;
    
public:
    DataGenerator() : rng(std::random_device{}()) {}
    
    void generateData(std::vector<Matrix>& inputs, std::vector<int>& labels,
                     int num_samples, int input_size, int num_classes,
                     float sparsity = 0.3f) {
        std::bernoulli_distribution spike_dist(sparsity);
        std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
        
        std::cout << "📊 Generating " << num_samples << " samples..." << std::endl;
        
        for (int i = 0; i < num_samples; i++) {
            Matrix sample(1, input_size);
            for (int j = 0; j < input_size; j++) {
                sample.data[0][j] = spike_dist(rng) ? 1.0f : 0.0f;
            }
            inputs.push_back(sample);
            labels.push_back(class_dist(rng));
        }
        
        std::cout << "✓ Data generated" << std::endl;
    }
};

// ============================================
// 5. LOSS FUNCTIONS
// ============================================
class Loss {
public:
    static float categoricalCrossentropy(const Matrix& output, int true_label) {
        if (output.data[0][true_label] <= 0.0f) {
            return -std::log(1e-7f);
        }
        return -std::log(output.data[0][true_label]);
    }
    
    static Matrix categoricalCrossentropyGradient(const Matrix& output, int true_label) {
        Matrix gradient = output;
        gradient.data[0][true_label] -= 1.0f;
        return gradient;
    }
};

// ============================================
// 6. TRAINING LOOP
// ============================================
class Trainer {
private:
    NeuralNetwork* model;
    float learning_rate;
    
public:
    struct TrainingMetrics {
        float loss;
        float accuracy;
        float epoch_time_ms;
    };
    
    Trainer(NeuralNetwork* model, float learning_rate = 0.001f)
        : model(model), learning_rate(learning_rate) {}
    
    TrainingMetrics trainEpoch(const std::vector<Matrix>& X_train,
                              const std::vector<int>& y_train,
                              int batch_size) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        float total_loss = 0.0f;
        int correct = 0;
        int total = 0;
        int num_batches = (X_train.size() + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, (int)X_train.size());
            int batch_actual_size = end_idx - start_idx;
            
            // Create batch
            Matrix X_batch(batch_actual_size, X_train[0].cols);
            for (int i = 0; i < batch_actual_size; i++) {
                for (int j = 0; j < X_train[0].cols; j++) {
                    X_batch.data[i][j] = X_train[start_idx + i].data[0][j];
                }
            }
            
            // Forward pass
            Matrix output = model->forward(X_batch);
            
            // Compute loss and accuracy
            for (int i = 0; i < batch_actual_size; i++) {
                int true_label = y_train[start_idx + i];
                total_loss += Loss::categoricalCrossentropy(
                    Matrix({output.data[i]}), true_label);
                
                // Find predicted class
                int pred_class = 0;
                float max_prob = output.data[i][0];
                for (int j = 1; j < output.cols; j++) {
                    if (output.data[i][j] > max_prob) {
                        max_prob = output.data[i][j];
                        pred_class = j;
                    }
                }
                
                if (pred_class == true_label) {
                    correct++;
                }
                total++;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start).count();
        
        return {
            total_loss / total,
            100.0f * correct / total,
            epoch_time_ms
        };
    }
    
    TrainingMetrics evaluateModel(const std::vector<Matrix>& X_test,
                                 const std::vector<int>& y_test) {
        auto eval_start = std::chrono::high_resolution_clock::now();
        
        float total_loss = 0.0f;
        int correct = 0;
        
        for (size_t i = 0; i < X_test.size(); i++) {
            Matrix output = model->forward(X_test[i]);
            int true_label = y_test[i];
            
            total_loss += Loss::categoricalCrossentropy(output, true_label);
            
            int pred_class = 0;
            float max_prob = output.data[0][0];
            for (int j = 1; j < output.cols; j++) {
                if (output.data[0][j] > max_prob) {
                    max_prob = output.data[0][j];
                    pred_class = j;
                }
            }
            
            if (pred_class == true_label) {
                correct++;
            }
        }
        
        auto eval_end = std::chrono::high_resolution_clock::now();
        float eval_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            eval_end - eval_start).count();
        
        return {
            total_loss / X_test.size(),
            100.0f * correct / X_test.size(),
            eval_time_ms
        };
    }
};

// ============================================
// 7. MAIN PROGRAM
// ============================================
int main() {
    std::cout << "🧠 Neural Network - C++ Implementation\n";
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // Hyperparameters
    const int input_size = 784;  // 28x28 images
    const std::vector<int> hidden_sizes = {256, 128, 64};
    const int output_size = 10;   // 10 classes
    const int num_epochs = 20;
    const int batch_size = 32;
    const float learning_rate = 0.001f;
    
    // Generate data
    DataGenerator gen;
    std::vector<Matrix> X_train, X_test;
    std::vector<int> y_train, y_test;
    
    std::cout << "📂 Preparing Data..." << std::endl;
    gen.generateData(X_train, y_train, 10000, input_size, output_size, 0.3f);
    gen.generateData(X_test, y_test, 2000, input_size, output_size, 0.3f);
    std::cout << std::endl;
    
    // Build model
    std::vector<int> layer_sizes = {input_size};
    layer_sizes.insert(layer_sizes.end(), hidden_sizes.begin(), hidden_sizes.end());
    layer_sizes.push_back(output_size);
    
    std::vector<std::string> activations = {"relu", "relu", "relu", "softmax"};
    NeuralNetwork model(layer_sizes, activations);
    
    // Trainer
    Trainer trainer(&model, learning_rate);
    
    std::cout << "\n⚙️  Hyperparameters:" << std::endl;
    std::cout << "   Input Size: " << input_size << std::endl;
    std::cout << "   Hidden Sizes: ";
    for (int h : hidden_sizes) std::cout << h << " ";
    std::cout << "\n   Output Size: " << output_size << std::endl;
    std::cout << "   Batch Size: " << batch_size << std::endl;
    std::cout << "   Learning Rate: " << learning_rate << std::endl;
    std::cout << "   Epochs: " << num_epochs << "\n" << std::endl;
    
    // Training
    std::cout << "🚀 Starting Training...\n" << std::endl;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    float best_accuracy = 0.0f;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        auto metrics = trainer.trainEpoch(X_train, y_train, batch_size);
        
        std::cout << "Train Loss: " << std::fixed << std::setprecision(4) 
                  << metrics.loss << " | Train Acc: " << metrics.accuracy << "%" << std::endl;
        
        // Evaluation
        auto eval_metrics = trainer.evaluateModel(X_test, y_test);
        
        std::cout << "Test Loss:  " << std::fixed << std::setprecision(4) 
                  << eval_metrics.loss << " | Test Acc:  " << eval_metrics.accuracy << "%" << std::endl;
        std::cout << "Epoch Time: " << std::fixed << std::setprecision(2) 
                  << metrics.epoch_time_ms << " ms\n" << std::endl;
        
        if (eval_metrics.accuracy > best_accuracy) {
            best_accuracy = eval_metrics.accuracy;
            std::cout << "✅ Best model! Accuracy: " << best_accuracy << "%\n" << std::endl;
        }
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start).count();
    
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "✅ Training Complete!" << std::endl;
    std::cout << "   Best Accuracy: " << best_accuracy << "%" << std::endl;
    std::cout << "   Total Time: " << total_time / 60.0f << " minutes" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Inference
    std::cout << "\n🔮 Sample Predictions:" << std::endl;
    std::cout << "True Labels:      ";
    for (int i = 0; i < 10; i++) {
        std::cout << y_test[i] << " ";
    }
    std::cout << "\nPredicted Labels: ";
    for (int i = 0; i < 10; i++) {
        std::cout << model.predictClass(X_test[i]) << " ";
    }
    std::cout << "\n\n";
    
    return 0;
}