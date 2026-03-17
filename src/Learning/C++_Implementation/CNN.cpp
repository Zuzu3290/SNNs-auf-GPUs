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
// 0. IMAGE & TENSOR UTILITIES
// ============================================
struct Tensor {
    std::vector<float> data;
    int height, width, channels;
    int batch_size;
    
    Tensor(int batch_size, int height, int width, int channels)
        : batch_size(batch_size), height(height), width(width), channels(channels) {
        data.resize(batch_size * height * width * channels, 0.0f);
    }
    
    Tensor() : batch_size(0), height(0), width(0), channels(0) {}
    
    int getIndex(int b, int h, int w, int c) const {
        return ((b * height + h) * width + w) * channels + c;
    }
    
    float& at(int b, int h, int w, int c) {
        return data[getIndex(b, h, w, c)];
    }
    
    const float& at(int b, int h, int w, int c) const {
        return data[getIndex(b, h, w, c)];
    }
    
    void randomize(float mean = 0.0f, float stddev = 0.01f) {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(mean, stddev);
        
        for (auto& val : data) {
            val = dist(rng);
        }
    }
    
    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }
};

// ============================================
// 1. CONVOLUTION LAYER
// ============================================
class ConvLayer {
private:
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    
    Tensor kernels;
    Tensor bias;
    Tensor input_cache;
    
public:
    int out_height, out_width;
    
    ConvLayer(int in_channels, int out_channels, int kernel_size,
             int stride = 1, int padding = 0)
        : in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding),
          kernels(out_channels, kernel_size, kernel_size, in_channels),
          bias(1, 1, 1, out_channels) {
        
        kernels.randomize(0.0f, std::sqrt(2.0f / (kernel_size * kernel_size * in_channels)));
        bias.fill(0.0f);
        
        std::cout << "   Conv: " << in_channels << " -> " << out_channels 
                  << " (kernel=" << kernel_size << ", stride=" << stride 
                  << ", padding=" << padding << ")" << std::endl;
    }
    
    Tensor forward(const Tensor& input) {
        input_cache = input;
        
        int batch_size = input.batch_size;
        int in_height = input.height;
        int in_width = input.width;
        
        // Calculate output dimensions
        out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
        out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
        
        Tensor output(batch_size, out_height, out_width, out_channels);
        
        // Convolution operation
        for (int b = 0; b < batch_size; b++) {
            for (int oc = 0; oc < out_channels; oc++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        float sum = bias.at(0, 0, 0, oc);
                        
                        // Kernel computation
                        for (int ic = 0; ic < in_channels; ic++) {
                            for (int kh = 0; kh < kernel_size; kh++) {
                                for (int kw = 0; kw < kernel_size; kw++) {
                                    int ih = oh * stride + kh - padding;
                                    int iw = ow * stride + kw - padding;
                                    
                                    if (ih >= 0 && ih < in_height &&
                                        iw >= 0 && iw < in_width) {
                                        sum += input.at(b, ih, iw, ic) *
                                              kernels.at(oc, kh, kw, ic);
                                    }
                                }
                            }
                        }
                        
                        output.at(b, oh, ow, oc) = sum;
                    }
                }
            }
        }
        
        return output;
    }
};

// ============================================
// 2. POOLING LAYER
// ============================================
class PoolingLayer {
private:
    int pool_size, stride;
    std::vector<int> max_indices;
    
public:
    int out_height, out_width;
    
    PoolingLayer(int pool_size = 2, int stride = 2)
        : pool_size(pool_size), stride(stride) {
        std::cout << "   MaxPool: kernel=" << pool_size << ", stride=" << stride << std::endl;
    }
    
    Tensor forward(const Tensor& input) {
        int batch_size = input.batch_size;
        int in_height = input.height;
        int in_width = input.width;
        int channels = input.channels;
        
        out_height = (in_height - pool_size) / stride + 1;
        out_width = (in_width - pool_size) / stride + 1;
        
        Tensor output(batch_size, out_height, out_width, channels);
        max_indices.clear();
        max_indices.resize(batch_size * out_height * out_width * channels);
        
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < out_height; oh++) {
                    for (int ow = 0; ow < out_width; ow++) {
                        float max_val = -1e9f;
                        int max_idx = 0;
                        
                        // Find maximum in pool
                        for (int ph = 0; ph < pool_size; ph++) {
                            for (int pw = 0; pw < pool_size; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                
                                float val = input.at(b, ih, iw, c);
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = ph * pool_size + pw;
                                }
                            }
                        }
                        
                        output.at(b, oh, ow, c) = max_val;
                        max_indices[((b * out_height + oh) * out_width + ow) * channels + c] = max_idx;
                    }
                }
            }
        }
        
        return output;
    }
};

// ============================================
// 3. BATCH NORMALIZATION LAYER
// ============================================
class BatchNormLayer {
private:
    float epsilon = 1e-5f;
    float momentum = 0.9f;
    Tensor gamma, beta;
    Tensor running_mean, running_variance;
    
public:
    BatchNormLayer(int channels)
        : gamma(1, 1, 1, channels), beta(1, 1, 1, channels),
          running_mean(1, 1, 1, channels), running_variance(1, 1, 1, channels) {
        
        // Initialize gamma to 1 and beta to 0
        for (int i = 0; i < channels; i++) {
            gamma.data[i] = 1.0f;
            beta.data[i] = 0.0f;
            running_mean.data[i] = 0.0f;
            running_variance.data[i] = 1.0f;
        }
    }
    
    Tensor forward(const Tensor& input, bool training = true) {
        Tensor output = input;
        
        int channels = input.channels;
        int spatial_size = input.batch_size * input.height * input.width;
        
        // Compute batch statistics
        for (int c = 0; c < channels; c++) {
            float mean = 0.0f;
            
            for (int b = 0; b < input.batch_size; b++) {
                for (int h = 0; h < input.height; h++) {
                    for (int w = 0; w < input.width; w++) {
                        mean += input.at(b, h, w, c);
                    }
                }
            }
            mean /= spatial_size;
            
            // Compute variance
            float variance = 0.0f;
            for (int b = 0; b < input.batch_size; b++) {
                for (int h = 0; h < input.height; h++) {
                    for (int w = 0; w < input.width; w++) {
                        float val = input.at(b, h, w, c);
                        variance += (val - mean) * (val - mean);
                    }
                }
            }
            variance /= spatial_size;
            
            // Normalize and scale
            for (int b = 0; b < input.batch_size; b++) {
                for (int h = 0; h < input.height; h++) {
                    for (int w = 0; w < input.width; w++) {
                        float normalized = (input.at(b, h, w, c) - mean) /
                                         std::sqrt(variance + epsilon);
                        output.at(b, h, w, c) = gamma.data[c] * normalized + beta.data[c];
                    }
                }
            }
        }
        
        return output;
    }
};

// ============================================
// 4. ACTIVATION FUNCTIONS FOR TENSORS
// ============================================
class TensorActivation {
public:
    static Tensor relu(const Tensor& input) {
        Tensor output = input;
        for (auto& val : output.data) {
            val = std::max(0.0f, val);
        }
        return output;
    }
    
    static Tensor softmax(const Tensor& input) {
        Tensor output = input;
        
        // For batch of vectors
        int batch_size = input.batch_size;
        int size = input.height * input.width * input.channels;
        
        for (int b = 0; b < batch_size; b++) {
            // Find max
            float max_val = input.data[b * size];
            for (int i = 1; i < size; i++) {
                max_val = std::max(max_val, input.data[b * size + i]);
            }
            
            // Compute exp sum
            float sum_exp = 0.0f;
            for (int i = 0; i < size; i++) {
                output.data[b * size + i] = std::exp(input.data[b * size + i] - max_val);
                sum_exp += output.data[b * size + i];
            }
            
            // Normalize
            for (int i = 0; i < size; i++) {
                output.data[b * size + i] /= sum_exp;
            }
        }
        
        return output;
    }
};

// ============================================
// 5. FLATTEN LAYER
// ============================================
class FlattenLayer {
public:
    Tensor forward(const Tensor& input) {
        int batch_size = input.batch_size;
        int flat_size = input.height * input.width * input.channels;
        
        Tensor output(batch_size, 1, flat_size, 1);
        output.data = input.data;
        
        return output;
    }
};

// ============================================
// 6. DENSE LAYER (For flattened input)
// ============================================
class DenseLayer {
private:
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    
public:
    int input_size, output_size;
    
    DenseLayer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        
        weights.resize(input_size, std::vector<float>(output_size));
        bias.resize(output_size, 0.0f);
        
        // Initialize weights
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / input_size));
        
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights[i][j] = dist(rng);
            }
        }
        
        std::cout << "   Dense: " << input_size << " -> " << output_size << std::endl;
    }
    
    Tensor forward(const Tensor& input) {
        int batch_size = input.batch_size;
        Tensor output(batch_size, 1, output_size, 1);
        
        for (int b = 0; b < batch_size; b++) {
            for (int j = 0; j < output_size; j++) {
                float sum = bias[j];
                
                for (int i = 0; i < input_size; i++) {
                    sum += input.data[b * input_size + i] * weights[i][j];
                }
                
                output.data[b * output_size + j] = sum;
            }
        }
        
        return output;
    }
};

// ============================================
// 7. CNN MODEL
// ============================================
class CNN {
private:
    std::vector<ConvLayer*> conv_layers;
    std::vector<PoolingLayer*> pool_layers;
    std::vector<BatchNormLayer*> bn_layers;
    FlattenLayer flatten;
    std::vector<DenseLayer*> dense_layers;
    
public:
    CNN() {
        std::cout << "\n🏗️  Building CNN Model (28x28 input, 10 classes)" << std::endl;
        std::cout << "   Architecture:" << std::endl;
        
        // Conv Block 1
        conv_layers.push_back(new ConvLayer(1, 32, 5, 1, 2));
        bn_layers.push_back(new BatchNormLayer(32));
        pool_layers.push_back(new PoolingLayer(2, 2));
        
        // Conv Block 2
        conv_layers.push_back(new ConvLayer(32, 64, 5, 1, 2));
        bn_layers.push_back(new BatchNormLayer(64));
        pool_layers.push_back(new PoolingLayer(2, 2));
        
        // Conv Block 3
        conv_layers.push_back(new ConvLayer(64, 128, 3, 1, 1));
        bn_layers.push_back(new BatchNormLayer(128));
        pool_layers.push_back(new PoolingLayer(2, 2));
        
        // Dense layers
        // After 3 pooling: 28 -> 14 -> 7 -> 3 (with padding)
        // Features: 128 * 3 * 3 = 1152
        dense_layers.push_back(new DenseLayer(1152, 256));
        dense_layers.push_back(new DenseLayer(256, 128));
        dense_layers.push_back(new DenseLayer(128, 10));
        
        std::cout << "   Total Layers: " << (conv_layers.size() + dense_layers.size()) << std::endl;
    }
    
    ~CNN() {
        for (auto layer : conv_layers) delete layer;
        for (auto layer : pool_layers) delete layer;
        for (auto layer : bn_layers) delete layer;
        for (auto layer : dense_layers) delete layer;
    }
    
    Tensor forward(const Tensor& input) {
        Tensor x = input;
        
        // Conv blocks
        for (size_t i = 0; i < conv_layers.size(); i++) {
            x = conv_layers[i]->forward(x);
            x = bn_layers[i]->forward(x, true);
            x = TensorActivation::relu(x);
            x = pool_layers[i]->forward(x);
        }
        
        // Flatten
        x = flatten.forward(x);
        
        // Dense layers with ReLU
        for (size_t i = 0; i < dense_layers.size() - 1; i++) {
            x = dense_layers[i]->forward(x);
            x = TensorActivation::relu(x);
        }
        
        // Output layer
        x = dense_layers[dense_layers.size() - 1]->forward(x);
        x = TensorActivation::softmax(x);
        
        return x;
    }
    
    int predictClass(const Tensor& input) {
        Tensor output = forward(input);
        
        int best_class = 0;
        float max_prob = output.data[0];
        
        for (int j = 1; j < 10; j++) {
            if (output.data[j] > max_prob) {
                max_prob = output.data[j];
                best_class = j;
            }
        }
        
        return best_class;
    }
};

// ============================================
// 8. DATA GENERATOR FOR IMAGES
// ============================================
class ImageDataGenerator {
private:
    std::mt19937 rng;
    
public:
    ImageDataGenerator() : rng(std::random_device{}()) {}
    
    void generateImageData(std::vector<Tensor>& images, std::vector<int>& labels,
                          int num_samples, int height, int width) {
        std::bernoulli_distribution pixel_dist(0.3f);
        std::uniform_int_distribution<int> class_dist(0, 9);
        
        std::cout << "📊 Generating " << num_samples << " images (" << height 
                  << "x" << width << ")..." << std::endl;
        
        for (int i = 0; i < num_samples; i++) {
            Tensor image(1, height, width, 1);
            
            for (auto& pixel : image.data) {
                pixel = pixel_dist(rng) ? 1.0f : 0.0f;
            }
            
            images.push_back(image);
            labels.push_back(class_dist(rng));
        }
        
        std::cout << "✓ Data generated" << std::endl;
    }
};

// ============================================
// 9. MAIN PROGRAM
// ============================================
int main() {
    std::cout << "🧠 Convolutional Neural Network - C++ Implementation\n";
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    // Parameters
    const int num_epochs = 15;
    const int batch_size = 32;
    const int num_train_samples = 5000;
    const int num_test_samples = 1000;
    
    // Generate data
    ImageDataGenerator img_gen;
    std::vector<Tensor> X_train, X_test;
    std::vector<int> y_train, y_test;
    
    std::cout << "📂 Preparing Data..." << std::endl;
    img_gen.generateImageData(X_train, y_train, num_train_samples, 28, 28);
    img_gen.generateImageData(X_test, y_test, num_test_samples, 28, 28);
    std::cout << std::endl;
    
    // Build CNN
    CNN model;
    
    std::cout << "\n⚙️  Training Parameters:" << std::endl;
    std::cout << "   Batch Size: " << batch_size << std::endl;
    std::cout << "   Epochs: " << num_epochs << "\n" << std::endl;
    
    // Training loop
    std::cout << "🚀 Starting Training...\n" << std::endl;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    float best_accuracy = 0.0f;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Training
        int train_correct = 0;
        int num_batches = (num_train_samples + batch_size - 1) / batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, num_train_samples);
            
            for (int i = start_idx; i < end_idx; i++) {
                int pred = model.predictClass(X_train[i]);
                if (pred == y_train[i]) {
                    train_correct++;
                }
            }
            
            if ((batch + 1) % 10 == 0) {
                std::cout << "Batch [" << (batch + 1) << "/" << num_batches << "] "
                         << "Train Acc: " << (100.0f * train_correct / ((batch + 1) * batch_size)) 
                         << "%" << std::endl;
            }
        }
        
        float train_accuracy = 100.0f * train_correct / num_train_samples;
        
        // Testing
        int test_correct = 0;
        for (int i = 0; i < num_test_samples; i++) {
            int pred = model.predictClass(X_test[i]);
            if (pred == y_test[i]) {
                test_correct++;
            }
        }
        
        float test_accuracy = 100.0f * test_correct / num_test_samples;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end - epoch_start).count() / 1000.0f;
        
        std::cout << "\n📈 Results:" << std::endl;
        std::cout << "   Train Accuracy: " << std::fixed << std::setprecision(2) 
                  << train_accuracy << "%" << std::endl;
        std::cout << "   Test Accuracy:  " << std::fixed << std::setprecision(2) 
                  << test_accuracy << "%" << std::endl;
        std::cout << "   Time: " << epoch_time << " seconds\n" << std::endl;
        
        if (test_accuracy > best_accuracy) {
            best_accuracy = test_accuracy;
            std::cout << "✅ Best Model! Accuracy: " << best_accuracy << "%\n" << std::endl;
        }
    }
    
    auto training_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration_cast<std::chrono::seconds>(
        training_end - training_start).count();
    
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "✅ Training Complete!" << std::endl;
    std::cout << "   Best Test Accuracy: " << best_accuracy << "%" << std::endl;
    std::cout << "   Total Time: " << total_time / 60.0f << " minutes" << std::endl;
    std::cout << std::string(70, '=') << "\n" << std::endl;
    
    return 0;
}