
//C++ wrapper
// some code below is just for explanation and visualization of the mapping process, not actual implementation
#include <iostream>
#include <vector>
#include <string>

class SNNNeuromorphicMapping {
public:
    static void explainMapping() {
        std::cout << "\n🔄 SNN to Neuromorphic Hardware Mapping\n";
        std::cout << std::string(70, '=') << "\n" << std::endl;
        
        std::cout << "┌─ Traditional ANN (GPU/CPU)" << std::endl;
        std::cout << "│  • Real-valued activations: 0.0 to 1.0" << std::endl;
        std::cout << "│  • Synchronous computation (every timestep)" << std::endl;
        std::cout << "│  • Dense matrix operations" << std::endl;
        std::cout << "│  • High power consumption" << std::endl;
        std::cout << "│  • High latency for inference" << std::endl;
        std::cout << "│\n" << std::endl;
        
        std::cout << "└─ Spiking Neural Network (Neuromorphic)" << std::endl;
        std::cout << "   • Binary spikes: 0 or 1" << std::endl;
        std::cout << "   • Event-driven computation (only when spikes occur)" << std::endl;
        std::cout << "   • Sparse operations (only active neurons)" << std::endl;
        std::cout << "   • Ultra-low power consumption (mW)" << std::endl;
        std::cout << "   • Ultra-low latency (microseconds)\n" << std::endl;
        
        std::cout << std::string(70, '=') << "\n" << std::endl;
        
        // Mapping details
        printMappingDetails();
    }
    
private:
    static void printMappingDetails() {
        std::cout << "📊 Detailed Mapping:\n" << std::endl;
        
        std::cout << "1️⃣  ARTIFICIAL NEURON (ANN)\n";
        std::cout << "   ┌─────────────────────────┐" << std::endl;
        std::cout << "   │  y = σ(∑ w*x + b)       │" << std::endl;
        std::cout << "   │  Output: continuous     │" << std::endl;
        std::cout << "   │  e.g., y = 0.735        │" << std::endl;
        std::cout << "   └─────────────────────────┘" << std::endl;
        std::cout << "   ✓ Works on: GPUs, CPUs, TPUs" << std::endl;
        std::cout << "   ✓ Power: 50-500W (inference)" << std::endl;
        std::cout << "   ✓ Latency: milliseconds\n" << std::endl;
        
        std::cout << "2️⃣  SPIKING NEURON (SNN)\n";
        std::cout << "   ┌─────────────────────────────┐" << std::endl;
        std::cout << "   │  V(t) = τV(t-1) + I(t)      │" << std::endl;
        std::cout << "   │  S(t) = 1 if V(t) > Vth     │" << std::endl;
        std::cout << "   │  Output: binary spike       │" << std::endl;
        std::cout << "   │  e.g., S(t) = 1 or 0        │" << std::endl;
        std::cout << "   └─────────────────────────────┘" << std::endl;
        std::cout << "   ✓ Works on: Neuromorphic chips" << std::endl;
        std::cout << "   ✓ Power: 1-100mW (inference)" << std::endl;
        std::cout << "   ✓ Latency: microseconds\n" << std::endl;
        
        std::cout << std::string(70, '=') << "\n" << std::endl;
    }
};


// the code below is totally different from the above, it is the actual wrapper for the CUDA kernel, which will be called from Python. It is just a simple example and not the actual implementation of the SNN forward pass.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declaration of the CUDA kernel
extern "C" void snn_forward_kernel(
    const float* input,
    float* v,
    float* spikes,
    float v_th,
    float tau_inv,
    int64_t B,
    int64_t N,
    int64_t T
);

torch::Tensor snn_forward_cuda(
    torch::Tensor input,   // [B, N, T]
    torch::Tensor v,       // [B, N]
    float v_th,
    float tau_inv
) {
    // input: [B, N, T]
    auto sizes = input.sizes();
    int64_t B = sizes[0];
    int64_t N = sizes[1];
    int64_t T = sizes[2];

    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(v.device().is_cuda(), "v must be on CUDA");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");

    // Ensure contiguous
    input = input.contiguous();
    v = v.contiguous();

    auto spikes = torch::zeros_like(input, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* v_ptr = v.mutable_data_ptr<float>();
    float* spikes_ptr = spikes.mutable_data_ptr<float>();

    int64_t total = B * N * T;
    int64_t threads_per_block = 256;
    int64_t num_blocks = (total + threads_per_block - 1) / threads_per_block;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    snn_forward_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_ptr, v_ptr, spikes_ptr, v_th, tau_inv, B, N, T
    );

    TORCH_CHECK_CUDA(cudaGetLastError());

    return spikes;
}