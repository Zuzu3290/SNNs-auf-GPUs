#include <iostream>
#include <vector>
#include <map>
#include <string>

// ============================================
// Neuromorphic Hardware Comparison
// ============================================
struct NeuromorphicHardware {
    std::string name;
    std::string architecture;
    std::string neuron_model;
    float power_consumption_mw;
    float throughput_fps;
    std::string programming_model;
    bool supports_snn;
    std::string description;
};

class NeuromorphicAnalysis {
public:
    static void printComparison() {
        std::vector<NeuromorphicHardware> devices = {
            {
                "Intel Loihi 2",
                "Asynchronous Many-Core",
                "Compartmental LIF",
                50.0f,
                1000000.0f,  // 1M neurons at real-time
                "Lava Framework",
                true,
                "State-of-the-art neuromorphic chip with learning rules"
            },
            {
                "IBM TrueNorth",
                "Event-Driven Many-Core",
                "Leaky Integrate-and-Fire (LIF)",
                70.0f,
                4096.0f,  // 4096 neurons
                "COREXY",
                true,
                "Pioneer in neuromorphic computing, 5.4B synapses"
            },
            {
                "SpiNNaker",
                "Spiking Neural Network Architecture",
                "Hodgkin-Huxley & custom",
                2000.0f,  // For full system
                1000000.0f,
                "PyNN, Brian2",
                true,
                "Distributed spiking neural network simulation"
            },
            {
                "BrainScaleS",
                "Analog Neuromorphic",
                "Conductance-based",
                5000.0f,  // Full system
                10000.0f,  // Real-time speedup
                "PyNN",
                true,
                "Ultra-fast analog neural network emulation"
            },
            {
                "Akida (BrainChip)",
                "Event-Based Neural Processor",
                "Spiking LIF variant",
                1.0f,  // Ultra-low power
                10000.0f,
                "TensorFlow/Keras compatible",
                true,
                "Edge AI with spiking neurons"
            },
            {
                "ARM Cortex-M (Standard)",
                "Von Neumann CPU",
                "ANN (not spike-based)",
                50.0f,  // Approximate
                1000.0f,
                "C/C++, ARM Assembly",
                false,
                "Traditional compute, not optimized for SNNs"
            }
        };
        
        printTable(devices);
    }
    
    static void printTable(const std::vector<NeuromorphicHardware>& devices) {
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "🧠 NEUROMORPHIC HARDWARE COMPARISON" << std::endl;
        std::cout << std::string(100, '=') << "\n" << std::endl;
        
        for (const auto& hw : devices) {
            std::cout << "📱 " << hw.name << std::endl;
            std::cout << "   Architecture: " << hw.architecture << std::endl;
            std::cout << "   Neuron Model: " << hw.neuron_model << std::endl;
            std::cout << "   Power: " << hw.power_consumption_mw << " mW" << std::endl;
            std::cout << "   Throughput: " << hw.throughput_fps << " ops/s" << std::endl;
            std::cout << "   Programming: " << hw.programming_model << std::endl;
            std::cout << "   SNN Compatible: " << (hw.supports_snn ? "✅ YES" : "❌ NO") << std::endl;
            std::cout << "   📝 " << hw.description << "\n" << std::endl;
        }
        
        std::cout << std::string(100, '=') << std::endl;
    }
};