



class HardwareScheduler {
    Kernel select_kernel(HardwareConfig config) {
        if (config.energy_target == "low") 
            return fused_low_energy_kernel;
        return standard_kernel;
    }
};