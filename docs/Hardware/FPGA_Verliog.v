// ============================================
// LIF (Leaky Integrate-and-Fire) Neuron Core
// FPGA Implementation in Verilog
// ============================================

`timescale 1ns / 1ps

module lif_neuron (
    input clk,
    input rst_n,
    input [31:0] input_current,      // Input synaptic current (32-bit float)
    input [31:0] threshold,           // Spike threshold (32-bit float)
    input [31:0] tau,                 // Time constant (leak rate)
    input [31:0] reset_voltage,       // Post-spike reset value
    output reg spike_out,             // Spike output (binary)
    output reg [31:0] voltage_out     // Membrane voltage output
);

    // Internal registers
    reg [31:0] voltage;               // Membrane potential
    reg [31:0] decayed_voltage;       // After decay
    reg [31:0] next_voltage;          // After current injection
    
    // Floating point calculation wires
    wire [31:0] decay_result;
    wire [31:0] add_result;
    wire voltage_exceeded_threshold;
    
    // Floating point multiplier for decay: V * tau
    fpmult_decay decay_unit (
        .clk(clk),
        .A(voltage),
        .B(tau),
        .P(decay_result)
    );
    
    // Floating point adder: decayed_V + I
    fpadd add_unit (
        .clk(clk),
        .A(decayed_voltage),
        .B(input_current),
        .S(add_result)
    );
    
    // Comparator: voltage > threshold
    assign voltage_exceeded_threshold = (voltage[30:0] > threshold[30:0]) && 
                                       (voltage[31] == threshold[31]);
    
    // Main neuron dynamics
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            voltage <= 32'h00000000;      // Reset to 0V
            spike_out <= 1'b0;
            voltage_out <= 32'h00000000;
        end else begin
            // Stage 1: Decay voltage (multiply by tau)
            decayed_voltage <= decay_result;
            
            // Stage 2: Add input current
            next_voltage <= add_result;
            
            // Stage 3: Check threshold and generate spike
            if (voltage_exceeded_threshold) begin
                spike_out <= 1'b1;
                voltage <= reset_voltage;          // Reset to post-spike value
            end else begin
                spike_out <= 1'b0;
                voltage <= next_voltage;           // Update with new voltage
            end
            
            // Output voltage for monitoring
            voltage_out <= voltage;
        end
    end
    
endmodule

// ============================================
// Floating Point Multiplier (32-bit)
// ============================================
module fpmult_decay (
    input clk,
    input [31:0] A,
    input [31:0] B,
    output reg [31:0] P
);
    
    // Simple single-precision float multiplication
    // Sign, exponent, mantissa handling
    
    wire sign = A[31] ^ B[31];
    wire [7:0] exp_sum = A[30:23] + B[30:23] - 8'd127;
    
    // Mantissa multiplication (simplified)
    wire [47:0] mant_prod = (A[22:0] | 23'h800000) * (B[22:0] | 23'h800000);
    wire [22:0] mant_out = mant_prod[47] ? mant_prod[46:24] : mant_prod[45:23];
    
    always @(posedge clk) begin
        P <= {sign, exp_sum, mant_out};
    end
    
endmodule

// ============================================
// Floating Point Adder (32-bit)
// ============================================
module fpadd (
    input clk,
    input [31:0] A,
    input [31:0] B,
    output reg [31:0] S
);
    
    // Simplified floating point addition
    // Align exponents and add mantissas
    
    wire [7:0] exp_a = A[30:23];
    wire [7:0] exp_b = B[30:23];
    wire [22:0] mant_a = A[22:0] | 23'h800000;
    wire [22:0] mant_b = B[22:0] | 23'h800000;
    
    wire [7:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [47:0] shifted_mant = (exp_a > exp_b) ? 
                               (mant_a << exp_diff) : 
                               (mant_b << exp_diff);
    
    wire [47:0] sum = (exp_a > exp_b) ? 
                      (shifted_mant + mant_b) : 
                      (shifted_mant + mant_a);
    
    wire [7:0] result_exp = (exp_a > exp_b) ? exp_a : exp_b;
    wire [22:0] result_mant = sum[46:24];
    
    always @(posedge clk) begin
        S <= {A[31], result_exp, result_mant};
    end
    
endmodule

// ============================================
// NEURON LAYER (Multiple LIF Neurons)
// ============================================
module neuron_layer #(
    parameter NUM_NEURONS = 256,
    parameter ADDR_WIDTH = 8
)(
    input clk,
    input rst_n,
    input [31:0] input_currents [NUM_NEURONS-1:0],
    input [31:0] threshold,
    input [31:0] tau,
    input [31:0] reset_voltage,
    output [NUM_NEURONS-1:0] spikes,
    output [31:0] voltages [NUM_NEURONS-1:0]
);
    
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_instances
            lif_neuron neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .input_current(input_currents[i]),
                .threshold(threshold),
                .tau(tau),
                .reset_voltage(reset_voltage),
                .spike_out(spikes[i]),
                .voltage_out(voltages[i])
            );
        end
    endgenerate
    
endmodule

// ============================================
// SYNAPTIC WEIGHT MODULE
// ============================================
module synapse_layer #(
    parameter INPUT_NEURONS = 784,
    parameter OUTPUT_NEURONS = 256,
    parameter WEIGHT_WIDTH = 32
)(
    input clk,
    input rst_n,
    input [INPUT_NEURONS-1:0] input_spikes,
    input [WEIGHT_WIDTH-1:0] weights [INPUT_NEURONS-1:0][OUTPUT_NEURONS-1:0],
    output reg [WEIGHT_WIDTH-1:0] output_currents [OUTPUT_NEURONS-1:0]
);
    
    integer i, j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < OUTPUT_NEURONS; i = i + 1) begin
                output_currents[i] <= 32'h00000000;
            end
        end else begin
            // For each output neuron
            for (i = 0; i < OUTPUT_NEURONS; i = i + 1) begin
                output_currents[i] <= 32'h00000000;  // Reset
                
                // Sum weighted inputs (only from spiking inputs)
                for (j = 0; j < INPUT_NEURONS; j = j + 1) begin
                    if (input_spikes[j]) begin
                        // Add weight to current (spike × weight)
                        output_currents[i] <= output_currents[i] + weights[j][i];
                    end
                end
            end
        end
    end
    
endmodule

// ============================================
// COMPLETE SNN CORE
// ============================================
module snn_core #(
    parameter INPUT_SIZE = 784,
    parameter HIDDEN_SIZE = 256,
    parameter OUTPUT_SIZE = 10
)(
    input clk,
    input rst_n,
    input [INPUT_SIZE-1:0] input_spikes,
    input [31:0] threshold,
    input [31:0] tau,
    input [31:0] reset_voltage,
    input [31:0] weights_in_hidden [INPUT_SIZE-1:0][HIDDEN_SIZE-1:0],
    input [31:0] weights_hidden_out [HIDDEN_SIZE-1:0][OUTPUT_SIZE-1:0],
    output [HIDDEN_SIZE-1:0] hidden_spikes,
    output [OUTPUT_SIZE-1:0] output_spikes,
    output [31:0] hidden_voltages [HIDDEN_SIZE-1:0],
    output [31:0] output_voltages [OUTPUT_SIZE-1:0]
);
    
    // Internal signals
    wire [31:0] hidden_currents [HIDDEN_SIZE-1:0];
    wire [31:0] output_currents [OUTPUT_SIZE-1:0];
    
    // Stage 1: Input → Hidden (Synapses)
    synapse_layer #(
        .INPUT_NEURONS(INPUT_SIZE),
        .OUTPUT_NEURONS(HIDDEN_SIZE),
        .WEIGHT_WIDTH(32)
    ) input_synapse (
        .clk(clk),
        .rst_n(rst_n),
        .input_spikes(input_spikes),
        .weights(weights_in_hidden),
        .output_currents(hidden_currents)
    );
    
    // Stage 2: Hidden Layer (LIF Neurons)
    neuron_layer #(
        .NUM_NEURONS(HIDDEN_SIZE),
        .ADDR_WIDTH(8)
    ) hidden_layer (
        .clk(clk),
        .rst_n(rst_n),
        .input_currents(hidden_currents),
        .threshold(threshold),
        .tau(tau),
        .reset_voltage(reset_voltage),
        .spikes(hidden_spikes),
        .voltages(hidden_voltages)
    );
    
    // Stage 3: Hidden → Output (Synapses)
    synapse_layer #(
        .INPUT_NEURONS(HIDDEN_SIZE),
        .OUTPUT_NEURONS(OUTPUT_SIZE),
        .WEIGHT_WIDTH(32)
    ) output_synapse (
        .clk(clk),
        .rst_n(rst_n),
        .input_spikes(hidden_spikes),
        .weights(weights_hidden_out),
        .output_currents(output_currents)
    );
    
    // Stage 4: Output Layer (LIF Neurons)
    neuron_layer #(
        .NUM_NEURONS(OUTPUT_SIZE),
        .ADDR_WIDTH(4)
    ) output_layer (
        .clk(clk),
        .rst_n(rst_n),
        .input_currents(output_currents),
        .threshold(threshold),
        .tau(tau),
        .reset_voltage(reset_voltage),
        .spikes(output_spikes),
        .voltages(output_voltages)
    );
    
endmodule

// ============================================
// TESTBENCH
// ============================================
module tb_snn_core ();
    
    reg clk, rst_n;
    reg [783:0] input_spikes;
    reg [31:0] threshold, tau, reset_voltage;
    wire [9:0] output_spikes;
    
    // Parameters
    reg [31:0] weights_in [783:0][255:0];
    reg [31:0] weights_out [255:0][9:0];
    
    // Instantiate SNN core
    snn_core #(
        .INPUT_SIZE(784),
        .HIDDEN_SIZE(256),
        .OUTPUT_SIZE(10)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_spikes(input_spikes),
        .threshold(threshold),
        .tau(tau),
        .reset_voltage(reset_voltage),
        .weights_in_hidden(weights_in),
        .weights_hidden_out(weights_out),
        .output_spikes(output_spikes)
    );
    
    // Clock generation
    always begin
        #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        clk = 0;
        rst_n = 0;
        
        // Initialize parameters
        threshold = 32'h3F800000;      // 1.0 in IEEE 754
        tau = 32'h3F000000;            // 0.75
        reset_voltage = 32'h00000000;  // 0.0
        
        // Wait for reset
        #100 rst_n = 1;
        
        // Apply test input
        input_spikes = 784'h1;  // Single neuron fires
        
        // Run for 1000 cycles
        repeat (1000) @(posedge clk);
        
        $finish;
    end
    
    // Monitoring
    initial begin
        $monitor("Time: %t, Output Spikes: %b", $time, output_spikes);
    end
    
endmodule