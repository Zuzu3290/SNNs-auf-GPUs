Compiler front-end

This layer takes the model computation and translates it into a form the compiler can understand.

This layer is responsible for:

inspecting PyTorch operations
capturing model execution regions
recognizing SNN-specific semantics
creating compiler IR nodes

This is where “framework code” becomes “compiler code.”

Compiler middle-end

This is the core of the system.

It is responsible for:

representing operations in IR
rewriting operations
annotating device placement
deciding scheduling
planning execution order
preparing a runnable execution plan

This is the actual compiler brain.

Backend

This layer knows how to execute the plan on real hardware.

It maps compiler operations to:

CPU routines
CUDA kernels
later possibly PTX modules or fused kernels

This is where abstract ops become hardware work

Runtime

This layer manages actual execution.

It handles:

memory allocation
transfers
streams
kernel launch configuration
synchronization
profiling hooks
execution feedback to the scheduler