# Compiler Architecture

The compiler is structured into four main layers: the front-end, middle-end, backend, and runtime. Each layer has a specific responsibility in transforming model computation into executable hardware operations.

---

## Compiler Front-End

The compiler front-end is responsible for taking model computation and translating it into a form that the compiler can understand.

It acts as the bridge between the deep learning framework and the compiler system.

### Responsibilities

- Inspect PyTorch operations
- Capture model execution regions
- Recognize SNN-specific semantics
- Create compiler IR nodes

This is the stage where framework-level code becomes compiler-level code.

---

## Compiler Middle-End

The compiler middle-end is the core decision-making layer of the system.

It represents, rewrites, schedules, and prepares operations before they are passed to the backend for execution.

### Responsibilities

- Represent operations in IR
- Rewrite operations for optimization
- Annotate device placement
- Decide scheduling strategy
- Plan execution order
- Prepare a runnable execution plan

This layer acts as the compiler brain.

---

## Backend

The backend is responsible for mapping the compiler execution plan onto real hardware.

It takes abstract compiler operations and connects them to the appropriate hardware-level implementation.

### It maps operations to:

- CPU routines
- CUDA kernels
- Future PTX modules
- Future fused kernels

This is the stage where abstract operations become hardware work.

---

## Runtime

The runtime layer manages the actual execution of the compiled plan.

It is responsible for handling memory, execution control, synchronization, profiling, and runtime feedback.

### Responsibilities

- Memory allocation
- Data transfers
- Stream management
- Kernel launch configuration
- Synchronization
- Profiling hooks
- Execution feedback to the scheduler

The runtime ensures that the compiled operations are executed efficiently and correctly on the target hardware.

---

## Summary

Together, these four layers define the compiler workflow:

```mermaid
flowchart TD
    A[PyTorch Model Operations] --> B[Compiler Front-End]
    B --> C[Compiler Middle-End]
    C --> D[Backend]
    D --> E[Runtime]
    E --> F[Hardware Execution]