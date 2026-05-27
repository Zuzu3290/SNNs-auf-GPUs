# CRSC

The `crsc/` folder contains the core bridge between the custom kernel setup and the acceleration layer of the project.

It provides the system-level connection required to expose custom GPU kernels, bind them to higher-level code, and support accelerator-driven execution.

---

## Purpose

The purpose of this module is to provide an internal bridge between kernel implementation and the broader acceleration workflow.

It helps connect low-level CUDA kernel logic with higher-level application code, making custom kernel execution easier to integrate, test, and extend.

---

## Main Files

The `crsc/` folder may include files such as:

- `engine.cu`
- `binding.cpp`
- Other custom kernel application files

These files support the setup, binding, and execution of custom kernel functionality.

---

## Role in the Project

The `crsc/` module provides a larger scope for custom application implementation by connecting:

- Kernel setup
- CUDA-based execution logic
- Binding interfaces
- Accelerator data flow
- Custom kernel applications
- Higher-level software integration

This allows the project to use optimized kernel operations as part of the wider acceleration and runtime workflow.

---

## How It Fits

```mermaid
flowchart TD
    A[Custom Kernel Logic] --> B[CRSC Layer]
    B --> C[Binding Interface]
    C --> D[Acceleration Layer]
    D --> E[Higher-Level Application]