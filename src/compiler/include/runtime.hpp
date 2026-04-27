// src/compiler/src/runtime.cpp

// This executes the plan.

// Its role is:

// call backend APIs
// allocate and release memory
// launch kernels
// synchronize streams
// collect timing/profiling info

// This is the execution engine from the compiler side.