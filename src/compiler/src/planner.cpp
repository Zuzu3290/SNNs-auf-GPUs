// src/compiler/src/planner.cpp

// The scheduler makes decisions.
// The planner turns those decisions into a runnable plan.

// For example:

// allocate input buffer on GPU
// run fused membrane kernel for timesteps 0–15
// run decode on CPU
// return output tensor

// So the planner builds the execution recipe.