// src/compiler/src/scheduler.cpp

// This decides how work is placed and ordered.

// Its role is:

// assign device
// decide offload
// determine execution order
// apply scheduling heuristics
// optionally fuse trivial sequences

// This file answers:
// “Should this run on the GPU now, or not?”