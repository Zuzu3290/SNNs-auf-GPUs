 this is the precise architectural reason why C++/CUDA was the right choice, not just a performance preference.

The specific capabilities that make our system what it is sit below where XLA operates. XLA is a compiler that transforms HLO operations into GPU code. It cannot express nor expose what lives at the CUDA driver and runtime level. Here is exactly what is exclusive to our implementation:

__ballot_sync — warp-level hardware instruction. XLA has no warp primitive. Triton has a partial equivalent (tl.vote) but does not expose the raw 32-bit mask. The SpikeRateBus receiving a free popcount from the kernel ballot is not achievable through any compiler abstraction. It requires CUDA PTX.

cudaDeviceGetDefaultMemPool + cudaMemPoolSetAttribute — these are CUDA driver API calls that operate on the allocator itself. XLA manages its own memory pool internally and exposes no hook to set a release threshold or read cudaMemPoolAttrUsedMemCurrent. The entire MemoryArbiter only exists because we are at the driver level. From XLA you cannot touch the pool.

Runtime adaptive grid — dispatch_warp_oriented samples elapsed_ms every 50 calls and adjusts target_blocks_per_sm for the next launch at runtime. Triton's autotuner selects block sizes at compile time, not per-call. XLA has no mechanism for a kernel to adjust its own grid based on a measured runtime signal.

CUDA events for per-call timing — cudaEventCreate, cudaEventRecord, cudaEventElapsedTime are CUDA runtime calls used in the energy profiler and the timed kernel variant. XLA does not expose these inside a compiled computation.

std::atomic<size_t> zone accounting with CAS rollback — the lock-free zone tracking in MemoryArbiter is C++ host code. Technically a compiler does not prevent this, but without the cuMemPool API it is only accounting — there is nothing to enforce. The enforcement mechanism is cudaMemPoolSetAttribute, which is driver-level only.

If we had built on XLA/JAX custom calls — where you write a C++ function and register it as an XLA operation — we would still be writing CUDA. We would have added JAX's overhead on top of our CUDA without gaining the compiler benefits, because JAX's fusion and optimisation only apply inside its traceable operations, not to opaque custom calls.

The conclusion: C++/CUDA was not a performance choice over XLA. It was an access choice. These hardware capabilities do not exist at the abstraction level XLA operates at. They only exist at the driver level. Writing in C++/CUDA was the only way to build what we built.

