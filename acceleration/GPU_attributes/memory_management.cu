// Thin abort-on-failure wrappers around cudaMalloc, cudaMallocHost, and cudaMemGetInfo.
// snn_malloc_device / snn_malloc_pinned allocate device and page-locked host memory respectively; both abort on failure.
// Used by dispatch_accelerated in engine.cu to audit VRAM headroom before allocating the output spike tensor.
#include "memory_management.h"
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[MemoryMgr] %s: %s\n", op, cudaGetErrorString(err));
        abort();
    }
}

void* snn_malloc_device(size_t bytes) {
    void* ptr = nullptr;
    check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    return ptr;
}

void snn_free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void* snn_malloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    check(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
    return ptr;
}

void snn_free_pinned(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void snn_query_memory(size_t* free_bytes, size_t* total_bytes) {
    check(cudaMemGetInfo(free_bytes, total_bytes), "cudaMemGetInfo");
}
