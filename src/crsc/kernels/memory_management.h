#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Allocate device memory; aborts on failure
void* snn_malloc_device(size_t bytes);

// Free device memory allocated via snn_malloc_device
void snn_free_device(void* ptr);

// Allocate page-locked host memory for fast async H2D/D2H transfers
void* snn_malloc_pinned(size_t bytes);

// Free pinned memory allocated via snn_malloc_pinned
void snn_free_pinned(void* ptr);

// Query current device free/total memory in bytes
void snn_query_memory(size_t* free_bytes, size_t* total_bytes);
