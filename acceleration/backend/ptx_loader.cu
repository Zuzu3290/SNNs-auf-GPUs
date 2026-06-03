// Full CUDA Driver API implementation for runtime PTX kernel loading, function lookup, launch, and cleanup.
// cuInit(0) is called once per process via a static guard; PTX source is JIT-compiled to cubin by the driver on load.
// Supports loading from a file path or an in-memory string, enabling both file-based and embedded kernel workflows.
#include "ptx_loader.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void check_cu(CUresult r, const char* op) {
    if (r == CUDA_SUCCESS) return;
    const char* msg = nullptr;
    cuGetErrorString(r, &msg);
    fprintf(stderr, "[PTXLoader] %s failed: %s\n", op, msg ? msg : "unknown error");
    abort();
}

// cuInit must be called once per process before any Driver API use.
static void ensure_driver_init() {
    static bool initialised = false;
    if (initialised) return;
    check_cu(cuInit(0), "cuInit");
    initialised = true;
}

// Read an entire file into a heap-allocated null-terminated buffer.
// Caller must free() the result.
static char* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[PTXLoader] Cannot open file: %s\n", path);
        abort();
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    char* buf = static_cast<char*>(malloc(size + 1));
    if (!buf) {
        fprintf(stderr, "[PTXLoader] malloc failed for %ld bytes\n", size + 1);
        abort();
    }
    size_t n = fread(buf, 1, static_cast<size_t>(size), f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

CUmodule ptx_load_from_file(const char* ptx_path) {
    ensure_driver_init();
    char* src = read_file(ptx_path);
    CUmodule mod;
    check_cu(cuModuleLoadData(&mod, src), "cuModuleLoadData(file)");
    free(src);
    printf("[PTXLoader] Loaded: %s\n", ptx_path);
    return mod;
}

CUmodule ptx_load_from_string(const char* ptx_source) {
    ensure_driver_init();
    CUmodule mod;
    check_cu(cuModuleLoadData(&mod, ptx_source), "cuModuleLoadData(string)");
    return mod;
}

CUfunction ptx_get_kernel(CUmodule module, const char* kernel_name) {
    CUfunction fn;
    check_cu(cuModuleGetFunction(&fn, module, kernel_name), "cuModuleGetFunction");
    return fn;
}

void ptx_launch_1d(CUfunction fn,
                   int n_elements, int block_size,
                   size_t shared_mem_bytes,
                   CUstream stream,
                   void** kernel_args) {
    const int grid = (n_elements + block_size - 1) / block_size;
    check_cu(
        cuLaunchKernel(
            fn,
            static_cast<unsigned>(grid),  1, 1,
            static_cast<unsigned>(block_size), 1, 1,
            static_cast<unsigned>(shared_mem_bytes),
            stream,
            kernel_args,
            nullptr     // extra — not used
        ),
        "cuLaunchKernel"
    );
}

void ptx_unload(CUmodule module) {
    if (module) cuModuleUnload(module);
}
