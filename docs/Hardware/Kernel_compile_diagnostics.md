# Kernel Mode Build Diagnostic Report

## Problem Summary
Kernel mode is set to `ON` in `configuration/SNN_module.yaml`, but **the CUDA kernel extension (`snn_cuda`) is not built**, so training/inference silently falls back to the Python framework.

## Current Configuration
- **Kernel Mode**: ON (in `SNN_module.yaml` line 13)
- **Framework**: Norse (default)
- **Expected Kernel Module**: `snn_cuda.snn_forward` (built from `src/crsc/kernels/snn_forward.cpp` + `snn_forward.cu`)

## What Should Happen (Kernel Mode Flow)
```
Training/Inference
    ↓
SNNTrainer.forward_pass() checks: if use_custom_kernel?
    ↓ YES → kernel.forward(data, voltage_buf, threshold, tau_inv) 
    ↓ NO  → self.model(data)  [fallback to Python framework]
```

## Current Issue
The `try-except` block in `src/learning/training.py` (lines 73-80) silently catches the import failure:

```python
if cfg.KERNEL == "ON":
    try:
        import snn_cuda.snn_forward as km
        self.use_custom_kernel = True
        print("[kernel] SNNTrainer: custom CRSC CUDA kernel active")
    except ImportError:
        print("[kernel] snn_cuda not built — run: python src/learning/setup.py build_ext --inplace")
```

**The module fails to import because it was never built.**

## Build Requirements & Issues

### Issue 1: Missing Microsoft Visual C++ Build Tools
**Error**: `Microsoft Visual C++ 14.0 or greater is required`

**Solution**: Install from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Download and run the installer
- Select "Desktop development with C++" workload
- Ensure MSVC v142 or higher is selected

### Issue 2: CUDA Version Mismatch
**Error**: PyTorch compiled with CUDA 12.8, but system has 12.8
- This shouldn't cause build failure, but may cause runtime issues
- **Recommendation**: Verify CUDA toolkit version matches PyTorch

### Issue 3: Missing Ninja Backend
**Warning**: Attempting to use ninja, but it's not found; falling back to distutils

**Solution (Optional but Recommended)**:
```bash
pip install ninja
```
- Ninja is much faster than distutils for C++ builds
- Not required, but significantly speeds up compilation

## Build Steps

### 1. Install Prerequisites
```bash
# Install C++ Build Tools (Windows)
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Optionally install Ninja for faster builds
pip install ninja

# Verify CUDA Toolkit
nvcc --version  # Should match PyTorch's CUDA version (12.8)
```

### 2. Build the CUDA Extension
```bash
cd C:\Users\zuhai\Desktop\Projects\SNN\SNNs-auf-GPUs.worktrees\agents-kernel-application-debugging

python src/learning/setup.py build_ext --inplace
```

**Expected Output on Success**:
```
running build_ext
building 'snn_cuda.snn_forward' extension
...
copying snn_cuda.cp312-win_amd64.pyd -> snn_cuda/
```

### 3. Verify the Build
```bash
python -c "import snn_cuda.snn_forward; print('✓ Kernel module loaded successfully')"
```

## Post-Build Verification

After building, run a test training session:

```bash
python src/learning/main.py
```

**Look for this in the output**:
```
[kernel] SNNTrainer: custom CRSC CUDA kernel active
```

If you see this, **kernel mode is now active**!

## Debugging Checklist

- [ ] Microsoft Visual C++ 14.0+ installed
- [ ] CUDA 12.x toolkit installed and in PATH
- [ ] `python src/learning/setup.py build_ext --inplace` succeeds
- [ ] `snn_cuda.pyd` file exists in `src/learning/` or site-packages
- [ ] Import test: `python -c "import snn_cuda.snn_forward"`
- [ ] Main script prints `[kernel] custom CRSC CUDA kernel active`

## If Build Still Fails

Run with verbose output:
```bash
python src/learning/setup.py build_ext --inplace -v
```

Check for:
1. **Compiler errors** in C++/CUDA code (files listed in setup.py)
2. **Missing include files** from `acceleration/GPU_attributes/`
3. **CUDA library linking issues**

## Temporary Workaround (if can't build)

If you cannot build the CUDA extension right now, **disable kernel mode**:

Edit `configuration/SNN_module.yaml`:
```yaml
training:
  kernel: OFF  # Temporarily disable to use Python frameworks
```

This will use the Norse/SNNTorch frameworks instead—slower, but functional.

## Files to Monitor

- `configuration/SNN_module.yaml` — kernel setting (line 13)
- `src/learning/training.py` — trainer kernel dispatch logic (lines 73-80, 101-140)
- `src/learning/inference.py` — inference kernel dispatch logic (lines 26-33, 38-62)
- `src/learning/setup.py` — build configuration
- `src/crsc/kernels/*.cpp` / `*.cu` — kernel implementation

## Next Steps

1. Install Visual C++ Build Tools
2. Run: `python src/learning/setup.py build_ext --inplace`
3. Verify: `python -c "import snn_cuda.snn_forward"`
4. Run training: `python src/learning/main.py`
5. Check output for `[kernel] custom CRSC CUDA kernel active`
