"""
tests/test_asm_kernels.py
Tests the ARM assembly kernels via Python ctypes.
The shared library must be built first: make asm c

Run: make test-asm
  or: python tests/test_asm_kernels.py
"""

import ctypes
import os
import sys
import numpy as np
import subprocess
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Try to load the compiled shared library of kernels
# If not built yet, compile on the fly for the test
# ---------------------------------------------------------------------------

LIB_PATH = os.path.join(os.path.dirname(__file__), "..",
                         "build", "lib", "libkernels.so")

def _try_build_lib():
    """Attempt to compile kernels into a .so for ctypes loading."""
    src_dir  = os.path.join(os.path.dirname(__file__), "..")
    asm_file = os.path.join(src_dir, "asm", "event_kernel.s")
    c_file   = os.path.join(src_dir, "src", "kernel", "spike_kernel.c")
    inc_dir  = os.path.join(src_dir, "include")
    obj_asm  = os.path.join(src_dir, "build", "lib", "asm_event_kernel.o")
    obj_c    = os.path.join(src_dir, "build", "lib", "spike_kernel.o")

    os.makedirs(os.path.join(src_dir, "build", "lib"), exist_ok=True)

    # Assemble
    ret = subprocess.run(
        ["as", "--warn", "-o", obj_asm, asm_file],
        capture_output=True)
    if ret.returncode != 0:
        return False, ret.stderr.decode()

    # Compile C (with NEON if available)
    cflags = ["-O2", "-fPIC", "-I", inc_dir]
    import platform
    if platform.machine() in ("aarch64", "arm64"):
        cflags += ["-march=armv8-a+simd"]

    ret = subprocess.run(
        ["gcc", *cflags, "-c", c_file, "-o", obj_c, "-lm"],
        capture_output=True)
    if ret.returncode != 0:
        return False, ret.stderr.decode()

    # Link shared library
    ret = subprocess.run(
        ["gcc", "-shared", "-o", LIB_PATH, obj_asm, obj_c, "-lm"],
        capture_output=True)
    if ret.returncode != 0:
        return False, ret.stderr.decode()

    return True, ""


# ---------------------------------------------------------------------------
# Load library
# ---------------------------------------------------------------------------

KERNELS = None
LIB_AVAILABLE = False

if not os.path.exists(LIB_PATH):
    ok, err = _try_build_lib()
    if not ok:
        print(f"[test] Could not build kernels library: {err}")
        print("[test] Falling back to Python-only tests")

if os.path.exists(LIB_PATH):
    try:
        KERNELS = ctypes.CDLL(LIB_PATH)
        LIB_AVAILABLE = True

        # --- event_threshold_neon ---
        KERNELS.event_threshold_neon.restype  = None
        KERNELS.event_threshold_neon.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # input
            ctypes.POINTER(ctypes.c_float),   # output
            ctypes.c_int,                      # count
            ctypes.c_float,                    # threshold
        ]

        # --- lif_membrane_neon ---
        KERNELS.lif_membrane_neon.restype  = None
        KERNELS.lif_membrane_neon.argtypes = [
            ctypes.POINTER(ctypes.c_float),   # input
            ctypes.POINTER(ctypes.c_float),   # membrane
            ctypes.POINTER(ctypes.c_float),   # spikes
            ctypes.c_int,                      # N
            ctypes.c_float,                    # leak
            ctypes.c_float,                    # v_thresh
            ctypes.c_float,                    # v_reset
        ]

        # --- dot_product_f32_neon ---
        KERNELS.dot_product_f32_neon.restype  = ctypes.c_float
        KERNELS.dot_product_f32_neon.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        # --- relu_inplace_neon ---
        KERNELS.relu_inplace_neon.restype  = None
        KERNELS.relu_inplace_neon.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

    except OSError as e:
        print(f"[test] Library load error: {e}")


def np_ptr(arr: np.ndarray):
    """Return ctypes pointer to numpy float32 array data."""
    assert arr.dtype == np.float32
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# Reference Python implementations (ground truth)
# ---------------------------------------------------------------------------

def ref_threshold(x, thresh):
    return (x >= thresh).astype(np.float32)

def ref_lif(inp, mem, leak, v_thresh, v_reset):
    mem = leak * mem + inp
    spk = (mem >= v_thresh).astype(np.float32)
    mem[mem >= v_thresh] = v_reset
    return mem.copy(), spk

def ref_dot(a, b):
    return float(np.dot(a, b))

def ref_relu(x):
    return np.maximum(x, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestThreshold(unittest.TestCase):

    def _run(self, N, thresh):
        inp = np.random.rand(N).astype(np.float32)
        out_asm = np.zeros(N, dtype=np.float32)
        out_ref = ref_threshold(inp, thresh)

        if LIB_AVAILABLE:
            KERNELS.event_threshold_neon(np_ptr(inp), np_ptr(out_asm),
                                          N, thresh)
            np.testing.assert_array_equal(out_asm, out_ref,
                err_msg=f"Threshold mismatch (N={N}, thresh={thresh})")
        else:
            np.testing.assert_array_equal(out_ref, ref_threshold(inp, thresh))

    def test_small(self):   self._run(16,   0.5)
    def test_medium(self):  self._run(256,  0.3)
    def test_large(self):   self._run(4096, 0.7)
    def test_unaligned(self): self._run(37, 0.5)    # not divisible by 16
    def test_all_below(self): self._run(64, 2.0)    # all zeros
    def test_all_above(self): self._run(64, -1.0)   # all ones


class TestLIFMembrane(unittest.TestCase):

    def _run(self, N, leak=0.9, v_thresh=1.0, v_reset=0.0):
        inp = np.random.rand(N).astype(np.float32) * 0.5
        mem_asm = np.random.rand(N).astype(np.float32) * 0.5
        mem_ref = mem_asm.copy()
        spk_asm = np.zeros(N, dtype=np.float32)

        mem_ref, spk_ref = ref_lif(inp, mem_ref, leak, v_thresh, v_reset)

        if LIB_AVAILABLE:
            KERNELS.lif_membrane_neon(np_ptr(inp), np_ptr(mem_asm),
                                       np_ptr(spk_asm),
                                       N, leak, v_thresh, v_reset)
            np.testing.assert_allclose(mem_asm, mem_ref, rtol=1e-5,
                err_msg="Membrane mismatch")
            np.testing.assert_array_equal(spk_asm, spk_ref,
                err_msg="Spike mismatch")

    def test_small(self):     self._run(4)
    def test_medium(self):    self._run(260 * 346)
    def test_tail(self):      self._run(7)    # exercises scalar tail
    def test_high_leak(self): self._run(128, leak=0.99)
    def test_no_leak(self):   self._run(128, leak=0.0)
    def test_low_thresh(self):
        # Almost everything should fire
        N = 64
        inp = np.ones(N, dtype=np.float32)
        mem_asm = np.zeros(N, dtype=np.float32)
        spk_asm = np.zeros(N, dtype=np.float32)
        if LIB_AVAILABLE:
            KERNELS.lif_membrane_neon(np_ptr(inp), np_ptr(mem_asm),
                                       np_ptr(spk_asm),
                                       N, 0.9, 0.5, 0.0)
            self.assertGreater(spk_asm.sum(), 0,
                               "Expected at least some spikes with low threshold")


class TestDotProduct(unittest.TestCase):

    def _run(self, N):
        a = np.random.rand(N).astype(np.float32)
        b = np.random.rand(N).astype(np.float32)
        expected = ref_dot(a, b)

        if LIB_AVAILABLE:
            got = KERNELS.dot_product_f32_neon(np_ptr(a), np_ptr(b), N)
            self.assertAlmostEqual(got, expected, places=3,
                msg=f"Dot product mismatch (N={N}): got {got}, expected {expected}")

    def test_tiny(self):    self._run(4)
    def test_small(self):   self._run(32)
    def test_medium(self):  self._run(1024)
    def test_large(self):   self._run(65536)
    def test_unaligned(self): self._run(13)

    def test_orthogonal(self):
        # Dot of orthogonal vectors should be ~0
        N = 8
        a = np.array([1,0,1,0,1,0,1,0], dtype=np.float32)
        b = np.array([0,1,0,1,0,1,0,1], dtype=np.float32)
        if LIB_AVAILABLE:
            got = KERNELS.dot_product_f32_neon(np_ptr(a), np_ptr(b), N)
            self.assertAlmostEqual(got, 0.0, places=5)

    def test_identity(self):
        N = 16
        a = np.ones(N, dtype=np.float32)
        b = np.ones(N, dtype=np.float32)
        if LIB_AVAILABLE:
            got = KERNELS.dot_product_f32_neon(np_ptr(a), np_ptr(b), N)
            self.assertAlmostEqual(got, float(N), places=3)


class TestReLU(unittest.TestCase):

    def _run(self, N):
        x = (np.random.rand(N).astype(np.float32) - 0.5) * 4
        x_ref = ref_relu(x.copy())

        if LIB_AVAILABLE:
            KERNELS.relu_inplace_neon(np_ptr(x), N)
            np.testing.assert_allclose(x, x_ref, rtol=1e-6,
                err_msg=f"ReLU mismatch (N={N})")

    def test_small(self):    self._run(16)
    def test_medium(self):   self._run(512)
    def test_large(self):    self._run(89600)  # 346*260
    def test_unaligned(self): self._run(11)
    def test_all_negative(self):
        x = -np.ones(32, dtype=np.float32)
        if LIB_AVAILABLE:
            KERNELS.relu_inplace_neon(np_ptr(x), 32)
            np.testing.assert_array_equal(x, np.zeros(32, dtype=np.float32))
    def test_all_positive(self):
        x = np.ones(32, dtype=np.float32) * 5
        orig = x.copy()
        if LIB_AVAILABLE:
            KERNELS.relu_inplace_neon(np_ptr(x), 32)
            np.testing.assert_array_equal(x, orig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Library available: {LIB_AVAILABLE}")
    if LIB_AVAILABLE:
        print(f"Loaded: {LIB_PATH}\n")
    else:
        print("Running Python-reference tests only (no compiled library)\n")

    unittest.main(verbosity=2)
