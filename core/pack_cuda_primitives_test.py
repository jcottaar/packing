"""
pack_cuda_primitives_test.py

Test suite for primitive CUDA device functions in pack_cuda_primitives.py.
This module exposes each primitive function via test kernels and provides
Python interfaces for both forward and backward passes.

Pattern:
- Each primitive gets a dedicated kernel that exposes forward + backward
- Python wrapper functions handle CuPy arrays and device memory
- Lazy initialization compiles kernels on first use
- All functions compute gradients (no optional gradient computation)
"""

import numpy as np
import cupy as cp
import os
import subprocess
import shutil

import pack_cuda_primitives

# ---------------------------------------------------------------------------
# Module State - Lazy Initialization
# ---------------------------------------------------------------------------

_raw_module: cp.RawModule | None = None
_initialized: bool = False

# Kernel references (populated during initialization)
_line_intersection_kernel: cp.RawKernel | None = None


# ---------------------------------------------------------------------------
# CUDA Test Kernels
# ---------------------------------------------------------------------------

_TEST_KERNELS_SRC = r"""
extern "C" {

#define MAX_VERTS_PER_PIECE 4
#define MAX_INTERSECTION_VERTS 8

""" + pack_cuda_primitives.PRIMITIVE_SRC + r"""

// ============================================================================
// TEST KERNEL: line_intersection
// ============================================================================
__global__ void test_line_intersection_fwd_bwd(
    const double* p1_x, const double* p1_y,  // inputs: line 1 point 1 (batch)
    const double* p2_x, const double* p2_y,  // inputs: line 1 point 2 (batch)
    const double* q1_x, const double* q1_y,  // inputs: line 2 point 1 (batch)
    const double* q2_x, const double* q2_y,  // inputs: line 2 point 2 (batch)
    const int n,                              // batch size
    double* out_x, double* out_y,             // output: intersection points (batch)
    double* d_p1_x, double* d_p1_y,          // output: gradients w.r.t. p1
    double* d_p2_x, double* d_p2_y,          // output: gradients w.r.t. p2
    double* d_q1_x, double* d_q1_y,          // output: gradients w.r.t. q1
    double* d_q2_x, double* d_q2_y)          // output: gradients w.r.t. q2
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Forward pass: call line_intersection
    d2 p1 = make_d2(p1_x[idx], p1_y[idx]);
    d2 p2 = make_d2(p2_x[idx], p2_y[idx]);
    d2 q1 = make_d2(q1_x[idx], q1_y[idx]);
    d2 q2 = make_d2(q2_x[idx], q2_y[idx]);
    
    d2 out = line_intersection(p1, p2, q1, q2);
    out_x[idx] = out.x;
    out_y[idx] = out.y;
    
    // Backward pass: call backward_line_intersection (assuming d_out = (1.0, 1.0))
    d2 d_out = make_d2(1.0, 1.0);
    d2 d_p1, d_p2, d_q1, d_q2;
    backward_line_intersection(p1, p2, q1, q2, d_out, &d_p1, &d_p2, &d_q1, &d_q2);
    
    d_p1_x[idx] = d_p1.x;
    d_p1_y[idx] = d_p1.y;
    d_p2_x[idx] = d_p2.x;
    d_p2_y[idx] = d_p2.y;
    d_q1_x[idx] = d_q1.x;
    d_q1_y[idx] = d_q1.y;
    d_q2_x[idx] = d_q2.x;
    d_q2_y[idx] = d_q2.y;
}

}  // extern "C"
"""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _ensure_initialized() -> None:
    """Lazy initialization: compile CUDA kernels on first use.
    
    This function:
    - Combines primitive sources with test kernels
    - Writes source to disk for profiling
    - Compiles to PTX using nvcc
    - Loads compiled module and extracts kernel references
    """
    global _initialized, _raw_module
    global _line_intersection_kernel
    
    if _initialized:
        return
    
    # Create temp directory if it doesn't exist
    temp_dir = '/tmp/pack_cuda_primitives_test'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Write CUDA source to disk
    cu_path = os.path.join(temp_dir, 'primitives_test.cu')
    with open(cu_path, 'w', encoding='utf-8') as f:
        f.write(_TEST_KERNELS_SRC)
    
    # Ensure nvcc is in PATH
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH; please install CUDA toolkit")
    
    # Compile to PTX
    ptx_path = os.path.join(temp_dir, 'primitives_test.ptx')
    cmd = [
        nvcc_path,
        "-O3",
        "-use_fast_math",
        "-arch=sm_89",  # Adjust for your GPU architecture
        "-ptx",
        cu_path,
        "-o",
        ptx_path
    ]
    
    print("=== Compiling CUDA Primitives Test Kernels ===")
    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print(f"STDOUT:\n{proc.stdout}")
        print(f"STDERR:\n{proc.stderr}")
        raise RuntimeError(f"nvcc compilation failed (exit {proc.returncode})")
    
    print("Compilation successful!")
    
    # Load compiled PTX module
    _raw_module = cp.RawModule(path=ptx_path)
    
    # Extract kernel references
    _line_intersection_kernel = _raw_module.get_function("test_line_intersection_fwd_bwd")
    
    _initialized = True
    print("Initialization complete!")


# ---------------------------------------------------------------------------
# Python API: line_intersection
# ---------------------------------------------------------------------------

def line_intersection_fwd_bwd(
    p1: cp.ndarray,  # shape (n, 2) - line 1 point 1
    p2: cp.ndarray,  # shape (n, 2) - line 1 point 2
    q1: cp.ndarray,  # shape (n, 2) - line 2 point 1
    q2: cp.ndarray   # shape (n, 2) - line 2 point 2
) -> tuple[cp.ndarray, tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]]:
    """Compute line intersection points and gradients.
    
    Forward pass:
        Computes intersection of line (p1, p2) with line (q1, q2)
        Solves: p1 + t*(p2-p1) = q1 + u*(q2-q1)
    
    Backward pass:
        Computes gradients w.r.t. all four input points (assuming upstream gradient = (1.0, 1.0))
    
    Parameters
    ----------
    p1 : cp.ndarray, shape (n, 2), dtype float64
        First point of first line (batch)
    p2 : cp.ndarray, shape (n, 2), dtype float64
        Second point of first line (batch)
    q1 : cp.ndarray, shape (n, 2), dtype float64
        First point of second line (batch)
    q2 : cp.ndarray, shape (n, 2), dtype float64
        Second point of second line (batch)
    
    Returns
    -------
    intersection : cp.ndarray, shape (n, 2), dtype float64
        Intersection points for each pair of lines
    gradients : tuple of (d_p1, d_p2, d_q1, d_q2)
        d_p1 : cp.ndarray, shape (n, 2) - gradient w.r.t. p1
        d_p2 : cp.ndarray, shape (n, 2) - gradient w.r.t. p2
        d_q1 : cp.ndarray, shape (n, 2) - gradient w.r.t. q1
        d_q2 : cp.ndarray, shape (n, 2) - gradient w.r.t. q2
    
    Notes
    -----
    - All inputs must be C-contiguous float64 arrays
    - Batch size n must match across all inputs
    - Assumes lines are not parallel (no division by zero check)
    - Gradients computed w.r.t. upstream gradient d_out = (1.0, 1.0)
    """
    _ensure_initialized()
    
    # Validate inputs
    if p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError(f"p1 must be shape (n, 2), got {p1.shape}")
    if p2.ndim != 2 or p2.shape[1] != 2:
        raise ValueError(f"p2 must be shape (n, 2), got {p2.shape}")
    if q1.ndim != 2 or q1.shape[1] != 2:
        raise ValueError(f"q1 must be shape (n, 2), got {q1.shape}")
    if q2.ndim != 2 or q2.shape[1] != 2:
        raise ValueError(f"q2 must be shape (n, 2), got {q2.shape}")
    
    n = p1.shape[0]
    if p2.shape[0] != n or q1.shape[0] != n or q2.shape[0] != n:
        raise ValueError(f"Batch size mismatch: p1={n}, p2={p2.shape[0]}, q1={q1.shape[0]}, q2={q2.shape[0]}")
    
    if p1.dtype != cp.float64 or p2.dtype != cp.float64 or q1.dtype != cp.float64 or q2.dtype != cp.float64:
        raise ValueError("All inputs must be float64")
    
    if not (p1.flags.c_contiguous and p2.flags.c_contiguous and q1.flags.c_contiguous and q2.flags.c_contiguous):
        raise ValueError("All inputs must be C-contiguous")
    
    # Allocate outputs
    out = cp.zeros((n, 2), dtype=cp.float64)
    d_p1 = cp.zeros((n, 2), dtype=cp.float64)
    d_p2 = cp.zeros((n, 2), dtype=cp.float64)
    d_q1 = cp.zeros((n, 2), dtype=cp.float64)
    d_q2 = cp.zeros((n, 2), dtype=cp.float64)
    
    # Launch kernel
    if n == 0:
        return out, (d_p1, d_p2, d_q1, d_q2)
    
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    
    _line_intersection_kernel(
        (blocks,),
        (threads_per_block,),
        (
            p1[:, 0],  # p1_x
            p1[:, 1],  # p1_y
            p2[:, 0],  # p2_x
            p2[:, 1],  # p2_y
            q1[:, 0],  # q1_x
            q1[:, 1],  # q1_y
            q2[:, 0],  # q2_x
            q2[:, 1],  # q2_y
            np.int32(n),
            out[:, 0],     # out_x
            out[:, 1],     # out_y
            d_p1[:, 0],    # d_p1_x
            d_p1[:, 1],    # d_p1_y
            d_p2[:, 0],    # d_p2_x
            d_p2[:, 1],    # d_p2_y
            d_q1[:, 0],    # d_q1_x
            d_q1[:, 1],    # d_q1_y
            d_q2[:, 0],    # d_q2_x
            d_q2[:, 1],    # d_q2_y
        )
    )
    
    return out, (d_p1, d_p2, d_q1, d_q2)

def run_all_tests():
    test_line_intersection()

def test_line_intersection():
    """Test line_intersection forward and backward passes using finite differences."""
    print("=== Testing line_intersection ===")
    _ensure_initialized()
    
    eps = 1e-6
    
    # Start with known good test case
    test_data = [
        (np.array([0.0, 0.0]), np.array([2.0, 2.0]), 
         np.array([0.0, 2.0]), np.array([2.0, 0.0]))  # Intersect at (1, 1)
    ]
    
    # Generate more random test cases
    np.random.seed(42)
    n_tests = 30
    
    for _ in range(n_tests):
        # Generate points closer together so intersection is nearby
        center = np.random.randn(2) * 2.0
        p1_val = center + np.random.randn(2) * 1.0
        p2_val = center + np.random.randn(2) * 1.0
        q1_val = center + np.random.randn(2) * 1.0
        q2_val = center + np.random.randn(2) * 1.0
        
        # Check that lines aren't too parallel
        rx = p2_val[0] - p1_val[0]
        ry = p2_val[1] - p1_val[1]
        sx = q2_val[0] - q1_val[0]
        sy = q2_val[1] - q1_val[1]
        denom = rx * sy - ry * sx
        
        if abs(denom) > 0.3:  # Not too parallel
            test_data.append((p1_val, p2_val, q1_val, q2_val))
    
    print(f"Testing {len(test_data)} cases...")
    
    def compute_intersection(p1_val, p2_val, q1_val, q2_val):
        """Helper to compute intersection on CPU."""
        result, _ = line_intersection_fwd_bwd(
            cp.array([p1_val], dtype=cp.float64),
            cp.array([p2_val], dtype=cp.float64),
            cp.array([q1_val], dtype=cp.float64),
            cp.array([q2_val], dtype=cp.float64)
        )
        return result.get()[0]
    
    # Test each case individually
    max_error = 0.0
    
    for idx, (p1_val, p2_val, q1_val, q2_val) in enumerate(test_data):
        # Run forward and backward pass for this case
        intersection, (d_p1, d_p2, d_q1, d_q2) = line_intersection_fwd_bwd(
            cp.array([p1_val], dtype=cp.float64),
            cp.array([p2_val], dtype=cp.float64),
            cp.array([q1_val], dtype=cp.float64),
            cp.array([q2_val], dtype=cp.float64)
        )
        
        # Get analytical gradients for this case
        d_p1_case = d_p1.get()[0]
        d_p2_case = d_p2.get()[0]
        d_q1_case = d_q1.get()[0]
        d_q2_case = d_q2.get()[0]
        
        # Test all 8 gradient components
        for coord_idx, coord_name in [(0, 'x'), (1, 'y')]:
            # d_p1
            p1_plus = p1_val.copy()
            p1_plus[coord_idx] += eps
            p1_minus = p1_val.copy()
            p1_minus[coord_idx] -= eps
            jacobian_col = (compute_intersection(p1_plus, p2_val, q1_val, q2_val) - 
                           compute_intersection(p1_minus, p2_val, q1_val, q2_val)) / (2 * eps)
            expected = jacobian_col.sum()
            analytical = d_p1_case[coord_idx]
            error = abs(analytical - expected)
            max_error = max(max_error, error)
            assert error < 1e-4, f"Case {idx}, d_p1.{coord_name}: {analytical:.6f} vs {expected:.6f}"
            
            # d_p2
            p2_plus = p2_val.copy()
            p2_plus[coord_idx] += eps
            p2_minus = p2_val.copy()
            p2_minus[coord_idx] -= eps
            jacobian_col = (compute_intersection(p1_val, p2_plus, q1_val, q2_val) - 
                           compute_intersection(p1_val, p2_minus, q1_val, q2_val)) / (2 * eps)
            expected = jacobian_col.sum()
            analytical = d_p2_case[coord_idx]
            error = abs(analytical - expected)
            max_error = max(max_error, error)
            assert error < 1e-4, f"Case {idx}, d_p2.{coord_name}: {analytical:.6f} vs {expected:.6f}"
            
            # d_q1
            q1_plus = q1_val.copy()
            q1_plus[coord_idx] += eps
            q1_minus = q1_val.copy()
            q1_minus[coord_idx] -= eps
            jacobian_col = (compute_intersection(p1_val, p2_val, q1_plus, q2_val) - 
                           compute_intersection(p1_val, p2_val, q1_minus, q2_val)) / (2 * eps)
            expected = jacobian_col.sum()
            analytical = d_q1_case[coord_idx]
            error = abs(analytical - expected)
            max_error = max(max_error, error)
            assert error < 1e-4, f"Case {idx}, d_q1.{coord_name}: {analytical:.6f} vs {expected:.6f}"
            
            # d_q2
            q2_plus = q2_val.copy()
            q2_plus[coord_idx] += eps
            q2_minus = q2_val.copy()
            q2_minus[coord_idx] -= eps
            jacobian_col = (compute_intersection(p1_val, p2_val, q1_val, q2_plus) - 
                           compute_intersection(p1_val, p2_val, q1_val, q2_minus)) / (2 * eps)
            expected = jacobian_col.sum()
            analytical = d_q2_case[coord_idx]
            error = abs(analytical - expected)
            max_error = max(max_error, error)
            assert error < 1e-4, f"Case {idx}, d_q2.{coord_name}: {analytical:.6f} vs {expected:.6f}"
    
    print(f"✓ All {len(test_data)} cases passed, max gradient error: {max_error:.2e}")
    
    print("\n✓ All gradients match finite differences!")
    print("✓ line_intersection test PASSED")

if __name__ == "__main__":
    run_all_tests()