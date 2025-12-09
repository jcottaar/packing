"""
pack_cuda_primitives_test.py

Test suite for primitive CUDA device functions in pack_cuda_primitives.py.
This module exposes each primitive function via test kernels and provides
Python interfaces for both forward and backward passes.

Pattern:
- Each primitive gets a dedicated kernel that exposes forward + backward
- Python wrapper functions handle CuPy arrays and device memory
- Lazy initialization compiles kernels on first use
- Gradient computation is optional (controlled by compute_gradients parameter)
- Tests verify forward results match with/without gradient computation
"""

import numpy as np
import cupy as cp
import os
import subprocess
import shutil

import pack_cuda_primitives
import kaggle_support as kgs
kgs.set_float32(False)
print(kgs.USE_FLOAT32)

# ---------------------------------------------------------------------------
# Module State - Lazy Initialization
# ---------------------------------------------------------------------------

_raw_module: cp.RawModule | None = None
_initialized: bool = False

# Kernel references (populated during initialization)
_line_intersection_kernel: cp.RawKernel | None = None
_clip_against_edge_kernel: cp.RawKernel | None = None
_polygon_area_kernel: cp.RawKernel | None = None
_sat_separation_kernel: cp.RawKernel | None = None


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
    const int compute_gradients,              // whether to compute gradients (0 or 1)
    double* out_x, double* out_y,             // output: intersection points (batch)
    double* d_p1_x, double* d_p1_y,          // output: gradients w.r.t. p1 (can be NULL)
    double* d_p2_x, double* d_p2_y,          // output: gradients w.r.t. p2 (can be NULL)
    double* d_q1_x, double* d_q1_y,          // output: gradients w.r.t. q1 (can be NULL)
    double* d_q2_x, double* d_q2_y)          // output: gradients w.r.t. q2 (can be NULL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Forward pass: call line_intersection
    d2 p1 = make_double2(p1_x[idx], p1_y[idx]);
    d2 p2 = make_double2(p2_x[idx], p2_y[idx]);
    d2 q1 = make_double2(q1_x[idx], q1_y[idx]);
    d2 q2 = make_double2(q2_x[idx], q2_y[idx]);
    
    d2 out = line_intersection(p1, p2, q1, q2);
    out_x[idx] = out.x;
    out_y[idx] = out.y;
    
    // Backward pass: call backward_line_intersection (assuming d_out = (1.0, 1.0))
    if (compute_gradients) {
        d2 d_out = make_double2(1.0, 1.0);
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
}

// ============================================================================
// TEST KERNEL: sat_separation_with_grad_pose (single instance)
// ============================================================================
__global__ void test_sat_separation_fwd_bwd(
    const double* verts1_flat, const int n1,
    const double* verts2_flat, const int n2,
    const double x1, const double y1,
    const double cos_th1, const double sin_th1,
    const int compute_gradients,
    double* out_sep,
    double* out_dsep_dx, double* out_dsep_dy, double* out_dsep_dtheta)
{
    // Single-threaded invocation: only thread 0 performs work
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;

    d2 verts1_local[MAX_VERTS_PER_PIECE];
    for (int i = 0; i < n1; ++i) {
        int offset = i * 2;
        verts1_local[i] = make_double2(verts1_flat[offset], verts1_flat[offset + 1]);
    }

    d2 verts2_world[MAX_VERTS_PER_PIECE];
    for (int i = 0; i < n2; ++i) {
        int offset = i * 2;
        verts2_world[i] = make_double2(verts2_flat[offset], verts2_flat[offset + 1]);
    }

    double sep_val = 0.0;
    double dsep_dx = 0.0, dsep_dy = 0.0, dsep_dtheta = 0.0;

    sat_separation_with_grad_pose(verts1_local, n1, verts2_world, n2,
                                  x1, y1, cos_th1, sin_th1,
                                  &sep_val, &dsep_dx, &dsep_dy, &dsep_dtheta);

    out_sep[0] = sep_val;
    out_dsep_dx[0] = dsep_dx;
    out_dsep_dy[0] = dsep_dy;
    out_dsep_dtheta[0] = dsep_dtheta;
}

// ============================================================================
// TEST KERNEL: clip_against_edge
// ============================================================================
__global__ void test_clip_against_edge_fwd_bwd(
    const double* in_pts_flat,     // input: polygon vertices (batch, max_n_in*2)
    const int* n_in,                // input: number of vertices per polygon (batch)
    const double* edge_A_x, const double* edge_A_y,  // input: edge point A (batch)
    const double* edge_B_x, const double* edge_B_y,  // input: edge point B (batch)
    const int batch_size,
    const int max_n_in,             // max vertices in input polygons
    const int compute_gradients,    // whether to compute gradients (0 or 1)
    double* out_pts_flat,           // output: clipped vertices (batch, MAX_INTERSECTION_VERTS*2)
    int* n_out,                     // output: number of output vertices (batch)
    double* d_in_pts_flat,          // output: gradients w.r.t. input vertices (can be NULL)
    double* d_edge_A_x, double* d_edge_A_y,  // output: gradients w.r.t. edge A (can be NULL)
    double* d_edge_B_x, double* d_edge_B_y)  // output: gradients w.r.t. edge B (can be NULL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Load input polygon
    int n_verts = n_in[idx];
    d2 in_pts[MAX_INTERSECTION_VERTS];
    for (int i = 0; i < n_verts; ++i) {
        int offset = idx * max_n_in * 2 + i * 2;
        in_pts[i] = make_double2(in_pts_flat[offset], in_pts_flat[offset + 1]);
    }
    
    // Load edge
    d2 A = make_double2(edge_A_x[idx], edge_A_y[idx]);
    d2 B = make_double2(edge_B_x[idx], edge_B_y[idx]);
    
    // Forward pass: call clip_against_edge with metadata
    d2 out_pts[MAX_INTERSECTION_VERTS];
    ClipMetadata meta;
    int n_out_verts = clip_against_edge(in_pts, n_verts, A, B, out_pts, &meta);
    
    // Store output
    n_out[idx] = n_out_verts;
    for (int i = 0; i < n_out_verts; ++i) {
        int offset = idx * MAX_INTERSECTION_VERTS * 2 + i * 2;
        out_pts_flat[offset] = out_pts[i].x;
        out_pts_flat[offset + 1] = out_pts[i].y;
    }
    
    // Backward pass: assume d_out = all ones
    if (compute_gradients) {
        d2 d_out_pts[MAX_INTERSECTION_VERTS];
        for (int i = 0; i < n_out_verts; ++i) {
            d_out_pts[i] = make_double2(1.0, 1.0);
        }
        
        d2 d_in_pts[MAX_INTERSECTION_VERTS];
        d2 d_A, d_B;
        backward_clip_against_edge(in_pts, n_verts, A, B, d_out_pts, &meta, d_in_pts, &d_A, &d_B);
        
        // Store gradients
        for (int i = 0; i < n_verts; ++i) {
            int offset = idx * max_n_in * 2 + i * 2;
            d_in_pts_flat[offset] = d_in_pts[i].x;
            d_in_pts_flat[offset + 1] = d_in_pts[i].y;
        }
        
        d_edge_A_x[idx] = d_A.x;
        d_edge_A_y[idx] = d_A.y;
        d_edge_B_x[idx] = d_B.x;
        d_edge_B_y[idx] = d_B.y;
    }
}

// ============================================================================
// TEST KERNEL: polygon_area
// ============================================================================
__global__ void test_polygon_area_fwd_bwd(
    const double* verts_flat,  // shape: (batch, max_n_verts * 2) - flattened (x,y) coords
    const int* n_verts,        // shape: (batch,) - number of vertices per polygon
    int batch_size,
    int max_n_verts,
    int compute_gradients,     // whether to compute gradients (0 or 1)
    double* areas,             // output: (batch,) - area of each polygon
    double* d_verts_flat)      // output: (batch, max_n_verts * 2) - gradients w.r.t. vertices (can be NULL)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int n = n_verts[idx];
    
    // Load vertices
    d2 v[MAX_VERTS_PER_PIECE];
    for (int i = 0; i < n; ++i) {
        int offset = idx * max_n_verts * 2 + i * 2;
        v[i].x = verts_flat[offset];
        v[i].y = verts_flat[offset + 1];
    }
    
    // Merged function: compute area and gradients
    d2 d_v[MAX_VERTS_PER_PIECE];
    double area = polygon_area(v, n, d_v);
    areas[idx] = area;
    
    // Store gradients if requested
    if (compute_gradients) {
        for (int i = 0; i < n; ++i) {
            int offset = idx * max_n_verts * 2 + i * 2;
            d_verts_flat[offset] = d_v[i].x;
            d_verts_flat[offset + 1] = d_v[i].y;
        }
    }
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
    global _line_intersection_kernel, _clip_against_edge_kernel, _polygon_area_kernel, _sat_separation_kernel
    
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
    _clip_against_edge_kernel = _raw_module.get_function("test_clip_against_edge_fwd_bwd")
    _polygon_area_kernel = _raw_module.get_function("test_polygon_area_fwd_bwd")
    _sat_separation_kernel = _raw_module.get_function("test_sat_separation_fwd_bwd")    
    
    _initialized = True
    print("Initialization complete!")


# ---------------------------------------------------------------------------
# Python API: line_intersection
# ---------------------------------------------------------------------------

def line_intersection_fwd_bwd(
    p1: cp.ndarray,  # shape (n, 2) - line 1 point 1
    p2: cp.ndarray,  # shape (n, 2) - line 1 point 2
    q1: cp.ndarray,  # shape (n, 2) - line 2 point 1
    q2: cp.ndarray,  # shape (n, 2) - line 2 point 2
    compute_gradients: bool = True
) -> tuple[cp.ndarray, tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray] | None]:
    """Compute line intersection points and optionally gradients.
    
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
    compute_gradients : bool, optional
        If True, compute and return gradients. Default: True.
    
    Returns
    -------
    intersection : cp.ndarray, shape (n, 2), dtype float64
        Intersection points for each pair of lines
    gradients : tuple of (d_p1, d_p2, d_q1, d_q2) or None
        If compute_gradients=True:
            d_p1 : cp.ndarray, shape (n, 2) - gradient w.r.t. p1
            d_p2 : cp.ndarray, shape (n, 2) - gradient w.r.t. p2
            d_q1 : cp.ndarray, shape (n, 2) - gradient w.r.t. q1
            d_q2 : cp.ndarray, shape (n, 2) - gradient w.r.t. q2
        If compute_gradients=False: None
    
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
    
    if p1.dtype != kgs.dtype_cp or p2.dtype != kgs.dtype_cp or q1.dtype != kgs.dtype_cp or q2.dtype != kgs.dtype_cp:
        raise ValueError("All inputs must be float64")
    
    if not (p1.flags.c_contiguous and p2.flags.c_contiguous and q1.flags.c_contiguous and q2.flags.c_contiguous):
        raise ValueError("All inputs must be C-contiguous")
    
    # Allocate outputs
    out = cp.zeros((n, 2), dtype=kgs.dtype_cp)
    
    if compute_gradients:
        d_p1 = cp.zeros((n, 2), dtype=kgs.dtype_cp)
        d_p2 = cp.zeros((n, 2), dtype=kgs.dtype_cp)
        d_q1 = cp.zeros((n, 2), dtype=kgs.dtype_cp)
        d_q2 = cp.zeros((n, 2), dtype=kgs.dtype_cp)
    else:
        # Use dummy arrays (won't be written to)
        dummy = cp.zeros((1, 2), dtype=kgs.dtype_cp)
        d_p1 = d_p2 = d_q1 = d_q2 = dummy
    
    # Launch kernel
    if n == 0:
        return out, (d_p1, d_p2, d_q1, d_q2) if compute_gradients else None
    
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
            np.int32(1 if compute_gradients else 0),
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
    
    if compute_gradients:
        return out, (d_p1, d_p2, d_q1, d_q2)
    else:
        return out, None


# ---------------------------------------------------------------------------
# Python API: clip_against_edge
# ---------------------------------------------------------------------------

def clip_against_edge_fwd_bwd(
    in_polygons: list[cp.ndarray],  # list of (n_i, 2) arrays - input polygons
    edge_A: cp.ndarray,              # shape (n_batch, 2) - edge point A
    edge_B: cp.ndarray,              # shape (n_batch, 2) - edge point B
    compute_gradients: bool = True
) -> tuple[list[cp.ndarray], tuple[list[cp.ndarray], cp.ndarray, cp.ndarray] | None]:
    """Clip polygons against an edge and optionally compute gradients.
    
    Forward pass:
        Clips each input polygon against the half-plane defined by edge A->B
        (keeps points where cross(A, B, point) >= 0)
    
    Backward pass:
        Computes gradients w.r.t. input vertices and edge points
        (assuming upstream gradient = 1.0 for all output vertices)
    
    Parameters
    ----------
    in_polygons : list of cp.ndarray
        List of input polygons, each shape (n_i, 2), dtype float64
    edge_A : cp.ndarray, shape (n_batch, 2), dtype float64
        First point of clipping edge for each polygon
    edge_B : cp.ndarray, shape (n_batch, 2), dtype float64
        Second point of clipping edge for each polygon
    compute_gradients : bool, optional
        If True, compute and return gradients. Default: True.
    
    Returns
    -------
    out_polygons : list of cp.ndarray
        Clipped polygons, each shape (n_out_i, 2), dtype float64
    gradients : tuple of (d_in_polygons, d_edge_A, d_edge_B) or None
        If compute_gradients=True:
            d_in_polygons : list of cp.ndarray, each shape (n_i, 2) - gradients w.r.t. input vertices
            d_edge_A : cp.ndarray, shape (n_batch, 2) - gradients w.r.t. edge A
            d_edge_B : cp.ndarray, shape (n_batch, 2) - gradients w.r.t. edge B
        If compute_gradients=False: None
    """
    _ensure_initialized()
    
    n_batch = len(in_polygons)
    if n_batch == 0:
        if compute_gradients:
            return [], ([], cp.array([], dtype=kgs.dtype_cp).reshape(0, 2), 
                        cp.array([], dtype=kgs.dtype_cp).reshape(0, 2))
        else:
            return [], None
    
    # Validate inputs
    assert edge_A.shape == (n_batch, 2) and edge_A.dtype == kgs.dtype_cp
    assert edge_B.shape == (n_batch, 2) and edge_B.dtype == kgs.dtype_cp
    
    # Find max input size and pad
    max_n_in = max(poly.shape[0] for poly in in_polygons)
    
    # Create padded input array
    in_pts_flat = cp.zeros((n_batch, max_n_in, 2), dtype=kgs.dtype_cp)
    n_in = cp.zeros(n_batch, dtype=cp.int32)
    
    for i, poly in enumerate(in_polygons):
        assert poly.ndim == 2 and poly.shape[1] == 2
        assert poly.dtype == kgs.dtype_cp
        n_verts = poly.shape[0]
        n_in[i] = n_verts
        in_pts_flat[i, :n_verts, :] = poly
    
    in_pts_flat = in_pts_flat.reshape(n_batch, max_n_in * 2)
    
    # Allocate outputs
    out_pts_flat = cp.zeros((n_batch, 8 * 2), dtype=kgs.dtype_cp)  # MAX_INTERSECTION_VERTS=8
    n_out = cp.zeros(n_batch, dtype=cp.int32)
    
    if compute_gradients:
        d_in_pts_flat = cp.zeros((n_batch, max_n_in * 2), dtype=kgs.dtype_cp)
        d_edge_A = cp.zeros((n_batch, 2), dtype=kgs.dtype_cp)
        d_edge_B = cp.zeros((n_batch, 2), dtype=kgs.dtype_cp)
    else:
        # Use dummy arrays (won't be written to)
        dummy = cp.zeros((1,), dtype=kgs.dtype_cp)
        d_in_pts_flat = dummy
        d_edge_A = dummy.reshape(1, 1)
        d_edge_B = dummy.reshape(1, 1)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (n_batch + threads_per_block - 1) // threads_per_block
    
    _clip_against_edge_kernel(
        (blocks,),
        (threads_per_block,),
        (
            in_pts_flat.ravel(),
            n_in,
            edge_A[:, 0],
            edge_A[:, 1],
            edge_B[:, 0],
            edge_B[:, 1],
            np.int32(n_batch),
            np.int32(max_n_in),
            np.int32(1 if compute_gradients else 0),
            out_pts_flat.ravel(),
            n_out,
            d_in_pts_flat.ravel() if compute_gradients else dummy.ravel(),
            d_edge_A[:, 0] if compute_gradients else dummy,
            d_edge_A[:, 1] if compute_gradients else dummy,
            d_edge_B[:, 0] if compute_gradients else dummy,
            d_edge_B[:, 1] if compute_gradients else dummy,
        )
    )
    
    # Extract output polygons
    n_out_cpu = n_out.get()
    out_pts_flat = out_pts_flat.reshape(n_batch, 8, 2)
    out_polygons = []
    for i in range(n_batch):
        n = n_out_cpu[i]
        out_polygons.append(out_pts_flat[i, :n, :].copy())
    
    if compute_gradients:
        # Extract gradient polygons
        d_in_pts_flat = d_in_pts_flat.reshape(n_batch, max_n_in, 2)
        d_in_polygons = []
        for i in range(n_batch):
            n = n_in.get()[i]
            d_in_polygons.append(d_in_pts_flat[i, :n, :].copy())
        
        return out_polygons, (d_in_polygons, d_edge_A, d_edge_B)
    else:
        return out_polygons, None


def run_all_tests():
    test_line_intersection()
    test_clip_against_edge()

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
            cp.array([p1_val], dtype=kgs.dtype_cp),
            cp.array([p2_val], dtype=kgs.dtype_cp),
            cp.array([q1_val], dtype=kgs.dtype_cp),
            cp.array([q2_val], dtype=kgs.dtype_cp)
        )
        return result.get()[0]
    
    # Test each case individually
    max_error = 0.0
    
    for idx, (p1_val, p2_val, q1_val, q2_val) in enumerate(test_data):
        # Run forward+backward pass for this case
        intersection_with_grad, (d_p1, d_p2, d_q1, d_q2) = line_intersection_fwd_bwd(
            cp.array([p1_val], dtype=kgs.dtype_cp),
            cp.array([p2_val], dtype=kgs.dtype_cp),
            cp.array([q1_val], dtype=kgs.dtype_cp),
            cp.array([q2_val], dtype=kgs.dtype_cp),
            compute_gradients=True
        )
        
        # Run forward-only pass
        intersection_no_grad, grads = line_intersection_fwd_bwd(
            cp.array([p1_val], dtype=kgs.dtype_cp),
            cp.array([p2_val], dtype=kgs.dtype_cp),
            cp.array([q1_val], dtype=kgs.dtype_cp),
            cp.array([q2_val], dtype=kgs.dtype_cp),
            compute_gradients=False
        )
        
        # Check that forward results match
        assert grads is None, "Expected None when compute_gradients=False"
        assert cp.allclose(intersection_with_grad, intersection_no_grad), \
            f"Case {idx}: Forward results differ with/without gradients"
        if idx == 0:
            print("✓ Forward results match with and without gradients")
        
        intersection = intersection_with_grad
        
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


def test_clip_against_edge():
    """Test clip_against_edge forward and backward passes using finite differences."""
    print("\n=== Testing clip_against_edge ===")
    _ensure_initialized()
    
    eps = 1e-6
    
    # Test cases: various polygons and clipping edges
    test_cases = [
        # Square clipped by diagonal edge - should produce triangle
        (np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]),
         np.array([0.0, 0.0]), np.array([2.0, 2.0])),
        
        # Triangle clipped by edge
        (np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]),
         np.array([0.0, 1.0]), np.array([2.0, 1.0])),
         
        # Pentagon clipped
        (np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.0], [1.0, 2.0], [-0.5, 1.0]]),
         np.array([0.5, 0.5]), np.array([1.5, 1.5])),
    ]
    
    # Add random test cases
    np.random.seed(123)
    for _ in range(10):
        # Random convex polygon (from circle)
        n_verts = np.random.randint(3, 6)
        angles = np.sort(np.random.rand(n_verts) * 2 * np.pi)
        radius = 1.0 + np.random.rand() * 0.5
        poly = np.column_stack([
            np.cos(angles) * radius,
            np.sin(angles) * radius
        ])
        
        # Random edge
        edge_A = np.random.randn(2) * 2.0
        edge_B = edge_A + np.random.randn(2) * 2.0
        test_cases.append((poly, edge_A, edge_B))
    
    print(f"Testing {len(test_cases)} cases...")
    
    max_error = 0.0
    
    for idx, (poly_np, edge_A_np, edge_B_np) in enumerate(test_cases):
        # Convert to CuPy
        poly = cp.array(poly_np, dtype=kgs.dtype_cp)
        edge_A = cp.array([edge_A_np], dtype=kgs.dtype_cp)
        edge_B = cp.array([edge_B_np], dtype=kgs.dtype_cp)
        
        # Forward+backward pass
        out_polys_with_grad, (d_in_polys, d_edge_A, d_edge_B) = clip_against_edge_fwd_bwd(
            [poly], edge_A, edge_B, compute_gradients=True)
        
        # Forward-only pass
        out_polys_no_grad, grads = clip_against_edge_fwd_bwd(
            [poly], edge_A, edge_B, compute_gradients=False)
        
        # Check that forward results match
        assert grads is None, "Expected None when compute_gradients=False"
        assert len(out_polys_with_grad) == len(out_polys_no_grad), \
            f"Case {idx}: Number of output polygons differ"
        assert len(out_polys_with_grad[0]) == len(out_polys_no_grad[0]), \
            f"Case {idx}: Output polygon sizes differ"
        assert cp.allclose(out_polys_with_grad[0], out_polys_no_grad[0]), \
            f"Case {idx}: Forward results differ with/without gradients"
        if idx == 0:
            print("✓ Forward results match with and without gradients")
        
        out_polys = out_polys_with_grad
        
        if len(out_polys[0]) == 0:
            # Polygon completely clipped away - skip gradient test
            continue
        
        # Get gradients
        d_poly = d_in_polys[0].get()
        d_A = d_edge_A.get()[0]
        d_B = d_edge_B.get()[0]
        
        # Test gradients with finite differences
        # For each input vertex coordinate
        for v_idx in range(poly_np.shape[0]):
            for coord_idx in range(2):
                # Perturb input
                poly_plus = poly_np.copy()
                poly_plus[v_idx, coord_idx] += eps
                poly_minus = poly_np.copy()
                poly_minus[v_idx, coord_idx] -= eps
                
                # Compute forward passes
                out_plus, _ = clip_against_edge_fwd_bwd(
                    [cp.array(poly_plus, dtype=kgs.dtype_cp)],
                    cp.array([edge_A_np], dtype=kgs.dtype_cp),
                    cp.array([edge_B_np], dtype=kgs.dtype_cp)
                )
                out_minus, _ = clip_against_edge_fwd_bwd(
                    [cp.array(poly_minus, dtype=kgs.dtype_cp)],
                    cp.array([edge_A_np], dtype=kgs.dtype_cp),
                    cp.array([edge_B_np], dtype=kgs.dtype_cp)
                )
                
                # Check if topology changed (number of vertices)
                if len(out_plus[0]) != len(out_polys[0]) or len(out_minus[0]) != len(out_polys[0]):
                    # Topology changed - skip this gradient test
                    continue
                
                # Finite difference: sum of all output coordinates (since d_out = all ones)
                fd_grad = (out_plus[0].get().sum() - out_minus[0].get().sum()) / (2 * eps)
                analytical = d_poly[v_idx, coord_idx]
                error = abs(analytical - fd_grad)
                max_error = max(max_error, error)
                assert error < 1e-3, f"Case {idx}, vertex {v_idx}, coord {coord_idx}: {analytical:.6f} vs {fd_grad:.6f}"
        
        # Test edge gradients
        for coord_idx in range(2):
            # Test d_edge_A
            edge_A_plus = edge_A_np.copy()
            edge_A_plus[coord_idx] += eps
            edge_A_minus = edge_A_np.copy()
            edge_A_minus[coord_idx] -= eps
            
            out_plus, _ = clip_against_edge_fwd_bwd(
                [cp.array(poly_np, dtype=kgs.dtype_cp)],
                cp.array([edge_A_plus], dtype=kgs.dtype_cp),
                cp.array([edge_B_np], dtype=kgs.dtype_cp)
            )
            out_minus, _ = clip_against_edge_fwd_bwd(
                [cp.array(poly_np, dtype=kgs.dtype_cp)],
                cp.array([edge_A_minus], dtype=kgs.dtype_cp),
                cp.array([edge_B_np], dtype=kgs.dtype_cp)
            )
            
            if len(out_plus[0]) != len(out_polys[0]) or len(out_minus[0]) != len(out_polys[0]):
                continue
            
            fd_grad = (out_plus[0].get().sum() - out_minus[0].get().sum()) / (2 * eps)
            analytical = d_A[coord_idx]
            error = abs(analytical - fd_grad)
            max_error = max(max_error, error)
            assert error < 1e-3, f"Case {idx}, edge_A coord {coord_idx}: {analytical:.6f} vs {fd_grad:.6f}"
            
            # Test d_edge_B
            edge_B_plus = edge_B_np.copy()
            edge_B_plus[coord_idx] += eps
            edge_B_minus = edge_B_np.copy()
            edge_B_minus[coord_idx] -= eps
            
            out_plus, _ = clip_against_edge_fwd_bwd(
                [cp.array(poly_np, dtype=kgs.dtype_cp)],
                cp.array([edge_A_np], dtype=kgs.dtype_cp),
                cp.array([edge_B_plus], dtype=kgs.dtype_cp)
            )
            out_minus, _ = clip_against_edge_fwd_bwd(
                [cp.array(poly_np, dtype=kgs.dtype_cp)],
                cp.array([edge_A_np], dtype=kgs.dtype_cp),
                cp.array([edge_B_minus], dtype=kgs.dtype_cp)
            )
            
            if len(out_plus[0]) != len(out_polys[0]) or len(out_minus[0]) != len(out_polys[0]):
                continue
            
            fd_grad = (out_plus[0].get().sum() - out_minus[0].get().sum()) / (2 * eps)
            analytical = d_B[coord_idx]
            error = abs(analytical - fd_grad)
            max_error = max(max_error, error)
            assert error < 1e-3, f"Case {idx}, edge_B coord {coord_idx}: {analytical:.6f} vs {fd_grad:.6f}"
    
    print(f"✓ All test cases passed, max gradient error: {max_error:.2e}")
    print("✓ clip_against_edge test PASSED")


# ---------------------------------------------------------------------------
# Python API: polygon_area
# ---------------------------------------------------------------------------

def polygon_area_fwd_bwd(
    polygons: list[cp.ndarray],  # list of (n_verts, 2) arrays
    compute_gradients: bool = True
) -> tuple[cp.ndarray, list[cp.ndarray] | None]:
    """Compute polygon areas and optionally gradients.
    
    Forward pass:
        Computes signed area using shoelace formula, returns absolute value
    
    Backward pass:
        Computes gradients w.r.t. all vertices (assuming upstream gradient = 1.0)
    
    Parameters
    ----------
    polygons : list of cp.ndarray
        Each element is shape (n_verts, 2), dtype float64
        Vertices of a convex polygon
    compute_gradients : bool, optional
        If True, compute and return gradients. Default: True.
    
    Returns
    -------
    areas : cp.ndarray, shape (batch,), dtype float64
        Area of each polygon
    gradients : list of cp.ndarray or None
        If compute_gradients=True:
            Each element is shape (n_verts, 2) - gradient w.r.t. vertices
        If compute_gradients=False: None
    
    Notes
    -----
    - All polygons must have <= MAX_VERTS_PER_PIECE vertices
    - Gradients computed w.r.t. upstream gradient d_area = 1.0
    """
    _ensure_initialized()
    
    batch_size = len(polygons)
    if batch_size == 0:
        return cp.array([], dtype=kgs.dtype_cp), [] if compute_gradients else None
    
    # Determine max_n_verts
    max_n_verts = max(poly.shape[0] for poly in polygons)
    if max_n_verts > 4:  # MAX_VERTS_PER_PIECE
        raise ValueError(f"Polygon has {max_n_verts} vertices, max is 4")
    
    # Prepare inputs
    n_verts_list = [poly.shape[0] for poly in polygons]
    n_verts = cp.array(n_verts_list, dtype=cp.int32)
    
    # Flatten vertices
    verts_flat = cp.zeros((batch_size, max_n_verts * 2), dtype=kgs.dtype_cp)
    for i, poly in enumerate(polygons):
        n = poly.shape[0]
        verts_flat[i, :n*2] = poly.flatten()
    
    # Allocate outputs
    areas = cp.zeros(batch_size, dtype=kgs.dtype_cp)
    
    if compute_gradients:
        d_verts_flat = cp.zeros((batch_size, max_n_verts * 2), dtype=kgs.dtype_cp)
    else:
        # Use dummy array (won't be written to)
        d_verts_flat = cp.zeros((1,), dtype=kgs.dtype_cp)
    
    # Launch kernel
    threads_per_block = 256
    blocks = (batch_size + threads_per_block - 1) // threads_per_block
    
    _polygon_area_kernel(
        (blocks,), (threads_per_block,),
        (verts_flat, n_verts, batch_size, max_n_verts, 
         np.int32(1 if compute_gradients else 0), areas, d_verts_flat)
    )
    
    if compute_gradients:
        # Extract gradients
        gradients = []
        for i in range(batch_size):
            n = n_verts_list[i]
            d_verts = d_verts_flat[i, :n*2].reshape(n, 2)
            gradients.append(d_verts)
        
        return areas, gradients
    else:
        return areas, None


def sat_separation_with_grad_pose_fwd_bwd(
    verts1: cp.ndarray,  # shape (n1, 2)
    verts2: cp.ndarray,  # shape (n2, 2)
    x1: float, y1: float, cos_th1: float, sin_th1: float,
    compute_gradients: bool = True
):
    """Expose sat_separation_with_grad_pose via a single-thread test kernel.

    Minimal wrapper (no input validation). Allocates outputs.
    """
    _ensure_initialized()

    n1 = int(verts1.shape[0])
    n2 = int(verts2.shape[0])

    verts1_flat = cp.zeros((n1 * 2,), dtype=kgs.dtype_cp)
    verts1_flat[:n1*2] = verts1.flatten()

    verts2_flat = cp.zeros((n2 * 2,), dtype=kgs.dtype_cp)
    verts2_flat[:n2*2] = verts2.flatten()

    out_sep = cp.zeros((1,), dtype=kgs.dtype_cp)
    if compute_gradients:
        out_dx = cp.zeros((1,), dtype=kgs.dtype_cp)
        out_dy = cp.zeros((1,), dtype=kgs.dtype_cp)
        out_dtheta = cp.zeros((1,), dtype=kgs.dtype_cp)
    else:
        out_dx = out_dy = out_dtheta = cp.zeros((1,), dtype=kgs.dtype_cp)

    # Launch single-thread kernel
    _sat_separation_kernel(
        (1,), (1,),
        (
            verts1_flat, np.int32(n1),
            verts2_flat, np.int32(n2),
            float(x1), float(y1), float(cos_th1), float(sin_th1),
            np.int32(1 if compute_gradients else 0),
            out_sep,
            out_dx, out_dy, out_dtheta,
        )
    )

    if compute_gradients:
        return out_sep[0], (out_dx[0], out_dy[0], out_dtheta[0])
    else:
        return out_sep[0], None


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

def test_polygon_area():
    """Test polygon_area and backward_polygon_area using finite differences."""
    print("\n=== Testing polygon_area ===")
    
    eps = 1e-6
    max_error = 0.0
    
    # Fixed test cases
    test_cases = [
        # Square
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        # Triangle
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        # Rectangle
        np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]),
    ]
    
    # Add random test cases
    np.random.seed(456)
    for _ in range(10):
        n_verts = np.random.randint(3, 5)
        # Generate random convex polygon by sorting angles
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_verts))
        radii = np.random.uniform(0.5, 2.0, n_verts)
        verts = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        test_cases.append(verts)
    
    # Test each case
    for idx, poly_np in enumerate(test_cases):
        poly_gpu = cp.array(poly_np, dtype=kgs.dtype_cp)
        
        # Forward+backward
        areas_with_grad, grads = polygon_area_fwd_bwd([poly_gpu], compute_gradients=True)
        
        # Forward-only
        areas_no_grad, grads_none = polygon_area_fwd_bwd([poly_gpu], compute_gradients=False)
        
        # Check that forward results match
        assert grads_none is None, "Expected None when compute_gradients=False"
        assert cp.allclose(areas_with_grad, areas_no_grad), \
            f"Case {idx}: Forward results differ with/without gradients"
        if idx == 0:
            print("✓ Forward results match with and without gradients")
        
        area = float(areas_with_grad[0])
        d_verts = grads[0].get()
        
        n_verts = poly_np.shape[0]
        
        # Finite difference for each vertex coordinate
        for vert_idx in range(n_verts):
            for coord_idx in range(2):
                poly_plus = poly_np.copy()
                poly_plus[vert_idx, coord_idx] += eps
                poly_minus = poly_np.copy()
                poly_minus[vert_idx, coord_idx] -= eps
                
                areas_plus, _ = polygon_area_fwd_bwd([cp.array(poly_plus, dtype=kgs.dtype_cp)])
                areas_minus, _ = polygon_area_fwd_bwd([cp.array(poly_minus, dtype=kgs.dtype_cp)])
                
                fd_grad = (float(areas_plus[0]) - float(areas_minus[0])) / (2 * eps)
                analytical = d_verts[vert_idx, coord_idx]
                error = abs(analytical - fd_grad)
                max_error = max(max_error, error)
                assert error < 1e-4, f"Case {idx}, vertex {vert_idx}, coord {coord_idx}: {analytical:.6f} vs {fd_grad:.6f}"
    
    print(f"✓ All {len(test_cases)} cases passed, max gradient error: {max_error:.2e}")
    print("✓ polygon_area test PASSED")


def run_all_tests():
    """Run all primitive tests."""
    kgs.set_float32(False)
    test_line_intersection()
    test_clip_against_edge()
    test_polygon_area()
    print('KERNEL', _sat_separation_kernel)
    # No all tests passed message -> intentional


if __name__ == "__main__":
    run_all_tests()