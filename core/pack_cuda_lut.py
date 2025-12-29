"""
LUT-based overlap computation for tree packing.

Uses a precomputed 3D lookup table indexed by relative pose (dx, dy, dtheta)
to compute overlap between tree pairs, replacing expensive polygon intersection.

Two modes are available:
- Array mode (USE_TEXTURE=False): Manual trilinear interpolation from global memory.
  Uses full float32/float64 precision for interpolation weights.
- Texture mode (USE_TEXTURE=True): Hardware trilinear interpolation via 3D texture.
  Faster but uses 9-bit fixed-point for interpolation weights (~0.2% precision).
  Results will differ slightly from array mode due to hardware precision limits.
"""
from __future__ import annotations

import numpy as np
import cupy as cp

import kaggle_support as kgs
import os
import subprocess
import shutil
import math

MAX_RADIUS = kgs.tree_max_radius

# Module setting to switch between array and texture modes
# Set this before calling _ensure_initialized() or reinitialize()
USE_TEXTURE: bool = True

# LUT arrays - must be set before calling _ensure_initialized()
# These define the grid and values for trilinear interpolation
LUT_X: np.ndarray | None = None      # shape (N_x,) - x coordinates of grid points
LUT_Y: np.ndarray | None = None      # shape (N_y,) - y coordinates of grid points  
LUT_theta: np.ndarray | None = None  # shape (N_theta,) - theta coordinates of grid points
LUT_vals: np.ndarray | None = None   # shape (N_x, N_y, N_theta) - overlap values at grid points

# Device-side LUT array (used in array mode)
_lut_d: cp.ndarray | None = None

# Texture object (used in texture mode)
_texture: cp.cuda.texture.TextureObject | None = None
_texture_array: cp.cuda.texture.CUDAarray | None = None

_CUDA_SRC = r"""
extern "C" {

#define MAX_RADIUS $MAX_RADIUS$
#define M_PI 3.14159265358979323846

// LUT grid parameters (set at compile time via string substitution)
#define LUT_N_X $LUT_N_X$
#define LUT_N_Y $LUT_N_Y$
#define LUT_N_THETA $LUT_N_THETA$
#define LUT_X_MIN $LUT_X_MIN$
#define LUT_X_MAX $LUT_X_MAX$
#define LUT_Y_MIN $LUT_Y_MIN$
#define LUT_Y_MAX $LUT_Y_MAX$
#define LUT_THETA_MIN $LUT_THETA_MIN$
#define LUT_THETA_MAX $LUT_THETA_MAX$

// Texture mode switch (0 = array, 1 = texture)
#define USE_TEXTURE $USE_TEXTURE$

// Global device state - set once by the kernel, used by all device functions
__device__ const double* g_lut;           // LUT array pointer (array mode)
__device__ cudaTextureObject_t g_tex;     // texture object (texture mode)

// Wrap angle to [-pi, pi]
__device__ __forceinline__ double wrap_angle(double theta) {
    while (theta > M_PI) theta -= 2.0 * M_PI;
    while (theta < -M_PI) theta += 2.0 * M_PI;
    return theta;
}

// LUT lookup - uses global g_tex or g_lut depending on mode
__device__ double lut_lookup(double x, double y, double theta)
{
#if USE_TEXTURE
    // Texture-based lookup with hardware trilinear interpolation
    // Normalize to grid coordinates [0.5, N-0.5] for texel centers
    double gx = (x - LUT_X_MIN) / (LUT_X_MAX - LUT_X_MIN) * (LUT_N_X - 1) + 0.5;
    double gy = (y - LUT_Y_MIN) / (LUT_Y_MAX - LUT_Y_MIN) * (LUT_N_Y - 1) + 0.5;
    double gt = (theta - LUT_THETA_MIN) / (LUT_THETA_MAX - LUT_THETA_MIN) * (LUT_N_THETA - 1) + 0.5;
    
    // tex3D(tex, x, y, z) for array shape (depth, height, width):
    //   x -> width index, y -> height index, z -> depth index
    // Texture shape is (N_theta, N_y, N_x) after transpose, so:
    //   width = N_x -> x = gx
    //   height = N_y -> y = gy
    //   depth = N_theta -> z = gt
    return (double)tex3D<float>(g_tex, (float)gx, (float)gy, (float)gt);
#else
    // Array-based trilinear interpolation with 8-point manual fetch
    // Normalize to grid coordinates [0, N-1]
    double gx = (x - LUT_X_MIN) / (LUT_X_MAX - LUT_X_MIN) * (LUT_N_X - 1);
    double gy = (y - LUT_Y_MIN) / (LUT_Y_MAX - LUT_Y_MIN) * (LUT_N_Y - 1);
    double gt = (theta - LUT_THETA_MIN) / (LUT_THETA_MAX - LUT_THETA_MIN) * (LUT_N_THETA - 1);
    
    // Clamp to valid range
    if (gx < 0.0) gx = 0.0;
    if (gx > LUT_N_X - 1.0) gx = LUT_N_X - 1.0;
    if (gy < 0.0) gy = 0.0;
    if (gy > LUT_N_Y - 1.0) gy = LUT_N_Y - 1.0;
    if (gt < 0.0) gt = 0.0;
    if (gt > LUT_N_THETA - 1.0) gt = LUT_N_THETA - 1.0;
    
    // Integer indices
    int ix0 = (int)floor(gx);
    int iy0 = (int)floor(gy);
    int it0 = (int)floor(gt);
    
    int ix1 = min(ix0 + 1, LUT_N_X - 1);
    int iy1 = min(iy0 + 1, LUT_N_Y - 1);
    int it1 = min(it0 + 1, LUT_N_THETA - 1);
    
    // Fractional parts
    double fx = gx - ix0;
    double fy = gy - iy0;
    double ft = gt - it0;
    
    // Fetch 8 corner values
    // Index: ix * (N_y * N_theta) + iy * N_theta + it
    #define LUT_IDX(ix, iy, it) ((ix) * (LUT_N_Y * LUT_N_THETA) + (iy) * LUT_N_THETA + (it))
    
    double v000 = g_lut[LUT_IDX(ix0, iy0, it0)];
    double v001 = g_lut[LUT_IDX(ix0, iy0, it1)];
    double v010 = g_lut[LUT_IDX(ix0, iy1, it0)];
    double v011 = g_lut[LUT_IDX(ix0, iy1, it1)];
    double v100 = g_lut[LUT_IDX(ix1, iy0, it0)];
    double v101 = g_lut[LUT_IDX(ix1, iy0, it1)];
    double v110 = g_lut[LUT_IDX(ix1, iy1, it0)];
    double v111 = g_lut[LUT_IDX(ix1, iy1, it1)];
    
    #undef LUT_IDX
    
    // Trilinear interpolation
    double v00 = v000 * (1.0 - ft) + v001 * ft;
    double v01 = v010 * (1.0 - ft) + v011 * ft;
    double v10 = v100 * (1.0 - ft) + v101 * ft;
    double v11 = v110 * (1.0 - ft) + v111 * ft;
    
    double v0 = v00 * (1.0 - fy) + v01 * fy;
    double v1 = v10 * (1.0 - fy) + v11 * fy;
    
    return v0 * (1.0 - fx) + v1 * fx;
#endif
}

// Compute total overlap of ref tree with all trees in xyt list using LUT
// Returns sum of overlaps; gradients not implemented yet
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double* __restrict__ xyt_Nx3,  // [n, 3] C-contiguous
    const int n,
    const int skip_index,                 // index to skip (self), use -1 to skip none
    double3* d_ref,                        // output: gradient w.r.t. ref (accumulated), can be NULL
    const int compute_grads)               // if non-zero, compute gradients (NOT IMPLEMENTED)
{
    double sum = 0.0;
    
    // Precompute sin/cos for ref orientation
    double c_ref = 0.0, s_ref = 0.0;
    sincos(ref.z, &s_ref, &c_ref);
    
    // Max distance for overlap check
    double max_dist_sq = 4.0 * MAX_RADIUS * MAX_RADIUS;
    
    for (int i = 0; i < n; ++i) {
        // Skip self
        if (i == skip_index) continue;
        
        // Read other tree pose
        double other_x = xyt_Nx3[i * 3 + 0];
        double other_y = xyt_Nx3[i * 3 + 1];
        double other_theta = xyt_Nx3[i * 3 + 2];
        
        // World-space displacement
        double dx_world = other_x - ref.x;
        double dy_world = other_y - ref.y;
        
        // Early exit: trees too far apart
        double dist_sq = dx_world * dx_world + dy_world * dy_world;
        if (dist_sq > max_dist_sq) continue;
        
        // Transform displacement into ref frame (rotate by -ref.theta)
        double dx_local = c_ref * dx_world + s_ref * dy_world;
        double dy_local = -s_ref * dx_world + c_ref * dy_world;
        
        // Relative angle
        double dtheta = wrap_angle(other_theta - ref.z);
        
        // LUT lookup (uses global g_lut or g_tex)
        double overlap = lut_lookup(dx_local, dy_local, dtheta);
        sum += overlap;
        
        // Gradients not implemented yet
        // TODO: compute gradients via trilinear derivative
    }
    
    return sum;
}

// Sum overlaps between trees in xyt1 and trees in xyt2.
// Each tree in xyt1 is compared against all trees in xyt2.
// Identical poses are automatically skipped (when xyt1 == xyt2).
// Result is divided by 2 since each pair is counted twice.
//
// Thread organization: 1 thread per tree
__device__ void overlap_list_total(
    const double* __restrict__ xyt1_Nx3,
    const int n1,
    const double* __restrict__ xyt2_Nx3,
    const int n2,
    double* __restrict__ out_total,
    double* __restrict__ out_grads)  // if non-NULL, write gradients [n1*3]
{
    int tid = threadIdx.x;
    int tree_idx = tid;
    
    if (tree_idx >= n1) return;
    
    // Read ref pose
    double3 ref;
    ref.x = xyt1_Nx3[tree_idx * 3 + 0];
    ref.y = xyt1_Nx3[tree_idx * 3 + 1];
    ref.z = xyt1_Nx3[tree_idx * 3 + 2];
    
    // Determine if we need gradients
    int compute_grads = (out_grads != NULL) ? 1 : 0;
    
    // Initialize gradient for this tree
    double3 d_ref_local = {0.0, 0.0, 0.0};
    
    // Compute overlap sum for this ref tree against all others
    // Skip self (tree_idx) when xyt1 == xyt2 (assumed same pointer)
    double local_sum = overlap_ref_with_list(
        ref, xyt2_Nx3, n2, tree_idx,
        compute_grads ? &d_ref_local : NULL, compute_grads);
    
    // Divide by 2 since each pair counted twice
    atomicAdd(out_total, local_sum / 2.0);
    
    // Accumulate gradients
    if (compute_grads) {
        atomicAdd(&out_grads[tree_idx * 3 + 0], d_ref_local.x);
        atomicAdd(&out_grads[tree_idx * 3 + 1], d_ref_local.y);
        atomicAdd(&out_grads[tree_idx * 3 + 2], d_ref_local.z);
    }
}

// Multi-ensemble kernel: one block per ensemble
// Thread organization: 1 thread per tree (not 4 like pack_cuda.py)
__global__ void multi_overlap_lut_total(
    const double* __restrict__ xyt1_base,      // [num_ensembles, n_trees, 3]
    const double* __restrict__ xyt2_base,      // [num_ensembles, n_trees, 3]
    const int n_trees,
    cudaTextureObject_t tex,                   // texture object (used in texture mode)
    const double* __restrict__ lut,            // LUT array (used in array mode)
    double* __restrict__ out_totals,           // [num_ensembles]
    double* __restrict__ out_grads_base,       // [num_ensembles, n_trees, 3] or NULL
    const int num_ensembles)
{
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) return;
    
    // First thread of first block sets up global state
    if (threadIdx.x == 0 && ensemble_id == 0) {
        g_lut = lut;
        g_tex = tex;
    }
    // All threads must wait for global state to be set
    __threadfence();
    __syncthreads();
    
    // Calculate offsets for this ensemble
    int ensemble_stride = n_trees * 3;
    const double* xyt1_ensemble = xyt1_base + ensemble_id * ensemble_stride;
    const double* xyt2_ensemble = xyt2_base + ensemble_id * ensemble_stride;
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_base != NULL) 
        ? (out_grads_base + ensemble_id * ensemble_stride) : NULL;
    
    // Initialize outputs
    if (threadIdx.x == 0) {
        *out_total = 0.0;
    }
    if (out_grads != NULL) {
        for (int idx = threadIdx.x; idx < n_trees * 3; idx += blockDim.x) {
            out_grads[idx] = 0.0;
        }
    }
    __syncthreads();
    
    // Each thread processes one tree
    overlap_list_total(xyt1_ensemble, n_trees, xyt2_ensemble, n_trees,
                       out_total, out_grads);
}

} // extern "C"
"""

# Compiled CUDA module and kernel
_raw_module: cp.RawModule | None = None
_multi_overlap_lut_kernel: cp.RawKernel | None = None

# Flag to indicate lazy initialization completed
_initialized: bool = False


def _ensure_initialized() -> None:
    """Lazy initialization hook.

    On first call, this:
    - Validates LUT arrays are set
    - Uploads LUT to device (and creates texture if USE_TEXTURE=True)
    - Compiles CUDA kernel with LUT parameters
    
    LUT_X, LUT_Y, LUT_theta, LUT_vals must be set before calling this.
    """
    global _initialized, _raw_module, _multi_overlap_lut_kernel, _lut_d
    global _texture, _texture_array
    
    if _initialized:
        return
    
    # Validate LUT arrays are set
    if LUT_X is None or LUT_Y is None or LUT_theta is None or LUT_vals is None:
        raise RuntimeError("LUT arrays must be set before calling _ensure_initialized()")
    
    print(f'init CUDA LUT (USE_TEXTURE={USE_TEXTURE})')
    
    # Validate LUT shapes
    n_x, n_y, n_theta = LUT_vals.shape
    if len(LUT_X) != n_x or len(LUT_Y) != n_y or len(LUT_theta) != n_theta:
        raise ValueError(f"LUT shape mismatch: vals={LUT_vals.shape}, "
                        f"X={len(LUT_X)}, Y={len(LUT_Y)}, theta={len(LUT_theta)}")
    
    # Upload LUT to device (array mode always needs this)
    _lut_d = cp.asarray(LUT_vals, dtype=kgs.dtype_cp)
    
    # Create texture if texture mode enabled
    if USE_TEXTURE:
        # Create 3D CUDA array for texture
        # Shape must be (depth, height, width) = (N_theta, N_y, N_x)
        lut_f32 = LUT_vals.astype(np.float32)
        # CuPy texture requires (depth, height, width) ordering
        # Our LUT_vals is (N_x, N_y, N_theta), so transpose to (N_theta, N_y, N_x)
        lut_for_tex = np.transpose(lut_f32, (2, 1, 0)).copy()
        
        # Create CUDA array
        _texture_array = cp.cuda.texture.CUDAarray(
            cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat),
            lut_for_tex.shape[2], lut_for_tex.shape[1], lut_for_tex.shape[0]
        )
        _texture_array.copy_from(lut_for_tex)
        
        # Create texture object with trilinear filtering
        res_desc = cp.cuda.texture.ResourceDescriptor(
            cp.cuda.runtime.cudaResourceTypeArray, cuArr=_texture_array
        )
        tex_desc = cp.cuda.texture.TextureDescriptor(
            addressModes=(cp.cuda.runtime.cudaAddressModeClamp,
                         cp.cuda.runtime.cudaAddressModeClamp,
                         cp.cuda.runtime.cudaAddressModeClamp),
            filterMode=cp.cuda.runtime.cudaFilterModeLinear,
            readMode=cp.cuda.runtime.cudaReadModeElementType,
            normalizedCoords=0  # Use unnormalized coordinates
        )
        _texture = cp.cuda.texture.TextureObject(res_desc, tex_desc)
        print(f"  Created 3D texture: {lut_for_tex.shape}")
    
    # Prepare CUDA source with LUT parameters
    cuda_src = _CUDA_SRC
    cuda_src = cuda_src.replace('$MAX_RADIUS$', str(MAX_RADIUS))
    cuda_src = cuda_src.replace('$LUT_N_X$', str(n_x))
    cuda_src = cuda_src.replace('$LUT_N_Y$', str(n_y))
    cuda_src = cuda_src.replace('$LUT_N_THETA$', str(n_theta))
    cuda_src = cuda_src.replace('$LUT_X_MIN$', str(float(LUT_X[0])))
    cuda_src = cuda_src.replace('$LUT_X_MAX$', str(float(LUT_X[-1])))
    cuda_src = cuda_src.replace('$LUT_Y_MIN$', str(float(LUT_Y[0])))
    cuda_src = cuda_src.replace('$LUT_Y_MAX$', str(float(LUT_Y[-1])))
    cuda_src = cuda_src.replace('$LUT_THETA_MIN$', str(float(LUT_theta[0])))
    cuda_src = cuda_src.replace('$LUT_THETA_MAX$', str(float(LUT_theta[-1])))
    cuda_src = cuda_src.replace('$USE_TEXTURE$', '1' if USE_TEXTURE else '0')
    
    # Handle float32 mode
    if kgs.USE_FLOAT32:
        cuda_src = cuda_src.replace('double', 'float')
        # Also replace double-precision math functions with float versions
        cuda_src = cuda_src.replace('sincos(', 'sincosf(')
        cuda_src = cuda_src.replace('floor(', 'floorf(')
        # Replace double-precision literals with float literals
        # Use regex to avoid breaking scientific notation (e.g., 1.0e-12)
        import re
        # Match X.Y not followed by 'e', 'E', or 'f'
        cuda_src = re.sub(r'(\d+\.\d+)(?![eEf])', r'\1f', cuda_src)
    
    # Persist source for debugging
    persist_dir = os.fspath(kgs.temp_dir)
    persist_path = os.path.join(persist_dir, 'pack_cuda_lut_saved.cu')
    with open(persist_path, 'w', encoding='utf-8') as f:
        f.write(cuda_src)
    
    # Find nvcc
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH")
    
    # Detect GPU
    device = cp.cuda.Device()
    compute_capability_str = device.compute_capability
    sm_arch = f"sm_{compute_capability_str}"
    print(f"Detected GPU compute capability: {compute_capability_str} (arch={sm_arch})")
    
    # Compile
    cubin_path = os.path.join(persist_dir, 'pack_cuda_lut.cubin')
    cmd = [
        nvcc_path,
        "-O3",
        "-use_fast_math",
        "--extra-device-vectorization",
        "--ptxas-options=-v,--warn-on-spills",
        "-lineinfo",  # Enable line info for profiling (ncu source correlation)
        f"-arch={sm_arch}",
        "-cubin", persist_path,
        "-o", cubin_path
    ]
    
    print(f"Compiling: {' '.join(cmd)}")
    proc = subprocess.run(cmd, text=True, capture_output=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{proc.stderr}")
    
    if proc.stderr:
        print(proc.stderr)
    
    # Load module
    _raw_module = cp.RawModule(path=cubin_path)
    _multi_overlap_lut_kernel = _raw_module.get_function("multi_overlap_lut_total")
    
    # Print kernel info
    kernel = _multi_overlap_lut_kernel
    print(f"Kernel multi_overlap_lut_total:")
    print(f"  Registers: {kernel.num_regs}")
    print(f"  Shared mem: {kernel.shared_size_bytes} bytes")
    print(f"  Max threads/block: {kernel.max_threads_per_block}")
    
    _initialized = True


def overlap_multi_ensemble(
    xyt1: cp.ndarray, 
    xyt2: cp.ndarray, 
    out_cost: cp.ndarray,
    out_grads: cp.ndarray | None = None,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Compute total overlap sum for multiple ensembles in parallel using LUT.

    Parameters
    ----------
    xyt1 : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays for first set of trees. Must be C-contiguous.
    xyt2 : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays for second set of trees. Must be C-contiguous.
    out_cost : cp.ndarray, shape (n_ensembles,)
        Preallocated array for output costs.
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
        NOTE: Gradient computation is not yet implemented.
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution.
    """
    _ensure_initialized()
    
    num_ensembles = xyt1.shape[0]
    n_trees = xyt1.shape[1]
    dtype = xyt1.dtype
    
    if num_ensembles == 0:
        return
    
    # Validation
    assert xyt1.ndim == 3 and xyt1.shape[2] == 3, f"xyt1 shape: {xyt1.shape}"
    assert xyt2.ndim == 3 and xyt2.shape[2] == 3, f"xyt2 shape: {xyt2.shape}"
    assert xyt1.shape == xyt2.shape, "xyt1 and xyt2 must have same shape"
    assert out_cost.shape == (num_ensembles,), f"out_cost shape: {out_cost.shape}"
    assert xyt1.flags.c_contiguous and xyt2.flags.c_contiguous
    
    if out_grads is not None:
        assert out_grads.shape == (num_ensembles, n_trees, 3)
        assert out_grads.flags.c_contiguous
    
    # Zero outputs
    out_cost[:] = 0
    if out_grads is not None:
        out_grads[:] = 0
    
    # Launch kernel: 1 block per ensemble, 1 thread per tree
    blocks = num_ensembles
    threads_per_block = n_trees
    
    out_grads_ptr = out_grads if out_grads is not None else np.intp(0)
    
    # Get texture handle (0 if not using texture mode)
    tex_handle = _texture.ptr if USE_TEXTURE and _texture is not None else np.uint64(0)
    
    _multi_overlap_lut_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt1,
            xyt2,
            np.int32(n_trees),
            np.uint64(tex_handle),
            _lut_d,
            out_cost,
            out_grads_ptr,
            np.int32(num_ensembles),
        ),
        stream=stream,
    )
