"""
LUT-based overlap computation for tree packing.

Uses a precomputed 3D lookup table indexed by relative pose (dx, dy, dtheta)
to compute overlap between tree pairs, replacing expensive polygon intersection.

Two modes are available:
- Array mode (USE_TEXTURE=False): Manual trilinear interpolation from global memory.
  Uses full float32/float64 precision for interpolation weights.
  Gradients computed analytically from trilinear interpolation formula.
- Texture mode (USE_TEXTURE=True): Hardware trilinear interpolation via 3D texture.
  Faster but uses 9-bit fixed-point for interpolation weights (~0.2% precision).
  Gradients computed via 1-sided finite differences (step = grid_size/4).
  Results will differ slightly from array mode due to hardware precision limits.
  (REMOVED FOR NOW)
"""
from __future__ import annotations

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Optional

import kaggle_support as kgs
import os
import subprocess
import shutil
import math

MAX_RADIUS = kgs.tree_max_radius


@dataclass
class LookupTable:
    """Container for 3D lookup table data with precomputed utilities.

    Stores the grid coordinates and values for trilinear interpolation,
    along with precomputed grid parameters to avoid repeated calculations
    in kernels.

    Attributes
    ----------
    X : np.ndarray, shape (N_x,)
        X coordinates of grid points (relative displacement)
    Y : np.ndarray, shape (N_y,)
        Y coordinates of grid points (relative displacement)
    theta : np.ndarray, shape (N_theta,)
        Theta coordinates of grid points (relative rotation, radians)
    vals : np.ndarray, shape (N_x, N_y, N_theta)
        Overlap values at grid points
    """

    X: np.ndarray
    Y: np.ndarray
    theta: np.ndarray
    vals: np.ndarray
    apply_quadratic_transform: bool = False  # If True, apply max(0, val)^2 per-pair

    # GPU resources (created in __post_init__)
    lut_d: cp.ndarray = None

    def __post_init__(self):
        """Validate shapes and precompute grid parameters + GPU resources."""
        n_x, n_y, n_theta = self.vals.shape
        if len(self.X) != n_x or len(self.Y) != n_y or len(self.theta) != n_theta:
            raise ValueError(f"LUT shape mismatch: vals={self.vals.shape}, "
                           f"X={len(self.X)}, Y={len(self.Y)}, theta={len(self.theta)}")

        # Precompute grid parameters (used in kernel via constant memory)
        self.N_x = n_x
        self.N_y = n_y
        self.N_theta = n_theta

        self.X_min = float(self.X[0])
        self.X_max = float(self.X[-1])
        self.Y_min = float(self.Y[0])
        self.Y_max = float(self.Y[-1])
        self.theta_min = float(self.theta[0])
        self.theta_max = float(self.theta[-1])

        # Grid spacing (world coordinates)
        self.grid_dx = (self.X_max - self.X_min) / (self.N_x - 1) if self.N_x > 1 else 0.0
        self.grid_dy = (self.Y_max - self.Y_min) / (self.N_y - 1) if self.N_y > 1 else 0.0
        self.grid_dtheta = (self.theta_max - self.theta_min) / (self.N_theta - 1) if self.N_theta > 1 else 0.0

        # Create GPU resources immediately
        self._prepare_gpu_resources()

    def _prepare_gpu_resources(self):
        """Prepare device arrays and textures for GPU use."""
        # Only create device array if not using texture
        self.lut_d = cp.asarray(self.vals, dtype=kgs.dtype_cp)

    @classmethod
    def build_from_function(
        cls,
        eval_fn,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        N_x: int = 400,
        N_y: int = 400,
        N_theta: int = 400,
        trim_zeros: bool = True,
        verbose: bool = True
    ) -> 'LookupTable':
        """Build lookup table by evaluating a function on grid.

        Parameters
        ----------
        eval_fn : callable
            Function with signature: eval_fn(dx: np.ndarray, dy: np.ndarray, theta: float) -> np.ndarray
            - dx, dy: (N,) arrays of relative positions
            - theta: scalar rotation angle
            - Returns: (N,) array of cost values
        X, Y, theta : np.ndarray, optional
            Grid coordinates. If None, reasonable defaults based on tree size.
        N_x, N_y, N_theta : int
            Grid dimensions (used if X/Y/theta not provided)
        trim_zeros : bool
            If True, remove edge rows/columns that are all zeros
            (keeping at least one zero row/column for proper interpolation)
        verbose : bool
            If True, print progress messages

        Returns
        -------
        LookupTable
            Built lookup table
        """
        # Set default grids if not provided
        if X is None:
            X = np.linspace(-2*MAX_RADIUS, 2*MAX_RADIUS, N_x, dtype=np.float32)
            X = X[N_x//2-1:]  # Use only non-negative X due to symmetry
            assert(X[0]<0) # Take one negative one to avoid edge issues
        if Y is None:
            Y = np.linspace(-2*MAX_RADIUS, 2*MAX_RADIUS, N_y, dtype=np.float32)
        if theta is None:
            theta = np.linspace(-np.pi, np.pi, N_theta, dtype=np.float32)

        N_x, N_y, N_theta = len(X), len(Y), len(theta)

        if verbose:
            print(f"Building LUT: {N_x} x {N_y} x {N_theta} = {N_x*N_y*N_theta:,} grid points")

        # Build LUT by looping over theta values to save memory
        vals = np.zeros((N_x, N_y, N_theta), dtype=np.float32)

        for t_idx, t_val in enumerate(theta):
            if verbose and t_idx % 50 == 0:
                print(f"  Processing theta {t_idx+1}/{N_theta}")

            # Create meshgrid for this theta slice
            dx_grid, dy_grid = np.meshgrid(X, Y, indexing='ij')
            
            # Flatten to 1D arrays
            dx_flat = dx_grid.ravel()
            dy_flat = dy_grid.ravel()
            
            # Call the evaluation function
            costs = eval_fn(dx_flat, dy_flat, t_val)
            
            # Store in LUT (reshape from flat to (N_x, N_y))
            vals[:, :, t_idx] = costs.reshape(N_x, N_y)

        if verbose:
            print(f"Cost range: [{vals.min():.6f}, {vals.max():.6f}]")

        # Trim zeros if requested
        if trim_zeros:
            X, Y, theta, vals = cls._trim_zero_edges(X, Y, theta, vals, verbose=verbose)

        return cls(X=X, Y=Y, theta=theta, vals=vals)

    @staticmethod
    def _trim_zero_edges(
        X: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        vals: np.ndarray,
        verbose: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trim edge rows/columns that are all zeros.

        Removes leading/trailing rows and columns along X and Y axes that
        contain only zeros, while ensuring at least one zero row/column
        remains for proper interpolation at the grid boundaries.

        Parameters
        ----------
        X, Y, theta : np.ndarray
            Grid coordinates
        vals : np.ndarray, shape (N_x, N_y, N_theta)
            Table values
        verbose : bool
            Print trimming info

        Returns
        -------
        X_trim, Y_trim, theta_trim, vals_trim
            Trimmed arrays
        """
        N_x, N_y, N_theta = vals.shape

        # Find non-zero ranges along X axis (collapse over Y and theta)
        x_sums = np.sum(np.abs(vals), axis=(1, 2))  # (N_x,)
        x_nonzero = np.where(x_sums > 0)[0]

        if len(x_nonzero) > 0:
            x_start = max(0, x_nonzero[0] - 1)  # Keep one zero row before
            x_end = min(N_x - 1, x_nonzero[-1] + 1)  # Keep one zero row after
        else:
            # All zeros - keep everything
            x_start, x_end = 0, N_x - 1

        # Find non-zero ranges along Y axis (collapse over X and theta)
        y_sums = np.sum(np.abs(vals), axis=(0, 2))  # (N_y,)
        y_nonzero = np.where(y_sums > 0)[0]

        if len(y_nonzero) > 0:
            y_start = max(0, y_nonzero[0] - 1)
            y_end = min(N_y - 1, y_nonzero[-1] + 1)
        else:
            y_start, y_end = 0, N_y - 1

        # Theta is typically non-zero throughout, but check anyway
        theta_sums = np.sum(np.abs(vals), axis=(0, 1))  # (N_theta,)
        theta_nonzero = np.where(theta_sums > 0)[0]

        if len(theta_nonzero) > 0:
            theta_start = theta_nonzero[0]  # Don't keep extra zeros for theta
            theta_end = theta_nonzero[-1]
        else:
            theta_start, theta_end = 0, N_theta - 1

        # Trim
        X_trim = X[x_start:x_end+1]
        Y_trim = Y[y_start:y_end+1]
        theta_trim = theta[theta_start:theta_end+1]
        vals_trim = vals[x_start:x_end+1, y_start:y_end+1, theta_start:theta_end+1]

        if verbose:
            removed_x = N_x - len(X_trim)
            removed_y = N_y - len(Y_trim)
            removed_theta = N_theta - len(theta_trim)
            total_removed = (N_x * N_y * N_theta) - (len(X_trim) * len(Y_trim) * len(theta_trim))
            pct = 100 * total_removed / (N_x * N_y * N_theta)

            print(f"Trimming zero edges:")
            print(f"  X: {N_x} -> {len(X_trim)} (removed {removed_x})")
            print(f"  Y: {N_y} -> {len(Y_trim)} (removed {removed_y})")
            print(f"  Theta: {N_theta} -> {len(theta_trim)} (removed {removed_theta})")
            print(f"  Total reduction: {pct:.1f}% ({total_removed:,} points)")

        return X_trim, Y_trim, theta_trim, vals_trim


# Cache last LUT to avoid redundant constant memory updates
_last_lut_id: int | None = None

_CUDA_SRC = r"""
extern "C" {

#define MAX_RADIUS $MAX_RADIUS$
#define M_PI 3.14159265358979323846

// LUT grid parameters in constant memory (can be updated without recompilation)
__constant__ int c_N_x;
__constant__ int c_N_y;
__constant__ int c_N_theta;
__constant__ double c_X_min;
__constant__ double c_X_max;
__constant__ double c_Y_min;
__constant__ double c_Y_max;
__constant__ double c_theta_min;
__constant__ double c_theta_max;

// Computed grid spacing (updated when constants are set)
__constant__ double c_grid_dx;
__constant__ double c_grid_dy;
__constant__ double c_grid_dtheta;

// Transform flag (updated when LUT changes)
__constant__ int c_apply_quadratic_transform;  // If non-zero, apply max(0, val)^2 per-pair

// Global device state - set once by the kernel, used by all device functions
__device__ const double* g_lut;           // LUT array pointer (array mode)
__device__ cudaTextureObject_t g_tex;     // texture object (texture mode)

// Wrap angle to [-pi, pi]
__device__ __forceinline__ double wrap_angle(double theta) {
    while (theta > M_PI) theta -= 2.0 * M_PI;
    while (theta < -M_PI) theta += 2.0 * M_PI;
    return theta;
}


// LUT lookup with analytical gradient (array mode only)
// Returns value, and if d_out != NULL, also computes gradient w.r.t. (x, y, theta)
__device__ double lut_lookup_with_grad(double x, double y, double theta, double3* d_out)
{
    // Early exit if outside LUT range - return 0 cost and gradient
    if (x < -c_X_max || x > c_X_max ||
        y < c_Y_min || y > c_Y_max ||
        theta < c_theta_min || theta > c_theta_max) {
        if (d_out != NULL) {
            d_out->x = 0.0;
            d_out->y = 0.0;
            d_out->z = 0.0;
        }
        return 0.0;
    }

    double x_t;
    double theta_t;
    if (x >= 0.) {
        x_t = x;
        theta_t = theta;
    } else {
        x_t = -x;
        theta_t = - theta;
    }
    
    // Normalize to grid coordinates [0, N-1]
    double gx = (x_t - c_X_min) / (c_X_max - c_X_min) * (c_N_x - 1);
    double gy = (y - c_Y_min) / (c_Y_max - c_Y_min) * (c_N_y - 1);
    double gt = (theta_t - c_theta_min) / (c_theta_max - c_theta_min) * (c_N_theta - 1);
    
    // Integer indices
    int ix0 = (int)floor(gx);
    int iy0 = (int)floor(gy);
    int it0 = (int)floor(gt);
    
    int ix1 = min(ix0 + 1, c_N_x - 1);
    int iy1 = min(iy0 + 1, c_N_y - 1);
    int it1 = min(it0 + 1, c_N_theta - 1);
    
    // Fractional parts
    double fx = gx - ix0;
    double fy = gy - iy0;
    double ft = gt - it0;
    
    // Fetch 8 corner values
    #define LUT_IDX(ix, iy, it) ((ix) * (c_N_y * c_N_theta) + (iy) * c_N_theta + (it))
    
    double v000 = g_lut[LUT_IDX(ix0, iy0, it0)];
    double v001 = g_lut[LUT_IDX(ix0, iy0, it1)];
    double v010 = g_lut[LUT_IDX(ix0, iy1, it0)];
    double v011 = g_lut[LUT_IDX(ix0, iy1, it1)];
    double v100 = g_lut[LUT_IDX(ix1, iy0, it0)];
    double v101 = g_lut[LUT_IDX(ix1, iy0, it1)];
    double v110 = g_lut[LUT_IDX(ix1, iy1, it0)];
    double v111 = g_lut[LUT_IDX(ix1, iy1, it1)];
    
    #undef LUT_IDX
    
    // Trilinear interpolation for value
    double v00 = v000 * (1.0 - ft) + v001 * ft;
    double v01 = v010 * (1.0 - ft) + v011 * ft;
    double v10 = v100 * (1.0 - ft) + v101 * ft;
    double v11 = v110 * (1.0 - ft) + v111 * ft;
    
    double v0 = v00 * (1.0 - fy) + v01 * fy;
    double v1 = v10 * (1.0 - fy) + v11 * fy;
    
    double value = v0 * (1.0 - fx) + v1 * fx;
    
    // Compute gradients if requested
    if (d_out != NULL) {
        // Gradient w.r.t. fractional coordinates (fx, fy, ft)
        // df/dfx = v1 - v0 = bilinear(v1**) - bilinear(v0**)
        double df_dfx = v1 - v0;
        
        // df/dfy at current fx:
        // = (1-fx) * (v01 - v00) + fx * (v11 - v10)
        double df_dfy = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10);
        
        // df/dft at current fx, fy:
        // Need to compute partial w.r.t. ft through the trilinear formula
        // v00 = v000*(1-ft) + v001*ft  => dv00/dft = v001 - v000
        double dv00_dft = v001 - v000;
        double dv01_dft = v011 - v010;
        double dv10_dft = v101 - v100;
        double dv11_dft = v111 - v110;
        
        double dv0_dft = dv00_dft * (1.0 - fy) + dv01_dft * fy;
        double dv1_dft = dv10_dft * (1.0 - fy) + dv11_dft * fy;
        double df_dft = dv0_dft * (1.0 - fx) + dv1_dft * fx;
        
        // Convert to world coordinates
        // gx = (x - X_MIN) / (X_MAX - X_MIN) * (N_X - 1)
        // dgx/dx = (N_X - 1) / (X_MAX - X_MIN)
        // df/dx = df/dfx * dfx/dgx * dgx/dx = df/dfx * 1 * dgx/dx
        double scale_x = (double)(c_N_x - 1) / (c_X_max - c_X_min);
        double scale_y = (double)(c_N_y - 1) / (c_Y_max - c_Y_min);
        double scale_t = (double)(c_N_theta - 1) / (c_theta_max - c_theta_min);
        
        double grad_x = df_dfx * scale_x;
        double grad_y = df_dfy * scale_y;
        double grad_z = df_dft * scale_t;
        
        // Apply symmetry transform to gradients
        // If x < 0, we used (x_t = -x, theta_t = -theta)
        // Chain rule: df/dx = df/dx_t * dx_t/dx = df/dx_t * (-1)
        //             df/dtheta = df/dtheta_t * dtheta_t/dtheta = df/dtheta_t * (-1)
        if (x < 0.0) {
            grad_x = -grad_x;
            grad_z = -grad_z;
        }
        
        d_out->x = grad_x;
        d_out->y = grad_y;
        d_out->z = grad_z;
    }
    
    return value;
}

// Compute total overlap of ref tree with all trees in xyt list using LUT
// Returns sum of overlaps
// If compute_grads is non-zero, accumulates gradients to out_grads via atomicAdd for all trees
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double3* __restrict__ s_trees,  // shared memory: [n] tree poses
    const int n,
    const int ref_index,                  // index of ref tree (skips self, accumulates gradients here)
    double* __restrict__ out_grads,        // output: gradients for all trees [n*3], can be NULL
    const int compute_grads)               // if non-zero, compute gradients
{
    double sum = 0.0;
    
    // Precompute sin/cos for ref orientation
    double c_ref = 0.0, s_ref = 0.0;
    sincos(ref.z, &s_ref, &c_ref);
    
    // Max distance for overlap check
    double max_dist_sq = 4.0 * MAX_RADIUS * MAX_RADIUS;
    
    for (int i = ref_index; i < n; ++i) {
        // Skip self
        if (i == ref_index) continue;
        
        // Read other tree pose from shared memory
        double3 other = s_trees[i];
        double other_x = other.x;
        double other_y = other.y;
        double other_theta = other.z;
        
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
        
        // Compute value and optionally gradients (both texture and array paths use lut_lookup_with_grad)
        if (compute_grads && out_grads != NULL) {
            // Get value and gradient w.r.t. local coords
            double3 d_local;  // gradient w.r.t. (dx_local, dy_local, dtheta)
            double overlap = lut_lookup_with_grad(dx_local, dy_local, dtheta, &d_local);
            
            // Apply quadratic transform if enabled: overlap_transformed = max(0, overlap)^2
            if (c_apply_quadratic_transform) {
                if (overlap > 0.0) {
                    // Transform gradients: d[overlap^2]/dx = 2*overlap * d[overlap]/dx
                    d_local.x *= 2.0 * overlap;
                    d_local.y *= 2.0 * overlap;
                    d_local.z *= 2.0 * overlap;
                    overlap = overlap * overlap;
                } else {
                    overlap = 0.0;
                    d_local.x = 0.0;
                    d_local.y = 0.0;
                    d_local.z = 0.0;
                }
            }
            
            sum += overlap;
            
            // ===== Gradient w.r.t. ref =====
            // dx_world = other_x - ref.x => d(dx_world)/d(ref.x) = -1
            // dy_world = other_y - ref.y => d(dy_world)/d(ref.y) = -1
            // dx_local = c_ref * dx_world + s_ref * dy_world
            // dy_local = -s_ref * dx_world + c_ref * dy_world
            //
            // d(overlap)/d(ref.x) = d_local.x * c_ref * (-1) + d_local.y * (-s_ref) * (-1)
            //                     = -d_local.x * c_ref + d_local.y * s_ref
            // d(overlap)/d(ref.y) = d_local.x * s_ref * (-1) + d_local.y * c_ref * (-1)
            //                     = -d_local.x * s_ref - d_local.y * c_ref
            // d(overlap)/d(ref.theta):
            //   d(dx_local)/d(ref.theta) = -s_ref * dx_world + c_ref * dy_world = dy_local
            //   d(dy_local)/d(ref.theta) = -c_ref * dx_world - s_ref * dy_world = -dx_local
            //   d(dtheta)/d(ref.theta) = -1
            //   => d_local.x * dy_local - d_local.y * dx_local - d_local.z
            
            double d_ref_x = -d_local.x * c_ref + d_local.y * s_ref;
            double d_ref_y = -d_local.x * s_ref - d_local.y * c_ref;
            double d_ref_z = d_local.x * dy_local - d_local.y * dx_local - d_local.z;
            
            // Accumulate to ref's gradient using atomicAdd
            atomicAdd(&out_grads[ref_index * 3 + 0], d_ref_x);
            atomicAdd(&out_grads[ref_index * 3 + 1], d_ref_y);
            atomicAdd(&out_grads[ref_index * 3 + 2], d_ref_z);
            
            // ===== Gradient w.r.t. other =====
            // dx_world = other_x - ref.x => d(dx_world)/d(other.x) = +1
            // dy_world = other_y - ref.y => d(dy_world)/d(other.y) = +1
            // dtheta = other_theta - ref.z => d(dtheta)/d(other.theta) = +1
            //
            // d(overlap)/d(other.x) = d_local.x * c_ref * (+1) + d_local.y * (-s_ref) * (+1)
            //                       = d_local.x * c_ref - d_local.y * s_ref
            // d(overlap)/d(other.y) = d_local.x * s_ref * (+1) + d_local.y * c_ref * (+1)
            //                       = d_local.x * s_ref + d_local.y * c_ref
            // d(overlap)/d(other.theta) = d_local.z
            
            double d_other_x = d_local.x * c_ref - d_local.y * s_ref;
            double d_other_y = d_local.x * s_ref + d_local.y * c_ref;
            double d_other_theta = d_local.z;
            
            // Accumulate to other's gradient using atomicAdd
            atomicAdd(&out_grads[i * 3 + 0], d_other_x);
            atomicAdd(&out_grads[i * 3 + 1], d_other_y);
            atomicAdd(&out_grads[i * 3 + 2], d_other_theta);
        } else {
            double overlap = lut_lookup_with_grad(dx_local, dy_local, dtheta, NULL);
            
            // Apply quadratic transform if enabled: overlap_transformed = max(0, overlap)^2
            if (c_apply_quadratic_transform) {
                if (overlap > 0.0) {
                    overlap = overlap * overlap;
                } else {
                    overlap = 0.0;
                }
            }
            
            sum += overlap;
        }
    }
    
    return sum;
}

// Sum overlaps between all tree pairs in xyt.
// Each tree is compared against all other trees.
// Self-overlap is skipped. Result is divided by 2 since each pair is counted twice.
//
// Thread organization: 1 thread per tree
// s_trees: shared memory array of size n (passed from kernel)
__device__ void overlap_list_total(
    const double* __restrict__ xyt_Nx3,
    double3* __restrict__ s_trees,  // shared memory for tree poses
    const int n,
    double* __restrict__ out_total,
    double* __restrict__ out_grads)  // if non-NULL, write gradients [n*3]
{
    int tid = threadIdx.x;
    
    // Cooperatively load all trees into shared memory (coalesced reads)
    for (int i = tid; i < n; i += blockDim.x) {
        s_trees[i].x = xyt_Nx3[i * 3 + 0];
        s_trees[i].y = xyt_Nx3[i * 3 + 1];
        s_trees[i].z = xyt_Nx3[i * 3 + 2];
    }
    __syncthreads();
    
    int tree_idx = tid;
    if (tree_idx >= n) return;
    
    // Read ref pose from shared memory
    double3 ref = s_trees[tree_idx];
    
    // Determine if we need gradients
    int compute_grads = (out_grads != NULL) ? 1 : 0;
    
    // Compute overlap sum for this ref tree against all others
    // Skip self (tree_idx)
    // This also accumulates gradients to all trees (including ref) via atomicAdd to out_grads
    double local_sum = overlap_ref_with_list(
        ref, s_trees, n, tree_idx,
        out_grads,
        compute_grads);
    
    // Divide by 2 since each pair counted twice
    atomicAdd(out_total, local_sum);
}

// Multi-ensemble kernel: one block per ensemble
// Thread organization: 1 thread per tree (not 4 like pack_cuda.py)
// Uses dynamic shared memory for tree poses: extern __shared__ double3 s_trees[]
__global__ void multi_overlap_lut_total(
    const double* __restrict__ xyt_base,       // [num_ensembles, n_trees, 3]
    const int n_trees,
    const double* __restrict__ lut,            // LUT array (used in array mode)
    double* __restrict__ out_totals,           // [num_ensembles]
    double* __restrict__ out_grads_base,       // [num_ensembles, n_trees, 3] or NULL
    const int num_ensembles)
{
    // Dynamic shared memory for tree poses
    extern __shared__ double3 s_trees[];
    
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) return;
    
    // First thread of first block sets up global state
    if (threadIdx.x == 0 && ensemble_id == 0) {
        g_lut = lut;
    }
    // All threads must wait for global state to be set
    __threadfence();
    __syncthreads();
    
    // Calculate offsets for this ensemble
    int ensemble_stride = n_trees * 3;
    const double* xyt_ensemble = xyt_base + ensemble_id * ensemble_stride;
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_base != NULL) 
        ? (out_grads_base + ensemble_id * ensemble_stride) : NULL;
    
    // Initialize outputs
    if (threadIdx.x == 0) {
        *out_total = 0.0;
    }
    // Zero gradients (needed since we use atomicAdd)
    if (out_grads != NULL) {
        for (int idx = threadIdx.x; idx < n_trees * 3; idx += blockDim.x) {
            out_grads[idx] = 0.0;
        }
    }
    __syncthreads();
    
    // Each thread processes one tree (shared mem loading happens inside)
    overlap_list_total(xyt_ensemble, s_trees, n_trees, out_total, out_grads);
}

} // extern "C"
"""

# Compiled CUDA module and kernel
_raw_module: cp.RawModule | None = None
_multi_overlap_lut_kernel: cp.RawKernel | None = None

# Flag to indicate lazy initialization completed
_initialized: bool = False


def _update_lut_constants(lut: LookupTable) -> None:
    """Update constant memory with LUT grid parameters (fast, no recompilation).

    Parameters
    ----------
    lut : LookupTable
        Lookup table with grid parameters to copy to device constant memory
    """
    global _raw_module

    if _raw_module is None:
        raise RuntimeError("Kernel not compiled yet - call _ensure_initialized() first")

    # Use proper type for constant memory (int32 for ints, float64/float32 for doubles)
    dtype = np.float32 if kgs.USE_FLOAT32 else np.float64

    # Helper to copy data to constant memory symbol
    def set_constant(name, value, dtype_val):
        ptr = _raw_module.get_global(name)
        src = cp.array([value], dtype=dtype_val)
        # Copy from device to device (constant memory is device memory)
        ptr.copy_from_device(src.data, src.nbytes)

    # Update grid dimensions
    set_constant('c_N_x', lut.N_x, np.int32)
    set_constant('c_N_y', lut.N_y, np.int32)
    set_constant('c_N_theta', lut.N_theta, np.int32)

    # Update grid bounds
    set_constant('c_X_min', lut.X_min, dtype)
    set_constant('c_X_max', lut.X_max, dtype)
    set_constant('c_Y_min', lut.Y_min, dtype)
    set_constant('c_Y_max', lut.Y_max, dtype)
    set_constant('c_theta_min', lut.theta_min, dtype)
    set_constant('c_theta_max', lut.theta_max, dtype)

    # Update grid spacing
    set_constant('c_grid_dx', lut.grid_dx, dtype)
    set_constant('c_grid_dy', lut.grid_dy, dtype)
    set_constant('c_grid_dtheta', lut.grid_dtheta, dtype)
    
    # Update transform flag
    set_constant('c_apply_quadratic_transform', 1 if lut.apply_quadratic_transform else 0, np.int32)


def _ensure_initialized() -> None:
    """Lazy initialization hook - compiles kernel once.

    On first call, this compiles the CUDA kernel with USE_TEXTURE setting.
    Grid parameters are set dynamically via constant memory, so no recompilation
    needed when switching between LUTs.
    """
    global _initialized, _raw_module, _multi_overlap_lut_kernel

    if _initialized:
        return

    print(f'Compiling CUDA LUT kernel one-time only)')

    # Prepare CUDA source (only MAX_RADIUS and USE_TEXTURE are compile-time constants)
    cuda_src = _CUDA_SRC
    cuda_src = cuda_src.replace('$MAX_RADIUS$', str(MAX_RADIUS))
    
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
    xyt: cp.ndarray,
    out_cost: cp.ndarray,
    lut: LookupTable,
    out_grads: cp.ndarray | None = None,
    stream: cp.cuda.Stream | None = None
) -> None:
    """Compute total overlap sum for multiple ensembles in parallel using LUT.

    Parameters
    ----------
    xyt : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays for trees. Must be C-contiguous.
    out_cost : cp.ndarray, shape (n_ensembles,)
        Preallocated array for output costs.
    lut : LookupTable
        Lookup table to use for overlap computation.
        Must have GPU resources already prepared (done automatically in __post_init__).
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
        Gradients are computed analytically (non-texture mode only).
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution.
    """
    global _last_lut_id

    _ensure_initialized()

    # Update constant memory only if LUT changed (cache by object ID)
    lut_id = id(lut)
    if _last_lut_id != lut_id:
        _update_lut_constants(lut)
        _last_lut_id = lut_id

    num_ensembles = xyt.shape[0]
    n_trees = xyt.shape[1]

    if num_ensembles == 0:
        return

    # Validation
    assert xyt.ndim == 3 and xyt.shape[2] == 3, f"xyt shape: {xyt.shape}"
    assert out_cost.shape == (num_ensembles,), f"out_cost shape: {out_cost.shape}"
    assert xyt.flags.c_contiguous

    if out_grads is not None:
        assert out_grads.shape == (num_ensembles, n_trees, 3)
        assert out_grads.flags.c_contiguous

    # Zero cost output (grads are written directly per-tree, no need to zero)
    out_cost[:] = 0

    # Launch kernel: 1 block per ensemble, 1 thread per tree
    blocks = num_ensembles
    threads_per_block = n_trees

    out_grads_ptr = out_grads if out_grads is not None else np.intp(0)

    # Get texture and LUT from passed table
    lut_d = lut.lut_d

    # Dynamic shared memory: n_trees * sizeof(double3) or float3
    # double3 = 24 bytes, float3 = 12 bytes
    elem_size = 12 if kgs.USE_FLOAT32 else 24
    shared_mem_bytes = n_trees * elem_size

    _multi_overlap_lut_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt,
            np.int32(n_trees),
            lut_d,
            out_cost,
            out_grads_ptr,
            np.int32(num_ensembles),
        ),
        stream=stream,
        shared_mem=shared_mem_bytes,
    )
