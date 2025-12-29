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

// Grid spacing (world coordinates)
#define GRID_DX ((LUT_X_MAX - LUT_X_MIN) / (LUT_N_X - 1))
#define GRID_DY ((LUT_Y_MAX - LUT_Y_MIN) / (LUT_N_Y - 1))
#define GRID_DT ((LUT_THETA_MAX - LUT_THETA_MIN) / (LUT_N_THETA - 1))

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
// Returns value only (for texture path or when gradients not needed)
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
    double result = tex3D<float>(g_tex, (float)gx, (float)gy, (float)gt);
    return result;
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

#if USE_TEXTURE
// LUT lookup with finite-difference gradient (texture mode)
// Uses 1-sided finite difference with step = GRID_SIZE/4
// Direction chosen to stay within current cell (toward cell center)
// Returns value, and if d_out != NULL, also computes gradient w.r.t. (x, y, theta)
__device__ double lut_lookup_with_grad(double x, double y, double theta, double3* d_out)
{
    // Early exit if outside LUT range - return 0 cost and gradient
    if (x < LUT_X_MIN || x > LUT_X_MAX ||
        y < LUT_Y_MIN || y > LUT_Y_MAX ||
        theta < LUT_THETA_MIN || theta > LUT_THETA_MAX) {
        if (d_out != NULL) {
            d_out->x = 0.0;
            d_out->y = 0.0;
            d_out->z = 0.0;
        }
        return 0.0;
    }
    
    // Finite difference step sizes (1/4 of grid cell)
    const double h_x = GRID_DX * 0.25;
    const double h_y = GRID_DY * 0.25;
    const double h_t = GRID_DT * 0.25;
    
    // Get base value
    double v0 = lut_lookup(x, y, theta);
    
    // Compute gradients if requested
    if (d_out != NULL) {
        // Compute fractional positions within cell to determine FD direction
        // We want to step toward cell center to stay within the cell
        double gx = (x - LUT_X_MIN) / (LUT_X_MAX - LUT_X_MIN) * (LUT_N_X - 1);
        double gy = (y - LUT_Y_MIN) / (LUT_Y_MAX - LUT_Y_MIN) * (LUT_N_Y - 1);
        double gt = (theta - LUT_THETA_MIN) / (LUT_THETA_MAX - LUT_THETA_MIN) * (LUT_N_THETA - 1);
        
        // Fractional parts (0 to 1 within cell)
        double fx = gx - floor(gx);
        double fy = gy - floor(gy);
        double ft = gt - floor(gt);
        
        // Choose step direction: if f >= 0.5, step backward; else step forward
        // This keeps us within the current cell
        double sign_x = (fx >= 0.5) ? -1.0 : 1.0;
        double sign_y = (fy >= 0.5) ? -1.0 : 1.0;
        double sign_t = (ft >= 0.5) ? -1.0 : 1.0;
        
        // One-sided finite difference: df/dx ≈ (f(x+h) - f(x)) / h  or  (f(x) - f(x-h)) / h
        double vx = lut_lookup(x + sign_x * h_x, y, theta);
        double vy = lut_lookup(x, y + sign_y * h_y, theta);
        double vt = lut_lookup(x, y, theta + sign_t * h_t);
        
        d_out->x = (vx - v0) / (sign_x * h_x);
        d_out->y = (vy - v0) / (sign_y * h_y);
        d_out->z = (vt - v0) / (sign_t * h_t);
    }
    
    return v0;
}
#endif

#if !USE_TEXTURE
// LUT lookup with analytical gradient (array mode only)
// Returns value, and if d_out != NULL, also computes gradient w.r.t. (x, y, theta)
__device__ double lut_lookup_with_grad(double x, double y, double theta, double3* d_out)
{
    // Early exit if outside LUT range - return 0 cost and gradient
    if (x < LUT_X_MIN || x > LUT_X_MAX ||
        y < LUT_Y_MIN || y > LUT_Y_MAX ||
        theta < LUT_THETA_MIN || theta > LUT_THETA_MAX) {
        if (d_out != NULL) {
            d_out->x = 0.0;
            d_out->y = 0.0;
            d_out->z = 0.0;
        }
        return 0.0;
    }
    
    // Normalize to grid coordinates [0, N-1]
    double gx = (x - LUT_X_MIN) / (LUT_X_MAX - LUT_X_MIN) * (LUT_N_X - 1);
    double gy = (y - LUT_Y_MIN) / (LUT_Y_MAX - LUT_Y_MIN) * (LUT_N_Y - 1);
    double gt = (theta - LUT_THETA_MIN) / (LUT_THETA_MAX - LUT_THETA_MIN) * (LUT_N_THETA - 1);
    
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
        double scale_x = (double)(LUT_N_X - 1) / (LUT_X_MAX - LUT_X_MIN);
        double scale_y = (double)(LUT_N_Y - 1) / (LUT_Y_MAX - LUT_Y_MIN);
        double scale_t = (double)(LUT_N_THETA - 1) / (LUT_THETA_MAX - LUT_THETA_MIN);
        
        d_out->x = df_dfx * scale_x;
        d_out->y = df_dfy * scale_y;
        d_out->z = df_dft * scale_t;
    }
    
    return value;
}
#endif

// Compute total overlap of ref tree with all trees in xyt list using LUT
// Returns sum of overlaps
// If compute_grads is non-zero, accumulates gradients:
//   - d_ref: gradient w.r.t. ref pose (accumulated directly)
//   - out_grads: gradient w.r.t. all other poses (accumulated via atomicAdd)
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double3* __restrict__ s_trees,  // shared memory: [n] tree poses
    const int n,
    const int skip_index,                 // index to skip (self), use -1 to skip none
    double3* d_ref,                        // output: gradient w.r.t. ref (accumulated), can be NULL
    double* __restrict__ out_grads,        // output: gradients for all trees [n*3], can be NULL
    const int compute_grads)               // if non-zero, compute gradients
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
        if (compute_grads && d_ref != NULL && out_grads != NULL) {
            // Get value and gradient w.r.t. local coords
            double3 d_local;  // gradient w.r.t. (dx_local, dy_local, dtheta)
            double overlap = lut_lookup_with_grad(dx_local, dy_local, dtheta, &d_local);
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
            
            d_ref->x += -d_local.x * c_ref + d_local.y * s_ref;
            d_ref->y += -d_local.x * s_ref - d_local.y * c_ref;
            d_ref->z += d_local.x * dy_local - d_local.y * dx_local - d_local.z;
            
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
            atomicAdd(&out_grads[i * 3 + 0], d_other_x/2.0);
            atomicAdd(&out_grads[i * 3 + 1], d_other_y/2.0);
            atomicAdd(&out_grads[i * 3 + 2], d_other_theta/2.0);
        } else {
            double overlap = lut_lookup(dx_local, dy_local, dtheta);
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
    
    // Initialize gradient for this tree
    double3 d_ref_local = {0.0, 0.0, 0.0};
    
    // Compute overlap sum for this ref tree against all others
    // Skip self (tree_idx)
    // This also accumulates gradients to both ref (via d_ref_local) and
    // to other trees (via atomicAdd to out_grads)
    double local_sum = overlap_ref_with_list(
        ref, s_trees, n, tree_idx,
        compute_grads ? &d_ref_local : NULL,
        out_grads,
        compute_grads);
    
    // Divide by 2 since each pair counted twice
    atomicAdd(out_total, local_sum / 2.0);
    
    // Write ref's gradient using atomicAdd (other threads may also contribute)
    if (compute_grads && out_grads != NULL) {
        atomicAdd(&out_grads[tree_idx * 3 + 0], d_ref_local.x/2.0);
        atomicAdd(&out_grads[tree_idx * 3 + 1], d_ref_local.y/2.0);
        atomicAdd(&out_grads[tree_idx * 3 + 2], d_ref_local.z/2.0);
    }
}

// Multi-ensemble kernel: one block per ensemble
// Thread organization: 1 thread per tree (not 4 like pack_cuda.py)
// Uses dynamic shared memory for tree poses: extern __shared__ double3 s_trees[]
__global__ void multi_overlap_lut_total(
    const double* __restrict__ xyt_base,       // [num_ensembles, n_trees, 3]
    const int n_trees,
    cudaTextureObject_t tex,                   // texture object (used in texture mode)
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
        g_tex = tex;
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
    xyt: cp.ndarray, 
    out_cost: cp.ndarray,
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
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
        Gradients are computed analytically (non-texture mode only).
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution.
    """
    _ensure_initialized()
    
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
    
    # Get texture handle (0 if not using texture mode)
    tex_handle = _texture.ptr if USE_TEXTURE and _texture is not None else np.uint64(0)
    
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
            np.uint64(tex_handle),
            _lut_d,
            out_cost,
            out_grads_ptr,
            np.int32(num_ensembles),
        ),
        stream=stream,
        shared_mem=shared_mem_bytes,
    )
