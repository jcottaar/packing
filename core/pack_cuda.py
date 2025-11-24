"""
gpu_overlap.py

Compute overlap area between TWO IDENTICAL non-convex polygons using CUDA via CuPy.

You provide a convex decomposition of your polygon as 4 convex Shapely Polygons.
The module uploads those convex pieces to the GPU and exposes

    compute_overlap_area_gpu(x1, y1, theta1, x2, y2, theta2)

which returns the overlap area between pose-1 and pose-2, computed on the GPU.

All inputs (x1, y1, theta1, x2, y2, theta2) are expected to be **scalars**
(Python floats or NumPy scalars). The function returns a single CuPy scalar
(cp.float64).
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import kaggle_support as kgs
import pack_cuda_primitives
import os
import subprocess
import shutil

# ---------------------------------------------------------------------------
# 1. User section: insert your convex decomposition here
# ---------------------------------------------------------------------------

# Replace the placeholders with YOUR 4 convex Shapely polygons.
# They should form a disjoint partition of your tree polygon
# (they may share edges/vertices, but must not overlap in area).
#
# Example pattern:
#
# from shapely.geometry import Polygon
# CONVEX_PIECES = [
#     Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]),
#     Polygon([...]),
#     Polygon([...]),
#     Polygon([...]),
# ]
#
# All polygons will be re-oriented to CCW automatically.

CONVEX_PIECES: list[Polygon] = kgs.convex_breakdown
MAX_RADIUS = kgs.tree_max_radius

# Global variable to control floating point precision
# Set this to 'float32' or 'float64' BEFORE calling any functions
# Default is 'float64' for maximum precision
USE_FLOAT32 = False  # Set to True to use float32 instead of float64


# ---------------------------------------------------------------------------
# 2. Internal constants and CUDA code
# ---------------------------------------------------------------------------

# Maximum number of convex pieces and vertices per piece that the CUDA kernel
# is written to handle. This matches your use case: 4 convex quads.
MAX_PIECES = 4
MAX_VERTS_PER_PIECE = 4           # each convex piece: up to 4 vertices
MAX_INTERSECTION_VERTS = 8        # ≤ n1 + n2 (4 + 4)

#include <stdio.h>
#include <math.h>

_CUDA_SRC = r"""
extern "C" {
#define MAX_PIECES 4
#define MAX_VERTS_PER_PIECE 4
#define MAX_INTERSECTION_VERTS 8
#define MAX_RADIUS """ + str(MAX_RADIUS) + r"""

// Constant memory for polygon vertices - cached on-chip, broadcast to all threads
__constant__ double const_piece_xy[MAX_PIECES * MAX_VERTS_PER_PIECE * 2];
__constant__ int const_piece_nverts[MAX_PIECES];

""" + pack_cuda_primitives.PRIMITIVE_SRC + r"""

__device__ __forceinline__ void compute_tree_poly_and_aabb(
    double3 pose,
    int pi,  // which piece to compute
    d2 out_poly[MAX_VERTS_PER_PIECE],
    double& out_aabb_min_x,
    double& out_aabb_max_x,
    double& out_aabb_min_y,
    double& out_aabb_max_y)
{
    // Precompute transform for this pose
    double c = 0.0;
    double s = 0.0;
    sincos(pose.z, &s, &c);
    
    int n = const_piece_nverts[pi];
    double min_x = 1e30, max_x = -1e30;
    double min_y = 1e30, max_y = -1e30;

    // Load and transform piece with this pose, compute AABB
    for (int v = 0; v < n; ++v) {
        int idx = pi * MAX_VERTS_PER_PIECE + v;
        int base = 2 * idx;
        double x = const_piece_xy[base + 0];
        double y = const_piece_xy[base + 1];
        
        // Apply pose transform
        double x_t = c * x - s * y + pose.x;
        double y_t = s * x + c * y + pose.y;
        out_poly[v] = make_d2(x_t, y_t);
        
        // Update AABB
        min_x = fmin(min_x, x_t);
        max_x = fmax(max_x, x_t);
        min_y = fmin(min_y, y_t);
        max_y = fmax(max_y, y_t);
    }
    
    out_aabb_min_x = min_x;
    out_aabb_max_x = max_x;
    out_aabb_min_y = min_y;
    out_aabb_max_y = max_y;
}

// Compute overlap for a single piece (pi) of the reference tree against all pieces of trees in the list.
// This enables parallelization across the 4 pieces of the reference tree.
__device__ double overlap_ref_with_list_piece(
    const double3 ref,
    const double* __restrict__ xyt_3xN, // flattened row-major: 3 rows, N cols
    const int n,
    const int pi) // piece index to process (0-3)
{
    double sum = 0.0;

    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;
    
    // Compute only the assigned ref piece
    d2 ref_poly[MAX_VERTS_PER_PIECE];
    double ref_aabb_min_x;
    double ref_aabb_max_x;
    double ref_aabb_min_y;
    double ref_aabb_max_y;
    
    compute_tree_poly_and_aabb(ref, pi, ref_poly, ref_aabb_min_x, ref_aabb_max_x,
                                ref_aabb_min_y, ref_aabb_max_y);
    
    int n1 = const_piece_nverts[pi];
    
    // Loop over all trees in the list
    for (int i = 0; i < n; ++i) {
        double3 other;
        other.x = row_x[i];
        other.y = row_y[i];
        other.z = row_t[i];

        // Skip if poses are identical (self-collision)
        if (other.x == ref.x && other.y == ref.y && other.z == ref.z) {
            continue;
        }

        // Early exit: check if tree centers are too far apart
        double dx = other.x - ref.x;
        double dy = other.y - ref.y;
        double dist_sq = dx*dx + dy*dy;
        double max_overlap_dist = 2.0 * MAX_RADIUS;
        
        if (dist_sq > max_overlap_dist * max_overlap_dist) {
            continue;  // Trees too far apart to overlap
        }
        
        double total = 0.0;

        // Process only the assigned piece (pi) against all pieces of other tree
        for (int pj = 0; pj < MAX_PIECES; ++pj) {
            int n2 = const_piece_nverts[pj];
            
            // Compute only this piece of the other tree
            d2 other_poly[MAX_VERTS_PER_PIECE];
            double other_aabb_min_x;
            double other_aabb_max_x;
            double other_aabb_min_y;
            double other_aabb_max_y;
            
            compute_tree_poly_and_aabb(other, pj, other_poly, other_aabb_min_x, other_aabb_max_x,
                                        other_aabb_min_y, other_aabb_max_y);

            // AABB overlap test - early exit if no overlap
            if (ref_aabb_max_x < other_aabb_min_x || other_aabb_max_x < ref_aabb_min_x ||
                ref_aabb_max_y < other_aabb_min_y || other_aabb_max_y < ref_aabb_min_y) {
                continue;  // No AABB overlap, skip expensive intersection
            }

            total += convex_intersection_area(ref_poly, n1, other_poly, n2);
        }
        
        sum += total;
    }

    return sum;
}

// Compute sum of overlap areas between a reference tree `ref` and a list
// of other trees provided as a flattened 3xN array (row-major: row0=x, row1=y, row2=theta).
// Always skips comparing ref with identical pose in the other list.
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double* __restrict__ xyt_3xN, // flattened row-major: 3 rows, N cols
    const int n)
{
    double sum = 0.0;
    // Sum across all 4 pieces
    for (int pi = 0; pi < MAX_PIECES; ++pi) {
        sum += overlap_ref_with_list_piece(ref, xyt_3xN, n, pi);
    }
    return sum;
}

// Backward pass for a single piece (pi) of the reference tree
// Computes gradient contribution from one piece against all trees in the list
__device__ void backward_overlap_ref_with_list_piece(
    const double3 ref,
    const double* __restrict__ xyt_3xN,
    const int n,
    double d_overlap_sum,  // gradient w.r.t. output overlap sum
    const int pi,          // piece index to process (0-3)
    double3* d_ref)        // output: gradient w.r.t. ref pose (accumulated)
{
    if (d_overlap_sum == 0.0) return;

    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;
    
    // Compute transform coefficients for ref tree
    double c_ref = 0.0, s_ref = 0.0;
    sincos(ref.z, &s_ref, &c_ref);
    
    // Compute only the assigned ref piece
    d2 ref_poly[MAX_VERTS_PER_PIECE];
    double ref_aabb_min_x;
    double ref_aabb_max_x;
    double ref_aabb_min_y;
    double ref_aabb_max_y;
    
    compute_tree_poly_and_aabb(ref, pi, ref_poly, ref_aabb_min_x, ref_aabb_max_x,
                                ref_aabb_min_y, ref_aabb_max_y);
    
    // Load local piece vertices from constant memory for this piece only
    d2 ref_local_piece[MAX_VERTS_PER_PIECE];
    int n1 = const_piece_nverts[pi];
    for (int v = 0; v < n1; ++v) {
        int idx = pi * MAX_VERTS_PER_PIECE + v;
        int base = 2 * idx;
        ref_local_piece[v].x = const_piece_xy[base + 0];
        ref_local_piece[v].y = const_piece_xy[base + 1];
    }
    
    // Accumulate gradients locally
    double3 d_ref_local;
    d_ref_local.x = 0.0;
    d_ref_local.y = 0.0;
    d_ref_local.z = 0.0;
    
    // Loop over all trees in the list
    for (int i = 0; i < n; ++i) {
        double3 other;
        other.x = row_x[i];
        other.y = row_y[i];
        other.z = row_t[i];

        // Skip if poses are identical
        if (other.x == ref.x && other.y == ref.y && other.z == ref.z) {
            continue;
        }

        // Early exit check
        double dx = other.x - ref.x;
        double dy = other.y - ref.y;
        double dist_sq = dx*dx + dy*dy;
        double max_overlap_dist = 2.0 * MAX_RADIUS;
        
        if (dist_sq > max_overlap_dist * max_overlap_dist) {
            continue;
        }
        
        // Process only the assigned piece (pi) against all pieces of other tree
        for (int pj = 0; pj < MAX_PIECES; ++pj) {
            int n2 = const_piece_nverts[pj];
            
            // Compute only this piece of the other tree
            d2 other_poly[MAX_VERTS_PER_PIECE];
            double other_aabb_min_x;
            double other_aabb_max_x;
            double other_aabb_min_y;
            double other_aabb_max_y;
            
            compute_tree_poly_and_aabb(other, pj, other_poly, other_aabb_min_x, other_aabb_max_x,
                                        other_aabb_min_y, other_aabb_max_y);

            // AABB overlap test
            if (ref_aabb_max_x < other_aabb_min_x || other_aabb_max_x < ref_aabb_min_x ||
                ref_aabb_max_y < other_aabb_min_y || other_aabb_max_y < ref_aabb_min_y) {
                continue;
            }

            // Backward through intersection area
            d2 d_ref_poly[MAX_VERTS_PER_PIECE];
            d2 d_other_poly[MAX_VERTS_PER_PIECE];
            
            backward_convex_intersection_area(
                ref_poly, n1,
                other_poly, n2,
                d_overlap_sum,  // gradient flows from output
                d_ref_poly,
                d_other_poly);
            
            // Backward through transform for ref piece
            double3 d_ref_pose_piece;
            backward_transform_vertices(
                ref_local_piece, n1,
                d_ref_poly,
                c_ref, s_ref,
                &d_ref_pose_piece);
            
            // Accumulate into local gradient
            d_ref_local.x += d_ref_pose_piece.x;
            d_ref_local.y += d_ref_pose_piece.y;
            d_ref_local.z += d_ref_pose_piece.z;
        }
    }
    
    // Atomically accumulate the local gradient into the output
    // Multiple threads (one per piece) will contribute to the same gradient
    atomicAdd(&d_ref->x, d_ref_local.x);
    atomicAdd(&d_ref->y, d_ref_local.y);
    atomicAdd(&d_ref->z, d_ref_local.z);
}

// Backward pass for overlap_ref_with_list
// Computes gradient of overlap sum w.r.t. ref pose using analytic derivatives
__device__ void backward_overlap_ref_with_list(
    const double3 ref,
    const double* __restrict__ xyt_3xN,
    const int n,
    double d_overlap_sum,  // gradient w.r.t. output overlap sum
    double3* d_ref)        // output: gradient w.r.t. ref pose
{
    d_ref->x = 0.0;
    d_ref->y = 0.0;
    d_ref->z = 0.0;
    
    // Sum gradients across all 4 pieces
    for (int pi = 0; pi < MAX_PIECES; ++pi) {
        backward_overlap_ref_with_list_piece(ref, xyt_3xN, n, d_overlap_sum, pi, d_ref);
    }
}

// Compute the area of a single convex polygon that lies outside the square [-h/2, h/2] x [-h/2, h/2]
// Uses the clipping algorithm to find the intersection with the square, then subtracts from polygon area
__device__ double convex_area_outside_square(
    const d2* __restrict__ poly,
    const int n,
    const double h)
{
    if (n < 3) return 0.0;
    
    double half_h = h * 0.5;
    
    // Define square vertices (CCW order)
    d2 square[4];
    square[0] = make_d2(-half_h, -half_h);
    square[1] = make_d2(half_h, -half_h);
    square[2] = make_d2(half_h, half_h);
    square[3] = make_d2(-half_h, half_h);
    
    // Compute intersection area between polygon and square
    double intersection_area = convex_intersection_area(poly, n, square, 4);
    
    // Compute total polygon area
    double total_area = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        total_area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
    }
    total_area = fabs(total_area) * 0.5;
    
    // Area outside = total area - intersection area
    return total_area - intersection_area;
}

// Compute total area outside square boundary for one piece (pi) of one tree
// Also computes gradients w.r.t. pose (x, y, theta) and h using finite differences
__device__ double boundary_area_piece(
    const double3 pose,
    const double h,
    const int pi,  // piece index to process (0-3)
    double3* d_pose,  // output: gradient w.r.t. pose (can be NULL)
    double* d_h)      // output: gradient w.r.t. h (can be NULL)
{
    // Compute transformed polygon for this piece
    d2 poly[MAX_VERTS_PER_PIECE];
    double aabb_min_x, aabb_max_x, aabb_min_y, aabb_max_y;
    
    compute_tree_poly_and_aabb(pose, pi, poly, aabb_min_x, aabb_max_x,
                                aabb_min_y, aabb_max_y);
    
    int n = const_piece_nverts[pi];
    
    // Compute area outside square for this piece
    double area = convex_area_outside_square(poly, n, h);
    
    // Compute gradients using finite differences if requested
    if (d_pose != NULL || d_h != NULL) {
        const double eps = 1e-3;
        
        if (d_pose != NULL) {
            // Gradient w.r.t. x
            double3 pose_plus = pose;
            pose_plus.x += eps;
            d2 poly_plus[MAX_VERTS_PER_PIECE];
            compute_tree_poly_and_aabb(pose_plus, pi, poly_plus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            double area_plus = convex_area_outside_square(poly_plus, n, h);
            
            double3 pose_minus = pose;
            pose_minus.x -= eps;
            d2 poly_minus[MAX_VERTS_PER_PIECE];
            compute_tree_poly_and_aabb(pose_minus, pi, poly_minus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            double area_minus = convex_area_outside_square(poly_minus, n, h);
            
            d_pose->x = (area_plus - area_minus) / (2.0 * eps);
            
            // Gradient w.r.t. y
            pose_plus = pose;
            pose_plus.y += eps;
            compute_tree_poly_and_aabb(pose_plus, pi, poly_plus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            area_plus = convex_area_outside_square(poly_plus, n, h);
            
            pose_minus = pose;
            pose_minus.y -= eps;
            compute_tree_poly_and_aabb(pose_minus, pi, poly_minus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            area_minus = convex_area_outside_square(poly_minus, n, h);
            
            d_pose->y = (area_plus - area_minus) / (2.0 * eps);
            
            // Gradient w.r.t. theta
            pose_plus = pose;
            pose_plus.z += eps;
            compute_tree_poly_and_aabb(pose_plus, pi, poly_plus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            area_plus = convex_area_outside_square(poly_plus, n, h);
            
            pose_minus = pose;
            pose_minus.z -= eps;
            compute_tree_poly_and_aabb(pose_minus, pi, poly_minus, aabb_min_x, aabb_max_x,
                                        aabb_min_y, aabb_max_y);
            area_minus = convex_area_outside_square(poly_minus, n, h);
            
            d_pose->z = (area_plus - area_minus) / (2.0 * eps);
        }
        
        if (d_h != NULL) {
            // Gradient w.r.t. h
            double area_plus = convex_area_outside_square(poly, n, h + eps);
            double area_minus = convex_area_outside_square(poly, n, h - eps);
            *d_h = (area_plus - area_minus) / (2.0 * eps);
        }
    }
    
    return area;
}

// Compute total boundary violation area for a list of trees
// Each tree consists of 4 convex pieces
// Uses N*4 threads: one thread per piece of each tree
__device__ void boundary_list_total(
    const double* __restrict__ xyt_3xN,  // flattened row-major: 3 rows, N cols
    const int n,
    const double h,
    double* __restrict__ out_total,
    double* __restrict__ out_grads,  // if non-NULL, write gradients [n*3]
    double* __restrict__ out_grad_h)  // if non-NULL, write gradient w.r.t. h
{
    // Thread organization: 4 threads per tree
    // tid = tree_idx * 4 + piece_idx
    int tid = threadIdx.x;
    int tree_idx = tid / 4;  // which tree (0 to n-1)
    int piece_idx = tid % 4; // which piece of that tree (0 to 3)
    
    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;
    
    double local_area = 0.0;
    
    if (tree_idx < n) {
        double3 pose;
        pose.x = row_x[tree_idx];
        pose.y = row_y[tree_idx];
        pose.z = row_t[tree_idx];
        
        // Compute gradients if requested
        double3 d_pose;
        double d_h_local;
        
        // Each thread computes area outside boundary for one piece with gradients
        local_area = boundary_area_piece(pose, h, piece_idx, 
                                          (out_grads != NULL) ? &d_pose : NULL,
                                          (out_grad_h != NULL) ? &d_h_local : NULL);
        
        // Atomic add for total area
        atomicAdd(out_total, local_area);
        
        // Accumulate gradients if computed
        if (out_grads != NULL) {
            // Each of the 4 threads for this tree contributes gradient from its piece
            atomicAdd(&out_grads[tree_idx * 3 + 0], d_pose.x);
            atomicAdd(&out_grads[tree_idx * 3 + 1], d_pose.y);
            atomicAdd(&out_grads[tree_idx * 3 + 2], d_pose.z);
        }
        
        if (out_grad_h != NULL) {
            // All threads contribute to d/dh
            atomicAdd(out_grad_h, d_h_local);
        }
    }
}

// Sum overlaps between trees in xyt1 and trees in xyt2.
// Each tree in xyt1 is compared against all trees in xyt2.
// Identical poses are automatically skipped.
// Result is divided by 2 since each pair is counted twice (when xyt1 == xyt2).
// Computes gradients for all trees in xyt1 if out_grads is non-NULL.
// When xyt1 == xyt2 (same pointer), also accumulates gradients from the "other" side.
//
// NEW: Uses 4 threads per reference tree, one for each polygon piece.
// Thread organization: tid = tree_idx * 4 + piece_idx
__device__ void overlap_list_total(
    const double* __restrict__ xyt1_3xN,
    const int n1,
    const double* __restrict__ xyt2_3xN,
    const int n2,
    double* __restrict__ out_total,
    double* __restrict__ out_grads) // if non-NULL, write gradients to out_grads[n1*3]
{
    // Thread organization: 4 threads per tree
    // tid = tree_idx * 4 + piece_idx
    int tid = threadIdx.x;
    int tree_idx = tid / 4;  // which reference tree (0 to n1-1)
    int piece_idx = tid % 4; // which piece of that tree (0 to 3)

    const double* row_x = xyt1_3xN + 0 * n1;
    const double* row_y = xyt1_3xN + 1 * n1;
    const double* row_t = xyt1_3xN + 2 * n1;

    double local_sum = 0.0;

    if (tree_idx < n1) {
        double3 ref;
        ref.x = row_x[tree_idx];
        ref.y = row_y[tree_idx];
        ref.z = row_t[tree_idx];

        // Each thread computes overlap for one piece of the reference tree
        local_sum = overlap_ref_with_list_piece(ref, xyt2_3xN, n2, piece_idx);

        // Atomic add for overlap sum
        atomicAdd(out_total, local_sum / 2.0);

        // For gradients, all 4 threads participate
        if (out_grads != NULL) {
            // Initialize gradient to zero (only first thread does this)
            if (piece_idx == 0) {
                out_grads[tree_idx * 3 + 0] = 0.0;
                out_grads[tree_idx * 3 + 1] = 0.0;
                out_grads[tree_idx * 3 + 2] = 0.0;
            }
            
            // Ensure initialization is complete before all threads start accumulating
            __syncthreads();
            
            // Point to the output gradient location for this tree
            double3* d_ref_output = (double3*)(&out_grads[tree_idx * 3]);
            
            // Compute gradient contribution from this piece
            backward_overlap_ref_with_list_piece(ref, xyt2_3xN, n2, 1.0, piece_idx, d_ref_output);
        }
    }
}



// Multi-ensemble kernel: one block per ensemble
// Each block processes one ensemble by calling overlap_list_total
// 
// Parameters:
//   xyt1_list: array of pointers to xyt1 data for each ensemble
//   n1_list: array of n1 values (number of trees) for each ensemble
//   xyt2_list: array of pointers to xyt2 data for each ensemble  
//   n2_list: array of n2 values for each ensemble
//   out_totals: array of output totals, one per ensemble
//   out_grads_list: array of pointers to gradient outputs, one per ensemble (can be NULL)
//   num_ensembles: number of ensembles to process
__global__ void multi_overlap_list_total(
    const double** __restrict__ xyt1_list,  // [num_ensembles] pointers
    const int* __restrict__ n1_list,        // [num_ensembles]
    const double** __restrict__ xyt2_list,  // [num_ensembles] pointers
    const int* __restrict__ n2_list,        // [num_ensembles]
    double* __restrict__ out_totals,        // [num_ensembles]
    double** __restrict__ out_grads_list,   // [num_ensembles] pointers (NULL entries allowed)
    const int num_ensembles)
{
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) {
        return;  // Extra blocks beyond num_ensembles
    }
    
    // Load parameters for this ensemble
    const double* xyt1 = xyt1_list[ensemble_id];
    int n1 = n1_list[ensemble_id];
    const double* xyt2 = xyt2_list[ensemble_id];
    int n2 = n2_list[ensemble_id];
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_list != NULL) ? out_grads_list[ensemble_id] : NULL;
    
    // Initialize output
    if (threadIdx.x == 0) {
        *out_total = 0.0;
    }
    __syncthreads();
    
    // Call the existing overlap_list_total device function
    // This function is designed to be called by ALL threads in the block
    // It internally uses threadIdx.x to determine which tree each thread processes
    overlap_list_total(xyt1, n1, xyt2, n2, out_total, out_grads);
}

// Multi-ensemble kernel for boundary: one block per ensemble
// Each block processes one ensemble by calling boundary_list_total
//
// Parameters:
//   xyt_list: array of pointers to xyt data for each ensemble
//   n_list: array of n values (number of trees) for each ensemble
//   h_list: array of h values (boundary size) for each ensemble
//   out_totals: array of output totals, one per ensemble
//   out_grads_list: array of pointers to gradient outputs, one per ensemble (can be NULL)
//   out_grad_h_list: array of pointers to h gradients, one per ensemble (can be NULL)
//   num_ensembles: number of ensembles to process
__global__ void multi_boundary_list_total(
    const double** __restrict__ xyt_list,      // [num_ensembles] pointers
    const int* __restrict__ n_list,            // [num_ensembles]
    const double* __restrict__ h_list,         // [num_ensembles]
    double* __restrict__ out_totals,           // [num_ensembles]
    double** __restrict__ out_grads_list,      // [num_ensembles] pointers (NULL entries allowed)
    double** __restrict__ out_grad_h_list,     // [num_ensembles] pointers (NULL entries allowed)
    const int num_ensembles)
{
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) {
        return;  // Extra blocks beyond num_ensembles
    }
    
    // Load parameters for this ensemble
    const double* xyt = xyt_list[ensemble_id];
    int n = n_list[ensemble_id];
    double h = h_list[ensemble_id];
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_list != NULL) ? out_grads_list[ensemble_id] : NULL;
    double* out_grad_h = (out_grad_h_list != NULL) ? out_grad_h_list[ensemble_id] : NULL;
    
    // Initialize output
    if (threadIdx.x == 0) {
        *out_total = 0.0;
    }
    __syncthreads();
    
    // Call the existing boundary_list_total device function
    // This function is designed to be called by ALL threads in the block
    // It internally uses threadIdx.x to determine which tree each thread processes
    boundary_list_total(xyt, n, h, out_total, out_grads, out_grad_h);
}

} // extern "C"
"""



# ---------------------------------------------------------------------------
# 3. Build GPU data from Shapely polygons
# ---------------------------------------------------------------------------

# Device-side arrays holding the convex pieces (in local coordinates):
_piece_xy_d: cp.ndarray | None = None        # flattened [num_pieces * MAX_VERTS_PER_PIECE * 2]
_piece_nverts_d: cp.ndarray | None = None    # [num_pieces]

_num_pieces: int = 0

# Compiled CUDA module and kernel
_raw_module: cp.RawModule | None = None
_multi_overlap_list_total_kernel: cp.RawKernel | None = None
_multi_boundary_list_total_kernel: cp.RawKernel | None = None

# Flag to indicate lazy initialization completed
_initialized: bool = False


def _build_convex_piece_arrays(polys: list[Polygon]) -> tuple[np.ndarray, np.ndarray]:
    """From a list of convex Shapely polygons, build device-ready arrays.

    This does a one-time preprocessing step on the CPU:
    - Forces each polygon to CCW orientation (so "inside" is always to the left).
    - Drops the closing vertex from the exterior ring.
    - Pads each polygon up to MAX_VERTS_PER_PIECE by repeating the last vertex.

    The output is:
        piece_xy_flat : float64, shape (num_pieces * MAX_VERTS_PER_PIECE * 2,)
            Flattened (x, y) coordinates for all convex pieces, ready for
            uploading to the GPU.
        piece_nverts  : int32, shape (num_pieces,)
            Number of *real* vertices for each convex piece.

    These are static for the lifetime of the program and are uploaded once
    into device memory.
    """
    if not polys:
        raise RuntimeError(
            "CONVEX_PIECES is empty. Please insert your 4 convex Shapely polygons."
        )

    if len(polys) > MAX_PIECES:
        raise ValueError(
            f"At most {MAX_PIECES} convex pieces are supported, got {len(polys)}."
        )

    piece_arrays = []
    nverts_list = []

    for idx, poly in enumerate(polys):
        if not isinstance(poly, Polygon):
            raise TypeError(f"Entry {idx} in CONVEX_PIECES is not a Shapely Polygon.")

        # Force CCW orientation (inside to the left of edges). This matches the
        # convention used by the clipping kernel (cross >= 0 is "inside").
        poly_ccw = orient(poly, sign=1.0)

        coords = np.asarray(poly_ccw.exterior.coords[:-1], dtype=np.float64)  # drop closing vertex
        nverts = coords.shape[0]

        if nverts < 3:
            raise ValueError(f"Convex piece {idx} has < 3 vertices (n={nverts}).")
        if nverts > MAX_VERTS_PER_PIECE:
            raise ValueError(
                f"Convex piece {idx} has {nverts} vertices, but MAX_VERTS_PER_PIECE={MAX_VERTS_PER_PIECE}."
            )

        # Pad to MAX_VERTS_PER_PIECE by repeating the last vertex so the GPU
        # layout is fixed-size for each piece.
        padded = np.empty((MAX_VERTS_PER_PIECE, 2), dtype=np.float64)
        padded[:nverts, :] = coords
        padded[nverts:, :] = coords[-1]

        piece_arrays.append(padded)
        nverts_list.append(nverts)

    piece_xy = np.stack(piece_arrays, axis=0)       # (num_pieces, MAX_VERTS_PER_PIECE, 2)
    piece_xy_flat = piece_xy.reshape(-1)           # flatten
    piece_nverts = np.asarray(nverts_list, dtype=np.int32)

    return piece_xy_flat, piece_nverts


def _ensure_initialized() -> None:
    """Lazy initialization hook.

    On first call, this:
    - Converts CONVEX_PIECES into flat NumPy arrays.
    - Uploads the convex pieces to constant memory on device.
    - Compiles the CUDA source and fetches the overlap kernel.

    Subsequent calls are no-ops, so you can safely call this at the start
    of public API functions.
    """
    global _initialized, _piece_xy_d, _piece_nverts_d
    global _num_pieces, _raw_module, _multi_overlap_list_total_kernel, _multi_boundary_list_total_kernel

    if _initialized:
        return

    piece_xy_flat, piece_nverts = _build_convex_piece_arrays(CONVEX_PIECES)

    _num_pieces = piece_nverts.shape[0]

    # Keep host copies for reference (not used by kernels anymore)
    _piece_xy_d = cp.asarray(piece_xy_flat, dtype=cp.float64)
    _piece_nverts_d = cp.asarray(piece_nverts, dtype=cp.int32)

    # Persist the CUDA source to a stable .cu file inside kgs.temp_dir
    # and compile from that file so profilers can correlate source lines.
    persist_dir = os.fspath(kgs.temp_dir)
    persist_path = os.path.join(persist_dir, 'pack_cuda_saved.cu')

    # Perform search-replace to switch between float32 and float64
    cuda_src = _CUDA_SRC
    if USE_FLOAT32:
        # Replace double with float throughout the CUDA code
        cuda_src = cuda_src.replace('double', 'float')
        # Fix double3 -> float3
        cuda_src = cuda_src.replace('float3', 'float3')

    # Overwrite the file each time to ensure it matches the compiled source.
    # Let any IO errors propagate so callers see a clear failure.
    with open(persist_path, 'w', encoding='utf-8') as _f:
        _f.write(cuda_src)

    # Compile CUDA module from the in-memory source string. This keeps
    # behavior compatible across CuPy versions that may not accept
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH','')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH; please install the CUDA toolkit or add nvcc to PATH")

    ptx_path = os.path.join(persist_dir, 'pack_cuda_saved.ptx')
    cmd = [nvcc_path, "-O3", "-use_fast_math", "--ptxas-options=-v", "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]

    # Run nvcc and capture output to display diagnostics
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print compiler output (including --ptxas-options=-v diagnostics)
    if proc.stdout:
        print("=== NVCC Compilation Output ===")
        print(proc.stdout)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (exit {proc.returncode}):\n{proc.stdout}")

    # Load compiled PTX into a CuPy RawModule
    _raw_module = cp.RawModule(path=ptx_path)
    #_raw_module = cp.RawModule(code=_CUDA_SRC, backend='nvcc', options=())
    _multi_overlap_list_total_kernel = _raw_module.get_function("multi_overlap_list_total")
    _multi_boundary_list_total_kernel = _raw_module.get_function("multi_boundary_list_total")

    # Copy polygon data to constant memory (cached on-chip, broadcast to all threads)
    # Convert to appropriate dtype if using float32
    if USE_FLOAT32:
        piece_xy_flat_device = piece_xy_flat.astype(np.float32)
    else:
        piece_xy_flat_device = piece_xy_flat
    
    const_piece_xy_ptr = _raw_module.get_global('const_piece_xy')
    const_piece_nverts_ptr = _raw_module.get_global('const_piece_nverts')
    # Use memcpyHtoD to copy to device constant memory
    cp.cuda.runtime.memcpy(const_piece_xy_ptr.ptr, piece_xy_flat_device.ctypes.data, piece_xy_flat_device.nbytes, cp.cuda.runtime.memcpyHostToDevice)
    cp.cuda.runtime.memcpy(const_piece_nverts_ptr.ptr, piece_nverts.ctypes.data, piece_nverts.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    _initialized = True


# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------


def overlap_multi_ensemble(xyt1_list, xyt2_list, compute_grad: bool = True):
    """Compute total overlap sum for multiple ensembles in parallel.
    
    This launches one GPU block per ensemble, allowing many ensembles to run
    concurrently and utilize more of the GPU's streaming multiprocessors.
    
    Parameters
    ----------
    xyt1_list : list of array-like
        List of ensemble pose arrays, each shape (N1_i, 3) for ensemble i.
    xyt2_list : list of array-like
        List of ensemble pose arrays, each shape (N2_i, 3) for ensemble i.
    compute_grad : bool, optional
        If True, compute and return gradients. Default is True.
    
    Returns
    -------
    totals : cp.ndarray, shape (num_ensembles,)
        Total overlap for each ensemble.
    grads_list : list of cp.ndarray, optional
        List of gradient arrays (N1_i, 3) for each ensemble.
        Only returned if compute_grad=True.
    """
    _ensure_initialized()
    
    if len(xyt1_list) != len(xyt2_list):
        raise ValueError("xyt1_list and xyt2_list must have same length")
    
    num_ensembles = len(xyt1_list)
    if num_ensembles == 0:
        return cp.array([]), [] if compute_grad else None
    
    # Determine dtype based on USE_FLOAT32 setting
    dtype = cp.float32 if USE_FLOAT32 else cp.float64
    
    # Process and validate all input arrays
    xyt1_arrays = []
    xyt2_arrays = []
    n1_list = []
    n2_list = []
    max_n1 = 0
    
    for i, (xyt1, xyt2) in enumerate(zip(xyt1_list, xyt2_list)):
        xyt1_arr = cp.asarray(xyt1, dtype=dtype)
        if xyt1_arr.ndim != 2 or xyt1_arr.shape[1] != 3:
            raise ValueError(f"xyt1_list[{i}] must be shape (N,3)")
        
        xyt2_arr = cp.asarray(xyt2, dtype=dtype)
        if xyt2_arr.ndim != 2 or xyt2_arr.shape[1] != 3:
            raise ValueError(f"xyt2_list[{i}] must be shape (N,3)")
        
        n1 = int(xyt1_arr.shape[0])
        n2 = int(xyt2_arr.shape[0])
        
        # Flatten to 3xN row-major
        xyt1_3xN = cp.ascontiguousarray(xyt1_arr.T).ravel()
        xyt2_3xN = cp.ascontiguousarray(xyt2_arr.T).ravel()
        
        xyt1_arrays.append(xyt1_3xN)
        xyt2_arrays.append(xyt2_3xN)
        n1_list.append(n1)
        n2_list.append(n2)
        max_n1 = max(max_n1, n1)
    
    # Create arrays of pointers (device addresses)
    # CuPy doesn't have a direct way to create pointer arrays, so we use data.ptr
    xyt1_ptrs = cp.array([arr.data.ptr for arr in xyt1_arrays], dtype=cp.uint64)
    xyt2_ptrs = cp.array([arr.data.ptr for arr in xyt2_arrays], dtype=cp.uint64)
    
    n1_array = cp.array(n1_list, dtype=cp.int32)
    n2_array = cp.array(n2_list, dtype=cp.int32)
    
    # Allocate outputs
    out_totals = cp.zeros(num_ensembles, dtype=dtype)
    
    # Allocate gradients if requested
    grads_arrays = []
    grads_ptrs = None
    if compute_grad:
        for n1 in n1_list:
            grads_arrays.append(cp.zeros(n1 * 3, dtype=dtype))
        grads_ptrs = cp.array([arr.data.ptr for arr in grads_arrays], dtype=cp.uint64)
    
    # Launch kernel: one block per ensemble, max_n1 * 4 threads per block
    # (4 threads per tree, one for each polygon piece)
    blocks = num_ensembles
    threads_per_block = max_n1 * 4
    
    # Cast pointer arrays to proper type for kernel
    null_ptr = cp.array([0], dtype=cp.uint64)
    
    _multi_overlap_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt1_ptrs,
            n1_array,
            xyt2_ptrs,
            n2_array,
            out_totals,
            grads_ptrs if grads_ptrs is not None else null_ptr,
            np.int32(num_ensembles),
        ),
    )
    
    if compute_grad:
        # Reshape gradient arrays back to (N1_i, 3)
        grads_list = [grads_arrays[i].reshape(n1_list[i], 3) for i in range(num_ensembles)]
        return out_totals, grads_list
    else:
        return out_totals, None


def boundary_multi_ensemble(xyt_list, h_list, compute_grad: bool = False):
    """Compute total boundary violation area for multiple ensembles in parallel.
    
    This launches one GPU block per ensemble, allowing many ensembles to run
    concurrently and utilize more of the GPU's streaming multiprocessors.
    
    Parameters
    ----------
    xyt_list : list of array-like
        List of ensemble pose arrays, each shape (N_i, 3) for ensemble i.
    h_list : list of float
        List of boundary sizes, one per ensemble.
    compute_grad : bool, optional
        If True, compute and return gradients. Default is False.
    
    Returns
    -------
    totals : cp.ndarray, shape (num_ensembles,)
        Total boundary violation area for each ensemble.
    grads_list : list of cp.ndarray, optional
        List of gradient arrays (N_i, 3) for each ensemble.
        Only returned if compute_grad=True.
    grad_h_list : list of cp.ndarray, optional
        List of h gradient arrays (1,) for each ensemble.
        Only returned if compute_grad=True.
    """
    _ensure_initialized()
    
    if len(xyt_list) != len(h_list):
        raise ValueError("xyt_list and h_list must have same length")
    
    num_ensembles = len(xyt_list)
    if num_ensembles == 0:
        if compute_grad:
            return cp.array([]), [], []
        else:
            return cp.array([]), None, None
    
    # Determine dtype based on USE_FLOAT32 setting
    dtype = cp.float32 if USE_FLOAT32 else cp.float64
    
    # Process and validate all input arrays
    xyt_arrays = []
    n_list = []
    max_n = 0
    
    for i, xyt in enumerate(xyt_list):
        xyt_arr = cp.asarray(xyt, dtype=dtype)
        if xyt_arr.ndim != 2 or xyt_arr.shape[1] != 3:
            raise ValueError(f"xyt_list[{i}] must be shape (N,3)")
        
        n = int(xyt_arr.shape[0])
        
        # Flatten to 3xN row-major
        xyt_3xN = cp.ascontiguousarray(xyt_arr.T).ravel()
        
        xyt_arrays.append(xyt_3xN)
        n_list.append(n)
        max_n = max(max_n, n)
    
    # Validate h_list
    h_array = cp.array(h_list, dtype=dtype)
    if h_array.shape[0] != num_ensembles:
        raise ValueError(f"h_list must have {num_ensembles} elements")
    
    # Create arrays of pointers (device addresses)
    xyt_ptrs = cp.array([arr.data.ptr for arr in xyt_arrays], dtype=cp.uint64)
    n_array = cp.array(n_list, dtype=cp.int32)
    
    # Allocate outputs
    out_totals = cp.zeros(num_ensembles, dtype=dtype)
    
    # Allocate gradients if requested
    grads_arrays = []
    grads_ptrs = None
    grad_h_arrays = []
    grad_h_ptrs = None
    
    if compute_grad:
        for n in n_list:
            grads_arrays.append(cp.zeros(n * 3, dtype=dtype))
        grads_ptrs = cp.array([arr.data.ptr for arr in grads_arrays], dtype=cp.uint64)
        
        for _ in range(num_ensembles):
            grad_h_arrays.append(cp.zeros(1, dtype=dtype))
        grad_h_ptrs = cp.array([arr.data.ptr for arr in grad_h_arrays], dtype=cp.uint64)
    
    # Launch kernel: one block per ensemble, max_n * 4 threads per block
    blocks = num_ensembles
    threads_per_block = max_n * 4
    
    # Cast pointer arrays to proper type for kernel
    null_ptr = cp.array([0], dtype=cp.uint64)
    
    _multi_boundary_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt_ptrs,
            n_array,
            h_array,
            out_totals,
            grads_ptrs if grads_ptrs is not None else null_ptr,
            grad_h_ptrs if grad_h_ptrs is not None else null_ptr,
            np.int32(num_ensembles),
        ),
    )
    
    if compute_grad:
        # Reshape gradient arrays back to (N_i, 3) and (1,)
        grads_list = [grads_arrays[i].reshape(n_list[i], 3) for i in range(num_ensembles)]
        grad_h_list = [grad_h_arrays[i] for i in range(num_ensembles)]
        return out_totals, grads_list, grad_h_list
    else:
        return out_totals, None, None


