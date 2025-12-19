"""
gpu_overlap.py

Compute overlap area between TWO IDENTICAL non-convex polygons using CUDA via CuPy.

You provide a convex decomposition of your polygon as 4 convex Shapely Polygons.
The module uploads those convex pieces to the GPU and exposes

    compute_overlap_area_gpu(x1, y1, theta1, x2, y2, theta2)

which returns the overlap area between pose-1 and pose-2, computed on the GPU.

All inputs (x1, y1, theta1, x2, y2, theta2) are expected to be **scalars**
(Python floats or NumPy scalars). The function returns a single CuPy scalar
(kgs.dtype_cp).
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

// Struct to hold crystal axes (axis A and axis B, each with x and y components)
struct CrystalAxes {
    double ax, ay;  // Axis A vector
    double bx, by;  // Axis B vector
};

// Constant memory for polygon vertices - cached on-chip, broadcast to all threads
__constant__ double const_piece_xy[MAX_PIECES * MAX_VERTS_PER_PIECE * 2];
__constant__ int const_piece_nverts[MAX_PIECES];

// Constant memory for tree vertices (precomputed from center tree)
__constant__ double const_tree_vertices_xy[256];  // Max 128 vertices (256 floats for x,y pairs)
__constant__ int const_n_tree_vertices;

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
        out_poly[v] = make_double2(x_t, y_t);
        
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
// Optionally computes gradients w.r.t. the reference pose when compute_grads is non-zero.
// Process overlap/separation between one ref piece and all pieces of one other tree
__device__ double process_tree_pair_piece(
    const double3 ref,
    const double3 other,
    const int pi,
    const d2* __restrict__ ref_poly,
    const int n1,
    const d2* __restrict__ ref_local_piece,
    const double ref_aabb_min_x,
    const double ref_aabb_max_x,
    const double ref_aabb_min_y,
    const double ref_aabb_max_y,
    const double c_ref,
    const double s_ref,
    const int use_separation,
    const int compute_grads,
    double3* d_ref_local)  // output: accumulated gradient (modified in-place)
{
    // Early exit: check if tree centers are too far apart
    double dx = other.x - ref.x;
    double dy = other.y - ref.y;
    double dist_sq = dx*dx + dy*dy;
    double max_overlap_dist = 2.0 * MAX_RADIUS;

    if (dist_sq > max_overlap_dist * max_overlap_dist) {
        return 0.0;  // Trees too far apart to overlap
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

        if (!use_separation) {
            
        } else {
            // Separation-based primitive
            double sep = 0.0;
            double dsep_dx = 0.0;
            double dsep_dy = 0.0;
            double dsep_dtheta = 0.0;

            // Pass NULL for gradient outputs when not computing gradients
            sat_separation_with_grad_pose(
                ref_local_piece, n1,
                other_poly, n2,
                ref.x, ref.y, c_ref, s_ref,
                &sep,
                compute_grads ? &dsep_dx : NULL,
                compute_grads ? &dsep_dy : NULL,
                compute_grads ? &dsep_dtheta : NULL);

            // Only positive penetration contributes
            if (sep > 0.0) {
                total += sep * sep;

                if (compute_grads) {
                    // Gradient of sep^2: 2 * sep * dsep/dparam
                    d_ref_local->x += 2.0 * sep * dsep_dx;
                    d_ref_local->y += 2.0 * sep * dsep_dy;
                    d_ref_local->z += 2.0 * sep * dsep_dtheta;
                }
            }
        }
    }

    return total;
}

__device__ double overlap_ref_with_list_piece(
    const double3 ref,
    const double* __restrict__ xyt_Nx3, // flattened: [n, 3] in C-contiguous layout
    const int n,
    const int pi,          // piece index to process (0-3)
    double3* d_ref,        // output: gradient w.r.t. ref pose (accumulated), can be NULL
    const int use_separation, // if non-zero, compute separation-based cost (sum of sep^2)
    const int skip_index,  // index to skip (self-collision), use -1 to skip none
    const int compute_grads, // if non-zero, compute gradients
    const int use_crystal, // if non-zero, loop over 3x3 cell for each tree
    const CrystalAxes crystal_axes, // crystal axes (only used if use_crystal is non-zero)
    const int only_self_interactions) // if non-zero, only compute interactions when i == skip_index
{
    double sum = 0.0;

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

    int n1 = const_piece_nverts[pi];

    // Load local piece vertices from constant memory (needed for gradients or separation)
    d2 ref_local_piece[MAX_VERTS_PER_PIECE];
    if (compute_grads || use_separation) {
        for (int v = 0; v < n1; ++v) {
            int idx = pi * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            ref_local_piece[v].x = const_piece_xy[base + 0];
            ref_local_piece[v].y = const_piece_xy[base + 1];
        }
    }

    // Accumulate gradients locally (only if computing gradients)
    double3 d_ref_local;
    if (compute_grads) {
        d_ref_local.x = 0.0;
        d_ref_local.y = 0.0;
        d_ref_local.z = 0.0;
    }

    // Loop over all trees in the list
    for (int i = 0; i < n; ++i) {
        // Read pose with strided access: [i, component]
        double3 other_base;
        other_base.x = xyt_Nx3[i * 3 + 0];
        other_base.y = xyt_Nx3[i * 3 + 1];
        other_base.z = xyt_Nx3[i * 3 + 2];

        // Check if this is a self-comparison
        int is_self = (i == skip_index) ? 1 : 0;

        // If only_self_interactions is set, skip non-self interactions
        if (only_self_interactions && !is_self) {
            continue;
        }

        if (use_crystal) {
            
        } else {
            // No crystal: skip self-comparison entirely
            if (is_self) {
                continue;
            }

            // Process the original tree
            double total = process_tree_pair_piece(
                ref, other_base, pi,
                ref_poly, n1, ref_local_piece,
                ref_aabb_min_x, ref_aabb_max_x,
                ref_aabb_min_y, ref_aabb_max_y,
                c_ref, s_ref,
                use_separation, compute_grads,
                &d_ref_local);

            sum += total;
        }
    }

    // Atomically accumulate the local gradient into the output (only if computing gradients)
    if (compute_grads && d_ref != NULL) {
        atomicAdd(&d_ref->x, d_ref_local.x);
        atomicAdd(&d_ref->y, d_ref_local.y);
        atomicAdd(&d_ref->z, d_ref_local.z);
    }

    return sum;
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
    square[0] = make_double2(-half_h, -half_h);
    square[1] = make_double2(half_h, -half_h);
    square[2] = make_double2(half_h, half_h);
    square[3] = make_double2(-half_h, half_h);
    
    // Compute intersection area between polygon and square (no gradients needed)
    double intersection_area = convex_intersection_area(poly, n, square, 4, NULL, NULL);
    
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
    const double* __restrict__ xyt_Nx3,  // flattened: [n, 3] in C-contiguous layout
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
    
    double local_area = 0.0;
    
    if (tree_idx < n) {
        // Read pose with strided access: [tree_idx, component]
        double3 pose;
        pose.x = xyt_Nx3[tree_idx * 3 + 0];
        pose.y = xyt_Nx3[tree_idx * 3 + 1];
        pose.z = xyt_Nx3[tree_idx * 3 + 2];
        
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
            // Write to [tree_idx, component] layout
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
// Uses 4 threads per reference tree, one for each polygon piece.
// Thread organization: tid = tree_idx * 4 + piece_idx
__device__ void overlap_list_total(
    const double* __restrict__ xyt1_Nx3,
    const int n1,
    const double* __restrict__ xyt2_Nx3,
    const int n2,
    double* __restrict__ out_total,
    double* __restrict__ out_grads, // if non-NULL, write gradients to out_grads[n1*3]
    const int use_separation, // non-zero -> use separation sum-of-squares path
    const int use_crystal, // if non-zero, loop over 3x3 cell for each tree
    const CrystalAxes crystal_axes, // crystal axes (only used if use_crystal is non-zero)
    const int only_self_interactions) // if non-zero, only compute interactions when i == skip_index
{
    // Thread organization: 4 threads per tree
    // tid = tree_idx * 4 + piece_idx
    int tid = threadIdx.x;
    int tree_idx = tid / 4;  // which reference tree (0 to n1-1)
    int piece_idx = tid % 4; // which piece of that tree (0 to 3)

    double local_sum = 0.0;

    if (tree_idx < n1) {
        // Read pose with strided access: [tree_idx, component]
        double3 ref;
        ref.x = xyt1_Nx3[tree_idx * 3 + 0];
        ref.y = xyt1_Nx3[tree_idx * 3 + 1];
        ref.z = xyt1_Nx3[tree_idx * 3 + 2];

        // Determine if we need to compute gradients
        int compute_grads = (out_grads != NULL) ? 1 : 0;

        // Initialize gradient to zero (only first thread does this)
        if (compute_grads && piece_idx == 0) {
            out_grads[tree_idx * 3 + 0] = 0.0;
            out_grads[tree_idx * 3 + 1] = 0.0;
            out_grads[tree_idx * 3 + 2] = 0.0;
        }

        // Ensure initialization is complete before all threads start accumulating
        __syncthreads();

        // Each thread computes overlap (or separation) for one piece of the reference tree
        double3* d_ref_output = compute_grads ? (double3*)(&out_grads[tree_idx * 3]) : NULL;
        local_sum = overlap_ref_with_list_piece(ref, xyt2_Nx3, n2, piece_idx, d_ref_output, use_separation, tree_idx, compute_grads, use_crystal, crystal_axes, only_self_interactions);

        // Atomic add for overlap sum
        // If only_self_interactions, don't divide by 2 since we're not double-counting
        // (each tree only computes its self-interaction once)
        if (only_self_interactions) {
            atomicAdd(out_total, local_sum);
        } else {
            atomicAdd(out_total, local_sum / 2.0);
        }
    }
}



// Multi-ensemble kernel: one block per ensemble
// Each block processes one ensemble by calling overlap_list_total
//
// Accepts single 3D arrays with strided access - no transpose needed
// Parameters:
//   xyt1_base: base pointer to 3D array [num_ensembles, n_trees, 3] in C-contiguous layout
//   xyt2_base: base pointer to 3D array [num_ensembles, n_trees, 3] in C-contiguous layout
//   n_trees: number of trees per ensemble (same for all)
//   out_totals: array of output totals, one per ensemble
//   out_grads_base: base pointer to gradient output [num_ensembles, n_trees, 3] (can be NULL)
//   num_ensembles: number of ensembles to process
//   use_separation: if non-zero, use separation-based cost
//   use_crystal: if non-zero, loop over 3x3 cell for each tree
//   crystal_axes_base: base pointer to crystal axes [num_ensembles, 4] (NULL if use_crystal is 0)
__global__ void multi_overlap_list_total(
    const double* __restrict__ xyt1_base,      // base pointer to [num_ensembles, n_trees, 3]
    const double* __restrict__ xyt2_base,      // base pointer to [num_ensembles, n_trees, 3]
    const int n_trees,                          // number of trees per ensemble
    double* __restrict__ out_totals,           // [num_ensembles]
    double* __restrict__ out_grads_base,       // base pointer to [num_ensembles, n_trees, 3] (NULL allowed)
    const int num_ensembles,
    const int use_separation,
    const int use_crystal,
    const double* __restrict__ crystal_axes_base, // base pointer to [num_ensembles, 4] (NULL if use_crystal is 0)
    const int only_self_interactions) // if non-zero, only compute interactions when i == skip_index
{
    int ensemble_id = blockIdx.x;

    if (ensemble_id >= num_ensembles) {
        return;  // Extra blocks beyond num_ensembles
    }

    // Calculate offset for this ensemble's data using strided access
    // Layout: [num_ensembles, n_trees, 3]
    // Stride: each ensemble is n_trees*3 elements apart
    int ensemble_stride = n_trees * 3;
    const double* xyt1_ensemble = xyt1_base + ensemble_id * ensemble_stride;
    const double* xyt2_ensemble = xyt2_base + ensemble_id * ensemble_stride;

    // Load crystal axes for this ensemble
    CrystalAxes crystal_axes;
    if (use_crystal && crystal_axes_base != NULL) {
        const double* axes = crystal_axes_base + ensemble_id * 4;
        crystal_axes.ax = axes[0];
        crystal_axes.ay = axes[1];
        crystal_axes.bx = axes[2];
        crystal_axes.by = axes[3];
    } else {
        // Initialize to zero (not used when use_crystal is 0)
        crystal_axes.ax = 0.0;
        crystal_axes.ay = 0.0;
        crystal_axes.bx = 0.0;
        crystal_axes.by = 0.0;
    }

    // Parameters for overlap_list_total
    int n1 = n_trees;
    int n2 = n_trees;
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_base != NULL) ? (out_grads_base + ensemble_id * ensemble_stride) : NULL;

    // Initialize output
    if (threadIdx.x == 0) {
        *out_total = 0.0;
    }
    // Initialize gradient buffer
    if (out_grads != NULL) {
        int tid = threadIdx.x;
        int max_tid = n_trees * 4;
        for (int idx = tid; idx < n_trees * 3; idx += max_tid) {
            out_grads[idx] = 0.0;
        }
    }
    __syncthreads();

    // Call overlap_list_total - it now reads (n_trees, 3) format directly
    overlap_list_total(xyt1_ensemble, n1, xyt2_ensemble, n2, out_total, out_grads, use_separation, use_crystal, crystal_axes, only_self_interactions);
}

// Multi-ensemble kernel for boundary: one block per ensemble
// Each block processes one ensemble by calling boundary_list_total
//
// Accepts single 3D array with strided access - no transpose needed
// Parameters:
//   xyt_base: base pointer to 3D array [num_ensembles, n_trees, 3] in C-contiguous layout
//   n_trees: number of trees per ensemble (same for all)
//   h_list: array of h values (boundary size) for each ensemble
//   out_totals: array of output totals, one per ensemble
//   out_grads_base: base pointer to gradient output [num_ensembles, n_trees, 3] (can be NULL)
//   out_grad_h: array of h gradients [num_ensembles] (can be NULL)
//   num_ensembles: number of ensembles to process
__global__ void multi_boundary_list_total(
    const double* __restrict__ xyt_base,       // base pointer to [num_ensembles, n_trees, 3]
    const int n_trees,                          // number of trees per ensemble
    const double* __restrict__ h_list,         // [num_ensembles]
    double* __restrict__ out_totals,           // [num_ensembles]
    double* __restrict__ out_grads_base,       // base pointer to [num_ensembles, n_trees, 3] (NULL allowed)
    double* __restrict__ out_grad_h,           // [num_ensembles] (NULL allowed)
    const int num_ensembles)
{
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) {
        return;  // Extra blocks beyond num_ensembles
    }
    
    // Calculate offset for this ensemble's data using strided access
    // Layout: [num_ensembles, n_trees, 3]
    // Stride: each ensemble is n_trees*3 elements apart
    int ensemble_stride = n_trees * 3;
    const double* xyt_ensemble = xyt_base + ensemble_id * ensemble_stride;
    
    // Parameters for boundary_list_total
    int n = n_trees;
    double h = h_list[ensemble_id];
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_base != NULL) ? (out_grads_base + ensemble_id * ensemble_stride) : NULL;
    double* out_grad_h_elem = (out_grad_h != NULL) ? &out_grad_h[ensemble_id] : NULL;
    
    // Initialize outputs
    if (threadIdx.x == 0) {
        *out_total = 0.0;
        if (out_grad_h_elem != NULL) {
            *out_grad_h_elem = 0.0;
        }
    }
    // Initialize gradient buffer
    if (out_grads != NULL) {
        int tid = threadIdx.x;
        int max_tid = n_trees * 4;
        for (int idx = tid; idx < n_trees * 3; idx += max_tid) {
            out_grads[idx] = 0.0;
        }
    }
    __syncthreads();
    
    // Call boundary_list_total - it now reads (n_trees, 3) format directly
    boundary_list_total(xyt_ensemble, n, h, out_total, out_grads, out_grad_h_elem);
}

// Compute boundary distance cost for a single tree
// Returns the maximum squared distance of any vertex outside the boundary
__device__ double boundary_distance_tree(
    const double3 pose,
    const double3 h,      // h.x = boundary size, h.y = x_offset, h.z = y_offset
    double3* d_pose,      // output: gradient w.r.t. pose (can be NULL)
    double3* d_h)         // output: gradient w.r.t. h (can be NULL)
{
    double half = h.x * 0.5;
    double offset_x = h.y;
    double offset_y = h.z;
    
    // Get number of vertices from constant memory
    int n_verts = const_n_tree_vertices;
    
    // Precompute rotation matrix
    double c = 0.0;
    double s = 0.0;
    sincos(pose.z, &s, &c);
    
    // Find maximum squared distance over all vertices
    double max_dist_sq = 0.0;
    int max_idx = 0;
    
    for (int v = 0; v < n_verts; ++v) {
        // Load vertex from constant memory
        double vx_local = const_tree_vertices_xy[v * 2 + 0];
        double vy_local = const_tree_vertices_xy[v * 2 + 1];
        
        // Apply rotation and translation, then subtract offsets
        double vx = c * vx_local - s * vy_local + pose.x - offset_x;
        double vy = s * vx_local + c * vy_local + pose.y - offset_y;
        
        // Compute distance outside boundary
        double abs_vx = fabs(vx);
        double abs_vy = fabs(vy);
        double dx = (abs_vx > half) ? (abs_vx - half) : 0.0;
        double dy = (abs_vy > half) ? (abs_vy - half) : 0.0;
        double dist_sq = dx * dx + dy * dy;
        
        if (dist_sq > max_dist_sq) {
            max_dist_sq = dist_sq;
            max_idx = v;
        }
    }
    
    // Compute gradients if requested
    if (d_pose != NULL || d_h != NULL) {
        // Recompute the max vertex transformed coordinates
        double vx_local = const_tree_vertices_xy[max_idx * 2 + 0];
        double vy_local = const_tree_vertices_xy[max_idx * 2 + 1];
        
        double vx_max = c * vx_local - s * vy_local + pose.x - offset_x;
        double vy_max = s * vx_local + c * vy_local + pose.y - offset_y;
        
        double abs_vx_max = fabs(vx_max);
        double abs_vy_max = fabs(vy_max);
        double dx_max = (abs_vx_max > half) ? (abs_vx_max - half) : 0.0;
        double dy_max = (abs_vy_max > half) ? (abs_vy_max - half) : 0.0;
        
        // Compute grad_vx_max and grad_vy_max (needed for both d_pose and d_h)
        // Analytical gradients
        // d(dist_sq)/d(vx) = 2*dx*sign(vx) if dx > 0 else 0
        // d(dist_sq)/d(vy) = 2*dy*sign(vy) if dy > 0 else 0
        double grad_vx_max = (dx_max > 0.0) ? 2.0 * dx_max * ((vx_max >= 0.0) ? 1.0 : -1.0) : 0.0;
        double grad_vy_max = (dy_max > 0.0) ? 2.0 * dy_max * ((vy_max >= 0.0) ? 1.0 : -1.0) : 0.0;
        
        if (d_pose != NULL) {
            // d(vx)/d(x) = 1, d(vy)/d(y) = 1
            d_pose->x = grad_vx_max;
            d_pose->y = grad_vy_max;
            
            // d(vx)/d(theta) = -sin(theta)*vx_local - cos(theta)*vy_local
            // d(vy)/d(theta) = cos(theta)*vx_local - sin(theta)*vy_local
            double dvx_dtheta = -s * vx_local - c * vy_local;
            double dvy_dtheta = c * vx_local - s * vy_local;
            
            d_pose->z = grad_vx_max * dvx_dtheta + grad_vy_max * dvy_dtheta;
        }
        
        if (d_h != NULL) {
            // d(dist_sq)/d(h.x) = d(dist_sq)/d(half) * d(half)/d(h.x)
            // d(dist_sq)/d(half) = -2*dx - 2*dy (chain rule through abs)
            // d(half)/d(h.x) = 0.5
            d_h->x = -(dx_max + dy_max);
            
            // d(dist_sq)/d(offset_x) = -d(dist_sq)/d(vx) = -grad_vx_max
            // d(dist_sq)/d(offset_y) = -d(dist_sq)/d(vy) = -grad_vy_max
            d_h->y = -grad_vx_max;
            d_h->z = -grad_vy_max;
        }
    }
    
    return max_dist_sq;
}

// Compute total boundary distance cost for a list of trees
// Uses 1 thread per tree
__device__ void boundary_distance_list_total(
    const double* __restrict__ xyt_Nx3,  // flattened: [n, 3] in C-contiguous layout
    const int n,
    const double3 h,                     // h.x = boundary size, h.y = x_offset, h.z = y_offset
    double* __restrict__ out_total,
    double* __restrict__ out_grads,      // if non-NULL, write gradients [n*3]
    double* __restrict__ out_grad_h)     // if non-NULL, write gradient w.r.t. h [3]
{
    // Thread organization: 1 thread per tree
    int tree_idx = threadIdx.x;
    
    double local_cost = 0.0;
    
    if (tree_idx < n) {
        // Read pose with strided access: [tree_idx, component]
        double3 pose;
        pose.x = xyt_Nx3[tree_idx * 3 + 0];
        pose.y = xyt_Nx3[tree_idx * 3 + 1];
        pose.z = xyt_Nx3[tree_idx * 3 + 2];
        
        // Compute gradients if requested
        double3 d_pose;
        double3 d_h_local;
        
        // Each thread computes boundary distance cost for one tree with gradients
        local_cost = boundary_distance_tree(pose, h, 
                                             (out_grads != NULL) ? &d_pose : NULL,
                                             (out_grad_h != NULL) ? &d_h_local : NULL);
        
        // Atomic add for total cost
        atomicAdd(out_total, local_cost);
        
        // Write gradients if computed
        if (out_grads != NULL) {
            out_grads[tree_idx * 3 + 0] = d_pose.x;
            out_grads[tree_idx * 3 + 1] = d_pose.y;
            out_grads[tree_idx * 3 + 2] = d_pose.z;
        }
        
        if (out_grad_h != NULL) {
            // All threads contribute to d/dh
            atomicAdd(&out_grad_h[0], d_h_local.x);
            atomicAdd(&out_grad_h[1], d_h_local.y);
            atomicAdd(&out_grad_h[2], d_h_local.z);
        }
    }
}

// Multi-ensemble kernel for boundary distance: one block per ensemble
// Each block processes one ensemble by calling boundary_distance_list_total
//
// Accepts single 3D array with strided access - no transpose needed
// Parameters:
//   xyt_base: base pointer to 3D array [num_ensembles, n_trees, 3] in C-contiguous layout
//   n_trees: number of trees per ensemble (same for all)
//   h_list: array of h values (boundary size+offsets) for each ensemble [num_ensembles, 3]
//   out_totals: array of output totals, one per ensemble
//   out_grads_base: base pointer to gradient output [num_ensembles, n_trees, 3] (can be NULL)
//   out_grad_h: array of h gradients [num_ensembles, 3] (can be NULL)
//   num_ensembles: number of ensembles to process
__global__ void multi_boundary_distance_list_total(
    const double* __restrict__ xyt_base,       // base pointer to [num_ensembles, n_trees, 3]
    const int n_trees,                          // number of trees per ensemble
    const double* __restrict__ h_list,         // [num_ensembles, 3]
    double* __restrict__ out_totals,           // [num_ensembles]
    double* __restrict__ out_grads_base,       // base pointer to [num_ensembles, n_trees, 3] (NULL allowed)
    double* __restrict__ out_grad_h,           // [num_ensembles, 3] (NULL allowed)
    const int num_ensembles)
{
    int ensemble_id = blockIdx.x;
    
    if (ensemble_id >= num_ensembles) {
        return;  // Extra blocks beyond num_ensembles
    }
    
    // Calculate offset for this ensemble's data using strided access
    // Layout: [num_ensembles, n_trees, 3]
    // Stride: each ensemble is n_trees*3 elements apart
    int ensemble_stride = n_trees * 3;
    const double* xyt_ensemble = xyt_base + ensemble_id * ensemble_stride;
    
    // Parameters for boundary_distance_list_total
    int n = n_trees;
    double3 h;
    h.x = h_list[ensemble_id * 3 + 0];
    h.y = h_list[ensemble_id * 3 + 1];
    h.z = h_list[ensemble_id * 3 + 2];
    double* out_total = &out_totals[ensemble_id];
    double* out_grads = (out_grads_base != NULL) ? (out_grads_base + ensemble_id * ensemble_stride) : NULL;
    double* out_grad_h_elem = (out_grad_h != NULL) ? &out_grad_h[ensemble_id * 3] : NULL;
    
    // Initialize outputs
    if (threadIdx.x == 0) {
        *out_total = 0.0;
        if (out_grad_h_elem != NULL) {
            out_grad_h_elem[0] = 0.0;
            out_grad_h_elem[1] = 0.0;
            out_grad_h_elem[2] = 0.0;
        }
    }
    // Initialize gradient buffer
    if (out_grads != NULL) {
        int tid = threadIdx.x;
        for (int idx = tid; idx < n_trees * 3; idx += n_trees) {
            out_grads[idx] = 0.0;
        }
    }
    __syncthreads();
    
    // Call boundary_distance_list_total - it reads (n_trees, 3) format directly
    boundary_distance_list_total(xyt_ensemble, n, h, out_total, out_grads, out_grad_h_elem);
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
_multi_boundary_distance_list_total_kernel: cp.RawKernel | None = None

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

        coords = np.asarray(poly_ccw.exterior.coords[:-1], dtype=kgs.dtype_np)  # drop closing vertex
        nverts = coords.shape[0]

        if nverts < 3:
            raise ValueError(f"Convex piece {idx} has < 3 vertices (n={nverts}).")
        if nverts > MAX_VERTS_PER_PIECE:
            raise ValueError(
                f"Convex piece {idx} has {nverts} vertices, but MAX_VERTS_PER_PIECE={MAX_VERTS_PER_PIECE}."
            )

        # Pad to MAX_VERTS_PER_PIECE by repeating the last vertex so the GPU
        # layout is fixed-size for each piece.
        padded = np.empty((MAX_VERTS_PER_PIECE, 2), dtype=kgs.dtype_np)
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
    global _num_pieces, _raw_module, _multi_overlap_list_total_kernel, _multi_boundary_list_total_kernel, _multi_boundary_distance_list_total_kernel

    if _initialized:
        return

    piece_xy_flat, piece_nverts = _build_convex_piece_arrays(CONVEX_PIECES)

    _num_pieces = piece_nverts.shape[0]

    # Keep host copies for reference (not used by kernels anymore)
    _piece_xy_d = cp.asarray(piece_xy_flat, dtype=kgs.dtype_cp)
    _piece_nverts_d = cp.asarray(piece_nverts, dtype=cp.int32)

    # Persist the CUDA source to a stable .cu file inside kgs.temp_dir
    # and compile from that file so profilers can correlate source lines.
    persist_dir = os.fspath(kgs.temp_dir)
    persist_path = os.path.join(persist_dir, 'pack_cuda_saved.cu')

    # Perform search-replace to switch between float32 and float64
    cuda_src = _CUDA_SRC
    if kgs.USE_FLOAT32:
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
    
    # # First compile to cubin to get ptxas verbose output (ptxas only runs for cubin, not ptx)
    # cubin_path = os.path.join(persist_dir, 'pack_cuda_saved.cubin')
    # cmd_cubin = [nvcc_path, "-O3", "-use_fast_math", "--ptxas-options=-v", "-arch=sm_89", "-cubin", persist_path, "-o", cubin_path]
    
    # print("=== Running NVCC Compilation (cubin for ptxas info) ===")
    # print(f"Command: {' '.join(cmd_cubin)}")
    # proc = subprocess.run(cmd_cubin, text=True)
    
    # if proc.returncode != 0:
    #     raise RuntimeError(f"nvcc cubin compilation failed (exit {proc.returncode})")
    
    # Now compile to PTX for actual use
    cmd = [nvcc_path, "-O3", "-use_fast_math", "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]
    proc = subprocess.run(cmd, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (exit {proc.returncode})")

    # Load compiled PTX into a CuPy RawModule
    _raw_module = cp.RawModule(path=ptx_path)
    #_raw_module = cp.RawModule(code=_CUDA_SRC, backend='nvcc', options=())
    _multi_overlap_list_total_kernel = _raw_module.get_function("multi_overlap_list_total")
    _multi_boundary_list_total_kernel = _raw_module.get_function("multi_boundary_list_total")
    _multi_boundary_distance_list_total_kernel = _raw_module.get_function("multi_boundary_distance_list_total")

    # Copy polygon data to constant memory (cached on-chip, broadcast to all threads)
    # Convert to appropriate dtype if using float32
    if kgs.USE_FLOAT32:
        piece_xy_flat_device = piece_xy_flat.astype(kgs.dtype_np)
    else:
        piece_xy_flat_device = piece_xy_flat
    
    const_piece_xy_ptr = _raw_module.get_global('const_piece_xy')
    const_piece_nverts_ptr = _raw_module.get_global('const_piece_nverts')
    # Use memcpyHtoD to copy to device constant memory
    cp.cuda.runtime.memcpy(const_piece_xy_ptr.ptr, piece_xy_flat_device.ctypes.data, piece_xy_flat_device.nbytes, cp.cuda.runtime.memcpyHostToDevice)
    cp.cuda.runtime.memcpy(const_piece_nverts_ptr.ptr, piece_nverts.ctypes.data, piece_nverts.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    # Copy tree vertices to constant memory for boundary distance computation
    # Get tree vertices from kgs.tree_vertices (should be CuPy array on GPU)
    tree_verts = kgs.tree_vertices.get()  # Convert to NumPy for copying to constant memory
    tree_verts_flat = tree_verts.ravel()  # Flatten (n_vertices, 2) to 1D array
    n_tree_verts = tree_verts.shape[0]
    
    tree_verts_flat_device = tree_verts_flat.astype(kgs.dtype_np)
    
    const_tree_vertices_xy_ptr = _raw_module.get_global('const_tree_vertices_xy')
    const_n_tree_vertices_ptr = _raw_module.get_global('const_n_tree_vertices')
    cp.cuda.runtime.memcpy(const_tree_vertices_xy_ptr.ptr, tree_verts_flat_device.ctypes.data, tree_verts_flat_device.nbytes, cp.cuda.runtime.memcpyHostToDevice)
    n_tree_verts_np = np.array([n_tree_verts], dtype=np.int32)
    cp.cuda.runtime.memcpy(const_n_tree_vertices_ptr.ptr, n_tree_verts_np.ctypes.data, n_tree_verts_np.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    _initialized = True


# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------


def overlap_multi_ensemble(xyt1: cp.ndarray, xyt2: cp.ndarray, use_separation: bool, out_cost: cp.ndarray = None, out_grads: cp.ndarray | None = None, crystal_axes: cp.ndarray | None = None, only_self_interactions: bool = False, stream: cp.cuda.Stream | None = None):
    """Compute total overlap sum for multiple ensembles in parallel.

    Parameters
    ----------
    xyt1 : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays for first set of trees. Must be C-contiguous and correct dtype.
    xyt2 : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays for second set of trees. Must be C-contiguous and correct dtype.
    use_separation : bool
        If True, compute separation-based cost (sum of sep^2) instead of overlap area.
        This flag is required (no default) and is propagated to the CUDA kernel.
    out_cost : cp.ndarray, shape (n_ensembles,)
        Preallocated array for output costs. Must be provided.
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
    crystal_axes : cp.ndarray, shape (n_ensembles, 4), optional
        Crystal axes for periodic boundary conditions. Each row contains [ax, ay, bx, by]
        where (ax, ay) is axis A and (bx, by) is axis B. If provided, each tree will be
        compared against a 3x3 grid of periodic images. If None, no periodic boundaries.
    only_self_interactions : bool, optional
        If True, only compute cost/gradients for self-interactions (tree with its own
        periodic copies). Returns 0 if crystal_axes is None. Default is False.
        Note: Gradients are NOT computed for self-interactions regardless of this flag.
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution. If None, uses default stream.
    """
    _ensure_initialized()

    num_ensembles = xyt1.shape[0]
    n_trees = xyt1.shape[1]
    dtype = xyt1.dtype

    if num_ensembles == 0:
        return

    if kgs.debugging_mode >= 2:
        # Determine expected dtype based on kgs.USE_FLOAT32
        expected_dtype = kgs.dtype_cp if kgs.USE_FLOAT32 else kgs.dtype_cp

        # Assert inputs are 3D arrays
        if xyt1.ndim != 3 or xyt1.shape[2] != 3:
            raise ValueError(f"xyt1 must be shape (n_ensembles, n_trees, 3), got {xyt1.shape}")
        if xyt2.ndim != 3 or xyt2.shape[2] != 3:
            raise ValueError(f"xyt2 must be shape (n_ensembles, n_trees, 3), got {xyt2.shape}")

        # Assert correct dtype
        if xyt1.dtype != expected_dtype:
            raise ValueError(f"xyt1 must have dtype {expected_dtype}, got {xyt1.dtype}")
        if xyt2.dtype != expected_dtype:
            raise ValueError(f"xyt2 must have dtype {expected_dtype}, got {xyt2.dtype}")

        # Assert they are contiguous
        if not xyt1.flags.c_contiguous:
            raise ValueError("xyt1 must be C-contiguous")
        if not xyt2.flags.c_contiguous:
            raise ValueError("xyt2 must be C-contiguous")

        # Validate matching dimensions
        if xyt1.shape[0] != xyt2.shape[0]:
            raise ValueError(f"xyt1 and xyt2 must have same number of ensembles: {xyt1.shape[0]} vs {xyt2.shape[0]}")
        if xyt1.shape[1] != xyt2.shape[1]:
            raise ValueError(f"xyt1 and xyt2 must have same number of trees: {xyt1.shape[1]} vs {xyt2.shape[1]}")

        # Assert outputs are provided
        if out_cost is None:
            raise ValueError("out_cost must be provided")
        # Validate use_separation flag
        if not isinstance(use_separation, (bool, np.bool_)):
            raise ValueError("use_separation must be a boolean")

        # Validate crystal_axes if provided
        if crystal_axes is not None:
            if crystal_axes.ndim != 2 or crystal_axes.shape[1] != 4:
                raise ValueError(f"crystal_axes must be shape (n_ensembles, 4), got {crystal_axes.shape}")
            if crystal_axes.shape[0] != num_ensembles:
                raise ValueError(f"crystal_axes must have {num_ensembles} rows, got {crystal_axes.shape[0]}")
            if crystal_axes.dtype != dtype:
                raise ValueError(f"crystal_axes must have dtype {dtype}, got {crystal_axes.dtype}")
            if not crystal_axes.flags.c_contiguous:
                raise ValueError("crystal_axes must be C-contiguous")

        # Validate output array shapes and types
        if out_cost.shape != (num_ensembles,):
            raise ValueError(f"out_cost must have shape ({num_ensembles},), got {out_cost.shape}")
        if out_grads is not None:
            if out_grads.shape != (num_ensembles, n_trees, 3):
                raise ValueError(f"out_grads must have shape ({num_ensembles}, {n_trees}, 3), got {out_grads.shape}")
            if out_grads.dtype != dtype:
                raise ValueError(f"out_grads must have dtype {dtype}, got {out_grads.dtype}")
            if not out_grads.flags.c_contiguous:
                raise ValueError("out_grads must be C-contiguous")
        if out_cost.dtype != dtype:
            raise ValueError(f"out_cost must have dtype {dtype}, got {out_cost.dtype}")
        if not out_cost.flags.c_contiguous:
            raise ValueError("out_cost must be C-contiguous")
    
    # Zero the output arrays
    out_cost[:] = 0
    if out_grads is not None:
        out_grads[:] = 0

    # Launch kernel: one block per ensemble, n_trees * 4 threads per block
    blocks = num_ensembles
    threads_per_block = n_trees * 4

    # Pass null pointers for optional parameters
    out_grads_ptr = out_grads if out_grads is not None else np.intp(0)
    use_crystal = 1 if crystal_axes is not None else 0
    crystal_axes_ptr = crystal_axes if crystal_axes is not None else np.intp(0)

    for _ in range(1):
        _multi_overlap_list_total_kernel(
            (blocks,),
            (threads_per_block,),
            (
                xyt1,
                xyt2,
                np.int32(n_trees),
                out_cost,
                out_grads_ptr,
                np.int32(num_ensembles),
                np.int32(1 if use_separation else 0),
                np.int32(use_crystal),
                crystal_axes_ptr,
                np.int32(1 if only_self_interactions else 0),
            ),
            stream=stream,
        )

def boundary_multi_ensemble(xyt: cp.ndarray, h: cp.ndarray, out_cost: cp.ndarray = None, out_grads: cp.ndarray | None = None, out_grad_h: cp.ndarray | None = None, stream: cp.cuda.Stream | None = None):
    """Compute total boundary violation area for multiple ensembles in parallel.
    
    Parameters
    ----------
    xyt : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays. Must be C-contiguous and correct dtype.
    h : cp.ndarray, shape (n_ensembles,)
        Boundary sizes for each ensemble.
    out_cost : cp.ndarray, shape (n_ensembles,)
        Preallocated array for output costs. Must be provided.
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
    out_grad_h : cp.ndarray, shape (n_ensembles,), optional
        Preallocated array for h gradients. If None, h gradients are not computed.
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution. If None, uses default stream.
    """
    _ensure_initialized()
    
    num_ensembles = xyt.shape[0]
    n_trees = xyt.shape[1]
    dtype = xyt.dtype
    
    if num_ensembles == 0:
        return
    
    if kgs.debugging_mode >= 2:
        # Determine expected dtype based on kgs.USE_FLOAT32
        expected_dtype = kgs.dtype_cp if kgs.USE_FLOAT32 else kgs.dtype_cp
        
        # Assert inputs are correct shape
        if xyt.ndim != 3 or xyt.shape[2] != 3:
            raise ValueError(f"xyt must be shape (n_ensembles, n_trees, 3), got {xyt.shape}")
        if h.ndim != 1:
            raise ValueError(f"h must be 1D array, got shape {h.shape}")
        
        # Assert correct dtype
        if xyt.dtype != expected_dtype:
            raise ValueError(f"xyt must have dtype {expected_dtype}, got {xyt.dtype}")
        if h.dtype != expected_dtype:
            raise ValueError(f"h must have dtype {expected_dtype}, got {h.dtype}")
        
        # Assert contiguous
        if not xyt.flags.c_contiguous:
            raise ValueError("xyt must be C-contiguous")
        
        if h.shape[0] != num_ensembles:
            raise ValueError(f"h must have {num_ensembles} elements, got {h.shape[0]}")
        
        # Assert outputs are provided
        if out_cost is None:
            raise ValueError("out_cost must be provided")
        
        # Validate output array shapes and types
        if out_cost.shape != (num_ensembles,):
            raise ValueError(f"out_cost must have shape ({num_ensembles},), got {out_cost.shape}")
        if out_grads is not None:
            if out_grads.shape != (num_ensembles, n_trees, 3):
                raise ValueError(f"out_grads must have shape ({num_ensembles}, {n_trees}, 3), got {out_grads.shape}")
            if out_grads.dtype != dtype:
                raise ValueError(f"out_grads must have dtype {dtype}, got {out_grads.dtype}")
            if not out_grads.flags.c_contiguous:
                raise ValueError("out_grads must be C-contiguous")
        if out_grad_h is not None:
            if out_grad_h.shape != (num_ensembles,):
                raise ValueError(f"out_grad_h must have shape ({num_ensembles},), got {out_grad_h.shape}")
            if out_grad_h.dtype != dtype:
                raise ValueError(f"out_grad_h must have dtype {dtype}, got {out_grad_h.dtype}")
            if not out_grad_h.flags.c_contiguous:
                raise ValueError("out_grad_h must be C-contiguous")
        if out_cost.dtype != dtype:
            raise ValueError(f"out_cost must have dtype {dtype}, got {out_cost.dtype}")
        if not out_cost.flags.c_contiguous:
            raise ValueError("out_cost must be C-contiguous")
    
    # Zero the output arrays
    out_cost[:] = 0
    if out_grads is not None:
        out_grads[:] = 0
    if out_grad_h is not None:
        out_grad_h[:] = 0
    
    # Launch kernel: one block per ensemble, n_trees * 4 threads per block
    blocks = num_ensembles
    threads_per_block = n_trees * 4
    
    # Pass null pointers if gradients are None
    out_grads_ptr = out_grads if out_grads is not None else np.intp(0)
    out_grad_h_ptr = out_grad_h if out_grad_h is not None else np.intp(0)
    
    _multi_boundary_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt,
            np.int32(n_trees),
            h,
            out_cost,
            out_grads_ptr,
            out_grad_h_ptr,
            np.int32(num_ensembles),
        ),
        stream=stream,
    )


def boundary_distance_multi_ensemble(xyt: cp.ndarray, h: cp.ndarray, out_cost: cp.ndarray = None, out_grads: cp.ndarray | None = None, out_grad_h: cp.ndarray | None = None, stream: cp.cuda.Stream | None = None):
    """Compute total boundary distance cost for multiple ensembles in parallel.
    
    For each tree, computes the maximum squared distance of any vertex outside the boundary.
    Sum over all trees in each ensemble.
    
    Parameters
    ----------
    xyt : cp.ndarray, shape (n_ensembles, n_trees, 3)
        Pose arrays. Must be C-contiguous and correct dtype.
    h : cp.ndarray, shape (n_ensembles, 3)
        Boundary parameters for each ensemble: [size, x_offset, y_offset].
    out_cost : cp.ndarray, shape (n_ensembles,)
        Preallocated array for output costs. Must be provided.
    out_grads : cp.ndarray, shape (n_ensembles, n_trees, 3), optional
        Preallocated array for gradients. If None, gradients are not computed.
    out_grad_h : cp.ndarray, shape (n_ensembles, 3), optional
        Preallocated array for h gradients. If None, h gradients are not computed.
    stream : cp.cuda.Stream, optional
        CUDA stream for kernel execution. If None, uses default stream.
    """
    _ensure_initialized()
    
    num_ensembles = xyt.shape[0]
    n_trees = xyt.shape[1]
    dtype = xyt.dtype
    
    if num_ensembles == 0:
        return
    
    if kgs.debugging_mode >= 2:
        # Determine expected dtype based on kgs.USE_FLOAT32
        expected_dtype = kgs.dtype_cp if kgs.USE_FLOAT32 else kgs.dtype_cp
        
        # Assert inputs are correct shape
        if xyt.ndim != 3 or xyt.shape[2] != 3:
            raise ValueError(f"xyt must be shape (n_ensembles, n_trees, 3), got {xyt.shape}")
        if h.ndim != 2 or h.shape[1] != 3:
            raise ValueError(f"h must be shape (n_ensembles, 3), got {h.shape}")
        
        # Assert correct dtype
        if xyt.dtype != expected_dtype:
            raise ValueError(f"xyt must have dtype {expected_dtype}, got {xyt.dtype}")
        if h.dtype != expected_dtype:
            raise ValueError(f"h must have dtype {expected_dtype}, got {h.dtype}")
        
        # Assert contiguous
        if not xyt.flags.c_contiguous:
            raise ValueError("xyt must be C-contiguous")
        if not h.flags.c_contiguous:
            raise ValueError("h must be C-contiguous")
        
        if h.shape[0] != num_ensembles:
            raise ValueError(f"h must have {num_ensembles} rows, got {h.shape[0]}")
        
        # Assert outputs are provided
        if out_cost is None:
            raise ValueError("out_cost must be provided")
        
        # Validate output array shapes and types
        if out_cost.shape != (num_ensembles,):
            raise ValueError(f"out_cost must have shape ({num_ensembles},), got {out_cost.shape}")
        if out_grads is not None:
            if out_grads.shape != (num_ensembles, n_trees, 3):
                raise ValueError(f"out_grads must have shape ({num_ensembles}, {n_trees}, 3), got {out_grads.shape}")
            if out_grads.dtype != dtype:
                raise ValueError(f"out_grads must have dtype {dtype}, got {out_grads.dtype}")
            if not out_grads.flags.c_contiguous:
                raise ValueError("out_grads must be C-contiguous")
        if out_grad_h is not None:
            if out_grad_h.shape != (num_ensembles, 3):
                raise ValueError(f"out_grad_h must have shape ({num_ensembles}, 3), got {out_grad_h.shape}")
            if out_grad_h.dtype != dtype:
                raise ValueError(f"out_grad_h must have dtype {dtype}, got {out_grad_h.dtype}")
            if not out_grad_h.flags.c_contiguous:
                raise ValueError("out_grad_h must be C-contiguous")
        if out_cost.dtype != dtype:
            raise ValueError(f"out_cost must have dtype {dtype}, got {out_cost.dtype}")
        if not out_cost.flags.c_contiguous:
            raise ValueError("out_cost must be C-contiguous")
    
    # Zero the output arrays
    out_cost[:] = 0
    if out_grads is not None:
        out_grads[:] = 0
    if out_grad_h is not None:
        out_grad_h[:] = 0
    
    # Launch kernel: one block per ensemble, n_trees threads per block (1 thread per tree)
    blocks = num_ensembles
    threads_per_block = n_trees
    
    # Pass null pointers if gradients are None
    out_grads_ptr = out_grads if out_grads is not None else np.intp(0)
    out_grad_h_ptr = out_grad_h if out_grad_h is not None else np.intp(0)
    
    _multi_boundary_distance_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt,
            np.int32(n_trees),
            h,
            out_cost,
            out_grads_ptr,
            out_grad_h_ptr,
            np.int32(num_ensembles),
        ),
        stream=stream,
    )
