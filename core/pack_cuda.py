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

typedef struct {
    double x;
    double y;
} d2;

__device__ __forceinline__ d2 make_d2(double x, double y) {
    d2 p; p.x = x; p.y = y; return p;
}

__device__ __forceinline__ double cross3(d2 a, d2 b, d2 c) {
    // Oriented area of triangle (a, b, c) = cross(b-a, c-a)
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ __forceinline__ d2 line_intersection(d2 p1, d2 p2,
                                                d2 q1, d2 q2)
{
    // Solve p1 + t*(p2-p1) == q1 + u*(q2-q1)
    double rx = p2.x - p1.x;
    double ry = p2.y - p1.y;
    double sx = q2.x - q1.x;
    double sy = q2.y - q1.y;

    double denom = rx * sy - ry * sx;

    // Assume denom != 0 for non-parallel lines; for robustness you could
    // add an epsilon check here.
    double t = ((q1.x - p1.x) * sy - (q1.y - p1.y) * sx) / denom;

    return make_d2(p1.x + t * rx, p1.y + t * ry);
}

__device__ __forceinline__ int clip_against_edge(
    const d2* in_pts, int in_count,
    d2 A, d2 B,
    d2* out_pts)
{
    // Clip a convex polygon "in_pts" against the half-plane defined by
    // the directed edge A->B: keep points on the left side (cross >= 0).
    if (in_count == 0) return 0;

    int out_count = 0;

    d2 S = in_pts[in_count - 1];
    double S_side = cross3(A, B, S);

    for (int i = 0; i < in_count; ++i) {
        d2 E = in_pts[i];
        double E_side = cross3(A, B, E);

        bool S_inside = (S_side >= 0.0);
        bool E_inside = (E_side >= 0.0);

        if (S_inside && E_inside) {
            // keep E
            out_pts[out_count++] = E;
        } else if (S_inside && !E_inside) {
            // leaving - keep intersection only
            out_pts[out_count++] = line_intersection(S, E, A, B);
        } else if (!S_inside && E_inside) {
            // entering - add intersection then E
            out_pts[out_count++] = line_intersection(S, E, A, B);
            out_pts[out_count++] = E;
        }
        // else both outside -> keep nothing

        S = E;
        S_side = E_side;
    }

    return out_count;
}

__device__ __forceinline__ double polygon_area(const d2* v, int n) {
    // Signed polygon area via the shoelace formula (absolute value returned).
    if (n < 3) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        sum += v[i].x * v[j].y - v[j].x * v[i].y;
    }
    double a = 0.5 * sum;
    return a >= 0.0 ? a : -a;
}

// ============================================================================
// BACKWARD PASS PRIMITIVES
// ============================================================================

// Backward for polygon_area: given d_area (gradient w.r.t. area output),
// compute gradients w.r.t. each vertex (x,y) coordinate.
// Shoelace: A = 0.5 * sum_i (x_i*y_{i+1} - x_{i+1}*y_i)
// ∂A/∂x_k = 0.5 * (y_{k+1} - y_{k-1})
// ∂A/∂y_k = 0.5 * (x_{k-1} - x_{k+1})
__device__ __forceinline__ void backward_polygon_area(
    const d2* v, int n,
    double d_area,
    d2* d_v)  // output: gradients w.r.t each vertex
{
    if (n < 3) {
        for (int i = 0; i < n; ++i) {
            d_v[i] = make_d2(0.0, 0.0);
        }
        return;
    }
    
    // Compute signed area to determine sign
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        sum += v[i].x * v[j].y - v[j].x * v[i].y;
    }
    double signed_area = 0.5 * sum;
    double sign = (signed_area >= 0.0) ? 1.0 : -1.0;
    
    // Gradient of |A| = sign(A) * ∂A/∂v
    double factor = sign * d_area;
    
    for (int k = 0; k < n; ++k) {
        int k_prev = (k - 1 + n) % n;
        int k_next = (k + 1) % n;
        
        d_v[k].x = factor * 0.5 * (v[k_next].y - v[k_prev].y);
        d_v[k].y = factor * 0.5 * (v[k_prev].x - v[k_next].x);
    }
}

// Backward for line_intersection: given d_out (gradient w.r.t. intersection point),
// compute gradients w.r.t. the four input points p1, p2, q1, q2.
// The intersection solves: p1 + t*(p2-p1) = q1 + u*(q2-q1)
// where t = ((q1-p1) × (q2-q1)) / ((p2-p1) × (q2-q1))
// and out = p1 + t*(p2-p1)
__device__ __forceinline__ void backward_line_intersection(
    d2 p1, d2 p2, d2 q1, d2 q2,
    d2 d_out,  // gradient w.r.t. output intersection point
    d2* d_p1, d2* d_p2, d2* d_q1, d2* d_q2)  // output gradients
{
    double rx = p2.x - p1.x;
    double ry = p2.y - p1.y;
    double sx = q2.x - q1.x;
    double sy = q2.y - q1.y;
    
    double denom = rx * sy - ry * sx;
    
    // Numerator for t
    double num = (q1.x - p1.x) * sy - (q1.y - p1.y) * sx;
    double t = num / denom;
    
    // out = p1 + t * (p2 - p1)
    // ∂out/∂p1 = I - t*I + (p2-p1) ⊗ ∂t/∂p1 = (1-t)*I + (p2-p1) ⊗ ∂t/∂p1
    // ∂out/∂p2 = t*I + (p2-p1) ⊗ ∂t/∂p2
    // ∂out/∂q1 = (p2-p1) ⊗ ∂t/∂q1
    // ∂out/∂q2 = (p2-p1) ⊗ ∂t/∂q2
    
    // First compute ∂t/∂(p1,p2,q1,q2)
    // t = num / denom
    // ∂t/∂x = (∂num/∂x * denom - num * ∂denom/∂x) / denom^2
    //
    // IMPORTANT: s = q2 - q1, so when q1 or q2 change, s also changes!
    // ∂s/∂q1 = -I,  ∂s/∂q2 = +I
    
    double denom2 = denom * denom;
    
    // ∂num/∂p1.x = -sy,  ∂num/∂p1.y = sx
    // ∂denom/∂p1.x = -sy (from ∂rx/∂p1.x = -1), ∂denom/∂p1.y = sx (from ∂ry/∂p1.y = -1)
    double dnum_dp1_x = -sy;
    double dnum_dp1_y = sx;
    double ddenom_dp1_x = -sy;
    double ddenom_dp1_y = sx;
    d2 dt_dp1 = make_d2((dnum_dp1_x * denom - num * ddenom_dp1_x) / denom2,
                        (dnum_dp1_y * denom - num * ddenom_dp1_y) / denom2);
    
    // ∂num/∂p2 = 0
    // ∂denom/∂p2.x = sy,  ∂denom/∂p2.y = -sx
    d2 dt_dp2 = make_d2((-num * sy) / denom2, (num * sx) / denom2);
    
    // ∂num/∂q1.x = sy + (q1.y - p1.y) (from ∂sy/∂q1.x = 0 and ∂sx/∂q1.x = -1)
    // ∂num/∂q1.y = -sx - (q1.x - p1.x) (from ∂sy/∂q1.y = -1 and ∂sx/∂q1.y = 0)
    // ∂denom/∂q1.x = ry (from ∂sy/∂q1.x = 0 and ∂sx/∂q1.x = -1)
    // ∂denom/∂q1.y = -rx (from ∂sy/∂q1.y = -1 and ∂sx/∂q1.y = 0)
    double dnum_dq1_x = sy + (q1.y - p1.y);
    double dnum_dq1_y = -sx - (q1.x - p1.x);
    double ddenom_dq1_x = ry;
    double ddenom_dq1_y = -rx;
    d2 dt_dq1 = make_d2((dnum_dq1_x * denom - num * ddenom_dq1_x) / denom2,
                        (dnum_dq1_y * denom - num * ddenom_dq1_y) / denom2);
    
    // ∂num/∂q2.x = -(q1.y - p1.y) (from ∂sy/∂q2.x = 0 and ∂sx/∂q2.x = +1)
    // ∂num/∂q2.y = (q1.x - p1.x) (from ∂sy/∂q2.y = +1 and ∂sx/∂q2.y = 0)
    // ∂denom/∂q2.x = -ry (from ∂sy/∂q2.x = 0 and ∂sx/∂q2.x = +1)
    // ∂denom/∂q2.y = rx (from ∂sy/∂q2.y = +1 and ∂sx/∂q2.y = 0)
    double dnum_dq2_x = -(q1.y - p1.y);
    double dnum_dq2_y = (q1.x - p1.x);
    double ddenom_dq2_x = -ry;
    double ddenom_dq2_y = rx;
    d2 dt_dq2 = make_d2((dnum_dq2_x * denom - num * ddenom_dq2_x) / denom2,
                        (dnum_dq2_y * denom - num * ddenom_dq2_y) / denom2);
    
    // Now compute ∂out/∂inputs using chain rule
    // out = [p1_x + t*rx, p1_y + t*ry]^T where rx = p2_x - p1_x, ry = p2_y - p1_y
    //
    // Jacobian ∂out/∂p1 (accounting for ∂rx/∂p1_x = -1, ∂ry/∂p1_y = -1):
    // ∂out_x/∂p1_x = 1 + t*(-1) + rx * ∂t/∂p1_x = 1 - t + rx * ∂t/∂p1_x
    // ∂out_x/∂p1_y = t*0 + rx * ∂t/∂p1_y = rx * ∂t/∂p1_y
    // ∂out_y/∂p1_x = t*0 + ry * ∂t/∂p1_x = ry * ∂t/∂p1_x
    // ∂out_y/∂p1_y = 1 + t*(-1) + ry * ∂t/∂p1_y = 1 - t + ry * ∂t/∂p1_y
    //
    // Gradient: d_p1 = J^T @ d_out
    d_p1->x = (1.0 - t + rx * dt_dp1.x) * d_out.x + (ry * dt_dp1.x) * d_out.y;
    d_p1->y = (rx * dt_dp1.y) * d_out.x + (1.0 - t + ry * dt_dp1.y) * d_out.y;
    
    // Jacobian ∂out/∂p2 (note: rx = p2_x - p1_x, so ∂rx/∂p2_x = 1, etc.):
    // ∂out_x/∂p2_x = t + rx * ∂t/∂p2_x,  ∂out_x/∂p2_y = rx * ∂t/∂p2_y
    // ∂out_y/∂p2_x = ry * ∂t/∂p2_x,      ∂out_y/∂p2_y = t + ry * ∂t/∂p2_y
    d_p2->x = (t + rx * dt_dp2.x) * d_out.x + (ry * dt_dp2.x) * d_out.y;
    d_p2->y = (rx * dt_dp2.y) * d_out.x + (t + ry * dt_dp2.y) * d_out.y;
    
    // Jacobian ∂out/∂q1:
    // ∂out_x/∂q1_x = rx * ∂t/∂q1_x,  ∂out_x/∂q1_y = rx * ∂t/∂q1_y
    // ∂out_y/∂q1_x = ry * ∂t/∂q1_x,  ∂out_y/∂q1_y = ry * ∂t/∂q1_y
    d_q1->x = (rx * dt_dq1.x) * d_out.x + (ry * dt_dq1.x) * d_out.y;
    d_q1->y = (rx * dt_dq1.y) * d_out.x + (ry * dt_dq1.y) * d_out.y;
    
    // Jacobian ∂out/∂q2:
    // ∂out_x/∂q2_x = rx * ∂t/∂q2_x,  ∂out_x/∂q2_y = rx * ∂t/∂q2_y
    // ∂out_y/∂q2_x = ry * ∂t/∂q2_x,  ∂out_y/∂q2_y = ry * ∂t/∂q2_y
    d_q2->x = (rx * dt_dq2.x) * d_out.x + (ry * dt_dq2.x) * d_out.y;
    d_q2->y = (rx * dt_dq2.y) * d_out.x + (ry * dt_dq2.y) * d_out.y;
}

// Backward for transform: given gradient w.r.t. transformed vertices,
// compute gradients w.r.t. pose (x, y, theta).
// Transform: v_t = R(theta) * v_local + (x, y)
// where R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
__device__ __forceinline__ void backward_transform_vertices(
    const d2* v_local, int n,
    const d2* d_v_transformed,  // gradient w.r.t. transformed vertices
    double c, double s,  // cos(theta), sin(theta)
    double3* d_pose)  // output: gradient w.r.t. (x, y, theta)
{
    d_pose->x = 0.0;
    d_pose->y = 0.0;
    d_pose->z = 0.0;
    
    for (int i = 0; i < n; ++i) {
        // v_t.x = c * v.x - s * v.y + pose.x
        // v_t.y = s * v.x + c * v.y + pose.y
        
        // ∂v_t.x/∂pose.x = 1, ∂v_t.y/∂pose.x = 0
        d_pose->x += d_v_transformed[i].x;
        
        // ∂v_t.x/∂pose.y = 0, ∂v_t.y/∂pose.y = 1
        d_pose->y += d_v_transformed[i].y;
        
        // ∂v_t.x/∂theta = -s * v.x - c * v.y
        // ∂v_t.y/∂theta = c * v.x - s * v.y
        d_pose->z += d_v_transformed[i].x * (-s * v_local[i].x - c * v_local[i].y);
        d_pose->z += d_v_transformed[i].y * (c * v_local[i].x - s * v_local[i].y);
    }
}

// Structure to track clipping metadata for backward pass
struct ClipMetadata {
    int n_out;  // number of output vertices
    // For each output vertex: source type and indices
    // type: 0 = original vertex from input, 1 = intersection point
    int src_type[MAX_INTERSECTION_VERTS];
    int src_idx[MAX_INTERSECTION_VERTS];  // if type==0: index in input; if type==1: edge index
    int src_idx2[MAX_INTERSECTION_VERTS];  // if type==1: next edge vertex index
};

// Enhanced clip_against_edge that also saves metadata for backward
__device__ __forceinline__ int clip_against_edge_with_metadata(
    const d2* in_pts, int in_count,
    d2 A, d2 B,
    d2* out_pts,
    ClipMetadata* meta)
{
    if (in_count == 0) {
        meta->n_out = 0;
        return 0;
    }

    int out_count = 0;
    d2 S = in_pts[in_count - 1];
    double S_side = cross3(A, B, S);
    int S_idx = in_count - 1;

    for (int i = 0; i < in_count; ++i) {
        d2 E = in_pts[i];
        double E_side = cross3(A, B, E);

        bool S_inside = (S_side >= 0.0);
        bool E_inside = (E_side >= 0.0);

        if (S_inside && E_inside) {
            // keep E - original vertex
            out_pts[out_count] = E;
            meta->src_type[out_count] = 0;
            meta->src_idx[out_count] = i;
            out_count++;
        } else if (S_inside && !E_inside) {
            // leaving - keep intersection only
            out_pts[out_count] = line_intersection(S, E, A, B);
            meta->src_type[out_count] = 1;
            meta->src_idx[out_count] = S_idx;
            meta->src_idx2[out_count] = i;
            out_count++;
        } else if (!S_inside && E_inside) {
            // entering - add intersection then E
            out_pts[out_count] = line_intersection(S, E, A, B);
            meta->src_type[out_count] = 1;
            meta->src_idx[out_count] = S_idx;
            meta->src_idx2[out_count] = i;
            out_count++;
            
            out_pts[out_count] = E;
            meta->src_type[out_count] = 0;
            meta->src_idx[out_count] = i;
            out_count++;
        }

        S = E;
        S_side = E_side;
        S_idx = i;
    }

    meta->n_out = out_count;
    return out_count;
}

// Backward for clip_against_edge: distribute gradients from output vertices
// back to input vertices and clip edge vertices
__device__ __forceinline__ void backward_clip_against_edge(
    const d2* in_pts, int in_count,
    d2 A, d2 B,
    const d2* d_out_pts,  // gradient w.r.t. output vertices
    const ClipMetadata* meta,
    d2* d_in_pts,  // output: gradient w.r.t. input vertices
    d2* d_A, d2* d_B)  // output: gradient w.r.t. clip edge
{
    // Initialize gradients to zero
    for (int i = 0; i < in_count; ++i) {
        d_in_pts[i] = make_d2(0.0, 0.0);
    }
    d_A->x = 0.0; d_A->y = 0.0;
    d_B->x = 0.0; d_B->y = 0.0;
    
    // Backpropagate through each output vertex
    for (int i = 0; i < meta->n_out; ++i) {
        if (meta->src_type[i] == 0) {
            // Original vertex - gradient flows directly to input
            int src = meta->src_idx[i];
            d_in_pts[src].x += d_out_pts[i].x;
            d_in_pts[src].y += d_out_pts[i].y;
        } else {
            // Intersection point - backprop through line_intersection
            int idx1 = meta->src_idx[i];
            int idx2 = meta->src_idx2[i];
            d2 S = in_pts[idx1];
            d2 E = in_pts[idx2];
            
            d2 d_S, d_E, d_A_local, d_B_local;
            backward_line_intersection(S, E, A, B, d_out_pts[i],
                                      &d_S, &d_E, &d_A_local, &d_B_local);
            
            d_in_pts[idx1].x += d_S.x;
            d_in_pts[idx1].y += d_S.y;
            d_in_pts[idx2].x += d_E.x;
            d_in_pts[idx2].y += d_E.y;
            d_A->x += d_A_local.x;
            d_A->y += d_A_local.y;
            d_B->x += d_B_local.x;
            d_B->y += d_B_local.y;
        }
    }
}

__device__ __forceinline__ void compute_tree_polys_and_aabbs(
    double3 pose,
    d2 out_polys[MAX_PIECES][MAX_VERTS_PER_PIECE],
    double out_aabb_min_x[MAX_PIECES],
    double out_aabb_max_x[MAX_PIECES],
    double out_aabb_min_y[MAX_PIECES],
    double out_aabb_max_y[MAX_PIECES])
{
    // Precompute transform for this pose
    double c = 0.0;
    double s = 0.0;
    sincos(pose.z, &s, &c);
    
    // Transform each convex piece and compute its AABB
    for (int pi = 0; pi < MAX_PIECES; ++pi) {
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
            out_polys[pi][v] = make_d2(x_t, y_t);
            
            // Update AABB
            min_x = fmin(min_x, x_t);
            max_x = fmax(max_x, x_t);
            min_y = fmin(min_y, y_t);
            max_y = fmax(max_y, y_t);
        }
        
        out_aabb_min_x[pi] = min_x;
        out_aabb_max_x[pi] = max_x;
        out_aabb_min_y[pi] = min_y;
        out_aabb_max_y[pi] = max_y;
    }
}

__device__ double convex_intersection_area(
    const d2* subj, int n_subj,
    const d2* clip, int n_clip)
{
    // Compute the area of intersection between two convex polygons using
    // Sutherland-Hodgman clipping of "subj" against all edges of "clip".
    if (n_subj == 0 || n_clip == 0) return 0.0;

    d2 polyA[MAX_INTERSECTION_VERTS];
    d2 polyB[MAX_INTERSECTION_VERTS];

    int nA = n_subj;
    // Copy subject polygon into work buffer
    for (int i = 0; i < n_subj; ++i) {
        polyA[i] = subj[i];
    }

    // Clip A against each edge of clip polygon
    for (int e = 0; e < n_clip && nA > 0; ++e) {
        d2 A = clip[e];
        d2 B = clip[(e + 1) % n_clip];

        int nB = clip_against_edge(polyA, nA, A, B, polyB);

        // Copy back to polyA for next iteration
        nA = nB;
        for (int i = 0; i < nA; ++i) {
            polyA[i] = polyB[i];
        }
    }

    return polygon_area(polyA, nA);
}

// Backward for convex_intersection_area
// Recomputes forward clipping sequence while tracking metadata,
// then backpropagates through clipping and area computation
__device__ void backward_convex_intersection_area(
    const d2* subj, int n_subj,
    const d2* clip, int n_clip,
    double d_area,  // gradient w.r.t. output area
    d2* d_subj,     // output: gradient w.r.t. subject vertices
    d2* d_clip)     // output: gradient w.r.t. clip vertices
{
    // Initialize output gradients
    for (int i = 0; i < n_subj; ++i) {
        d_subj[i] = make_d2(0.0, 0.0);
    }
    for (int i = 0; i < n_clip; ++i) {
        d_clip[i] = make_d2(0.0, 0.0);
    }
    
    if (n_subj == 0 || n_clip == 0 || d_area == 0.0) return;

    // Forward pass: recompute clipping with metadata AND save intermediate polygons
    d2 forward_polys[MAX_VERTS_PER_PIECE + 1][MAX_INTERSECTION_VERTS];
    int forward_counts[MAX_VERTS_PER_PIECE + 1];
    ClipMetadata metadata[MAX_VERTS_PER_PIECE];  // one per clipping edge
    
    // Initialize with subject polygon
    forward_counts[0] = n_subj;
    for (int i = 0; i < n_subj; ++i) {
        forward_polys[0][i] = subj[i];
    }

    // Apply each clipping edge, save metadata and intermediate results
    int clip_count = 0;
    for (int e = 0; e < n_clip && forward_counts[e] > 0; ++e) {
        d2 A = clip[e];
        d2 B = clip[(e + 1) % n_clip];

        int n_out = clip_against_edge_with_metadata(
            forward_polys[e], forward_counts[e], 
            A, B, 
            forward_polys[e + 1], &metadata[e]);
        forward_counts[e + 1] = n_out;
        clip_count++;
    }

    // Final polygon is in forward_polys[clip_count]
    int final_n = forward_counts[clip_count];
    
    // Backward through polygon_area
    d2 d_polyFinal[MAX_INTERSECTION_VERTS];
    backward_polygon_area(forward_polys[clip_count], final_n, d_area, d_polyFinal);
    
    // Backward through each clipping stage in reverse
    d2 d_current[MAX_INTERSECTION_VERTS];
    for (int i = 0; i < final_n; ++i) {
        d_current[i] = d_polyFinal[i];
    }
    
    for (int e = clip_count - 1; e >= 0; --e) {
        int e_idx = e % n_clip;
        int e_next = (e + 1) % n_clip;
        d2 A = clip[e_idx];
        d2 B = clip[e_next];
        
        d2 d_in[MAX_INTERSECTION_VERTS];
        d2 d_A, d_B;
        
        backward_clip_against_edge(forward_polys[e], forward_counts[e], A, B,
                                   d_current, &metadata[e],
                                   d_in, &d_A, &d_B);
        
        // Accumulate gradients for clip vertices
        d_clip[e_idx].x += d_A.x;
        d_clip[e_idx].y += d_A.y;
        d_clip[e_next].x += d_B.x;
        d_clip[e_next].y += d_B.y;
        
        // Update current gradient for next iteration
        for (int i = 0; i < forward_counts[e]; ++i) {
            d_current[i] = d_in[i];
        }
    }
    
    // After all clipping stages, d_current contains gradients w.r.t. subject
    for (int i = 0; i < n_subj; ++i) {
        d_subj[i] = d_current[i];
    }
}

// Compute sum of overlap areas between a reference tree `ref` and a list
// of other trees provided as a flattened 3xN array (row-major: row0=x, row1=y, row2=theta).
// Always skips comparing ref with identical pose in the other list.
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double* __restrict__ xyt_3xN, // flattened row-major: 3 rows, N cols
    const int n)
{
    const double eps = 1e-6;
    double sum = 0.0;

    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;
    
    // Precompute and cache transformed polygons for ref tree ONCE
    d2 ref_polys[MAX_PIECES][MAX_VERTS_PER_PIECE];
    double ref_aabb_min_x[MAX_PIECES];
    double ref_aabb_max_x[MAX_PIECES];
    double ref_aabb_min_y[MAX_PIECES];
    double ref_aabb_max_y[MAX_PIECES];
    
    compute_tree_polys_and_aabbs(ref, ref_polys, ref_aabb_min_x, ref_aabb_max_x,
                                  ref_aabb_min_y, ref_aabb_max_y);
    
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
        
        // Precompute and cache transformed polygons for other tree
        d2 other_polys[MAX_PIECES][MAX_VERTS_PER_PIECE];
        double other_aabb_min_x[MAX_PIECES];
        double other_aabb_max_x[MAX_PIECES];
        double other_aabb_min_y[MAX_PIECES];
        double other_aabb_max_y[MAX_PIECES];
        
        compute_tree_polys_and_aabbs(other, other_polys, other_aabb_min_x, other_aabb_max_x,
                                      other_aabb_min_y, other_aabb_max_y);
        
        double total = 0.0;

        // Loop over convex pieces of ref tree (now use cached transforms)
        for (int pi = 0; pi < MAX_PIECES; ++pi) {
            int n1 = const_piece_nverts[pi];

            // Loop over convex pieces of other tree
            for (int pj = 0; pj < MAX_PIECES; ++pj) {
                int n2 = const_piece_nverts[pj];

                // AABB overlap test - early exit if no overlap
                if (ref_aabb_max_x[pi] < other_aabb_min_x[pj] || other_aabb_max_x[pj] < ref_aabb_min_x[pi] ||
                    ref_aabb_max_y[pi] < other_aabb_min_y[pj] || other_aabb_max_y[pj] < ref_aabb_min_y[pi]) {
                    continue;  // No AABB overlap, skip expensive intersection
                }

                total += convex_intersection_area(ref_polys[pi], n1, other_polys[pj], n2);
            }
        }
        
        sum += total;
    }

    return sum;
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
    
    if (d_overlap_sum == 0.0) return;

    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;
    
    // Precompute ref tree transformations
    double c_ref = 0.0, s_ref = 0.0;
    sincos(ref.z, &s_ref, &c_ref);
    
    d2 ref_polys[MAX_PIECES][MAX_VERTS_PER_PIECE];
    double ref_aabb_min_x[MAX_PIECES];
    double ref_aabb_max_x[MAX_PIECES];
    double ref_aabb_min_y[MAX_PIECES];
    double ref_aabb_max_y[MAX_PIECES];
    
    compute_tree_polys_and_aabbs(ref, ref_polys, ref_aabb_min_x, ref_aabb_max_x,
                                  ref_aabb_min_y, ref_aabb_max_y);
    
    // Load local piece vertices from constant memory
    d2 ref_local[MAX_PIECES][MAX_VERTS_PER_PIECE];
    for (int pi = 0; pi < MAX_PIECES; ++pi) {
        int n_verts = const_piece_nverts[pi];
        for (int v = 0; v < n_verts; ++v) {
            int idx = pi * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            ref_local[pi][v].x = const_piece_xy[base + 0];
            ref_local[pi][v].y = const_piece_xy[base + 1];
        }
    }
    
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
        
        // Compute other tree transformations
        d2 other_polys[MAX_PIECES][MAX_VERTS_PER_PIECE];
        double other_aabb_min_x[MAX_PIECES];
        double other_aabb_max_x[MAX_PIECES];
        double other_aabb_min_y[MAX_PIECES];
        double other_aabb_max_y[MAX_PIECES];
        
        compute_tree_polys_and_aabbs(other, other_polys, other_aabb_min_x, other_aabb_max_x,
                                      other_aabb_min_y, other_aabb_max_y);
        
        // Loop over all piece pairs and backpropagate
        for (int pi = 0; pi < MAX_PIECES; ++pi) {
            int n1 = const_piece_nverts[pi];

            for (int pj = 0; pj < MAX_PIECES; ++pj) {
                int n2 = const_piece_nverts[pj];

                // AABB overlap test
                if (ref_aabb_max_x[pi] < other_aabb_min_x[pj] || other_aabb_max_x[pj] < ref_aabb_min_x[pi] ||
                    ref_aabb_max_y[pi] < other_aabb_min_y[pj] || other_aabb_max_y[pj] < ref_aabb_min_y[pi]) {
                    continue;
                }

                // Backward through intersection area
                d2 d_ref_poly[MAX_VERTS_PER_PIECE];
                d2 d_other_poly[MAX_VERTS_PER_PIECE];
                
                backward_convex_intersection_area(
                    ref_polys[pi], n1,
                    other_polys[pj], n2,
                    d_overlap_sum,  // gradient flows from output
                    d_ref_poly,
                    d_other_poly);
                
                // Backward through transform for ref piece
                double3 d_ref_pose_piece;
                backward_transform_vertices(
                    ref_local[pi], n1,
                    d_ref_poly,
                    c_ref, s_ref,
                    &d_ref_pose_piece);
                
                d_ref->x += d_ref_pose_piece.x;
                d_ref->y += d_ref_pose_piece.y;
                d_ref->z += d_ref_pose_piece.z;
            }
        }
    }
}

// Sum overlaps between trees in xyt1 and trees in xyt2.
// Each tree in xyt1 is compared against all trees in xyt2.
// Identical poses are automatically skipped.
// Result is divided by 2 since each pair is counted twice (when xyt1 == xyt2).
// Computes gradients for all trees in xyt1 if out_grads is non-NULL.
// When xyt1 == xyt2 (same pointer), also accumulates gradients from the "other" side.
__device__ void overlap_list_total(
    const double* __restrict__ xyt1_3xN,
    const int n1,
    const double* __restrict__ xyt2_3xN,
    const int n2,
    double* __restrict__ out_total,
    double* __restrict__ out_grads) // if non-NULL, write gradients to out_grads[n1*3]
{
    // Single-block version: assume gridDim.x == 1 and no cross-block striding.
    // Each thread computes at most one reference index (its threadIdx.x from xyt1) and
    // we reduce across the block into shared memory, then thread 0 writes the
    // final total to out_total[0]. This avoids atomics.
    int tid = threadIdx.x;

    const double* row_x = xyt1_3xN + 0 * n1;
    const double* row_y = xyt1_3xN + 1 * n1;
    const double* row_t = xyt1_3xN + 2 * n1;

    double local_sum = 0.0;
    double local_grad[3] = {0.0, 0.0, 0.0};

    if (tid < n1) {
        double3 ref;
        ref.x = row_x[tid];
        ref.y = row_y[tid];
        ref.z = row_t[tid];

        // Compute overlap sum
        local_sum = overlap_ref_with_list(ref, xyt2_3xN, n2);

        if (out_grads != NULL) {
            // Compute gradients using analytic backward pass
            double3 d_ref;
            backward_overlap_ref_with_list(ref, xyt2_3xN, n2, 1.0, &d_ref);
            
            local_grad[0] = d_ref.x;
            local_grad[1] = d_ref.y;
            local_grad[2] = d_ref.z;

            // Write per-tree gradient to output
            out_grads[tid * 3 + 0] = local_grad[0];
            out_grads[tid * 3 + 1] = local_grad[1];
            out_grads[tid * 3 + 2] = local_grad[2];
        }
    }

    // Atomic reduction - faster for small thread counts, avoids __syncthreads() overhead
    if (tid < n1) {
        atomicAdd(out_total, local_sum / 2.0);
    }
    
}

// Kernel: one thread per reference element in xyt1 computes sum(ref vs all in xyt2)
__global__ void overlap_list_total_kernel(
    const double* __restrict__ xyt1_3xN, // flattened row-major: 3 rows, N1 cols
    const int n1,
    const double* __restrict__ xyt2_3xN, // flattened row-major: 3 rows, N2 cols
    const int n2,
    double* __restrict__ out_total,
    double* __restrict__ out_grads) // if non-NULL, write gradients for xyt1
{
    // Delegate to the threaded device helper which accumulates
    // per-thread partial sums into out_total[0] and computes gradients.
    // Note: piece data comes from constant memory (const_piece_xy, const_piece_nverts)
    overlap_list_total(xyt1_3xN, n1, xyt2_3xN, n2, out_total, out_grads);
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
__global__ void multi_ensemble_kernel(
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
_overlap_list_total_kernel: cp.RawKernel | None = None
_multi_ensemble_kernel: cp.RawKernel | None = None

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
    global _num_pieces, _raw_module,  _overlap_list_total_kernel, _multi_ensemble_kernel

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
    cmd = [nvcc_path, "-lineinfo", "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]
    cmd = [nvcc_path, "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (exit {proc.returncode}):\n{proc.stderr.decode(errors='ignore')}")

    # Load compiled PTX into a CuPy RawModule
    _raw_module = cp.RawModule(path=ptx_path)
    #_raw_module = cp.RawModule(code=_CUDA_SRC, backend='nvcc', options=())
    _overlap_list_total_kernel = _raw_module.get_function("overlap_list_total_kernel")
    _multi_ensemble_kernel = _raw_module.get_function("multi_ensemble_kernel")

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


def overlap_list_total(xyt1, xyt2, compute_grad: bool = True):
    """Compute total overlap sum between poses in xyt1 and xyt2.

    Parameters
    ----------
    xyt1 : array-like, shape (N1,3)
        First list of poses (x, y, theta).
    xyt2 : array-like, shape (N2,3)
        Second list of poses (x, y, theta).
    compute_grad : bool, optional
        If True, compute and return gradients. Default is True.

    Returns
    -------
    total : float
        Sum of overlap areas between each tree in xyt1 and each tree in xyt2,
        divided by 2 (since each pair is counted twice).
    grads : cp.ndarray, shape (N1,3), optional
        Gradients with respect to each pose in xyt1 (x, y, theta).
        Only returned if compute_grad=True.
    """
    _ensure_initialized()

    # Determine dtype based on USE_FLOAT32 setting
    dtype = cp.float32 if USE_FLOAT32 else cp.float64

    xyt1_arr = cp.asarray(xyt1, dtype=dtype)
    if xyt1_arr.ndim != 2 or xyt1_arr.shape[1] != 3:
        raise ValueError("xyt1 must be shape (N,3)")

    xyt2_arr = cp.asarray(xyt2, dtype=dtype)
    if xyt2_arr.ndim != 2 or xyt2_arr.shape[1] != 3:
        raise ValueError("xyt2 must be shape (N,3)")

    n1 = int(xyt1_arr.shape[0])
    n2 = int(xyt2_arr.shape[0])

    # Flatten to 3xN row-major so rows are x,y,theta
    xyt1_3xN = cp.ascontiguousarray(xyt1_arr.T).ravel()
    xyt2_3xN = cp.ascontiguousarray(xyt2_arr.T).ravel()

    # Allocate a single-element output to hold the accumulated total
    out_total = cp.zeros(1, dtype=dtype)
    
    # Allocate gradient output if requested (gradients are w.r.t. xyt1)
    out_grads = cp.zeros(n1 * 3, dtype=dtype) if compute_grad else None

    threads_per_block = n1
    blocks = 1

    # Dummy empty array to use when gradient output is not requested.
    # Passing an empty CuPy array is safer than constructing an unowned
    # memory pointer (some CuPy versions reject a zero-sized UnownedMemory).
    null_ptr = cp.asarray([], dtype=dtype)

    _overlap_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt1_3xN,
            np.int32(n1),
            xyt2_3xN,
            np.int32(n2),
            out_total,
            out_grads if out_grads is not None else null_ptr,
        ),
    )

    # Ensure kernel finished and results are visible
    # cp.cuda.Stream.null.synchronize()
    
    if compute_grad:
        # Reshape gradients back to (N1, 3)
        grads = out_grads.reshape(n1, 3)
        return out_total, grads
    else:
        return out_total, None


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
    
    # Launch kernel: one block per ensemble, max_n1 threads per block
    blocks = num_ensembles
    threads_per_block = max_n1
    
    # Cast pointer arrays to proper type for kernel
    null_ptr = cp.array([0], dtype=cp.uint64)
    
    _multi_ensemble_kernel(
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


