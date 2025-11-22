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

_CUDA_SRC = r"""
extern "C" {
#define MAX_PIECES 4
#define MAX_VERTS_PER_PIECE 4
#define MAX_INTERSECTION_VERTS 8
#define MAX_RADIUS """ + str(MAX_RADIUS) + r"""

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

__device__ double overlap_two_trees(
    const double3 a,
    const double3 b,
    const double* __restrict__ piece_xy,
    const int* __restrict__ piece_nverts,
    const int num_pieces)
{
    // Compute overlap area between two trees given their absolute transforms.
    // Tree 1: (a.x, a.y, a.z)
    // Tree 2: (b.x, b.y, b.z)
    
    // Early exit: check if tree centers are too far apart
    double dx = b.x - a.x;
    double dy = b.y - a.y;
    double dist_sq = dx*dx + dy*dy;
    double max_overlap_dist = 2.0 * MAX_RADIUS;
    
    if (dist_sq > max_overlap_dist * max_overlap_dist) {
        return 0.0;  // Trees too far apart to overlap
    }
    
    double c1 = cos(a.z);
    double s1 = sin(a.z);
    double c2 = cos(b.z);
    double s2 = sin(b.z);
    double total = 0.0;

    // Loop over convex pieces of tree 1
    for (int i = 0; i < num_pieces; ++i) {
        int n1 = piece_nverts[i];
        d2 poly1[MAX_VERTS_PER_PIECE];
        double min1_x = 1e30, max1_x = -1e30;
        double min1_y = 1e30, max1_y = -1e30;

        // Load and transform piece with tree 1's pose, compute AABB
        for (int v = 0; v < n1; ++v) {
            int idx = i * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            double x = piece_xy[base + 0];
            double y = piece_xy[base + 1];
            
            // Apply tree 1 transform
            double x1 = c1 * x - s1 * y + a.x;
            double y1 = s1 * x + c1 * y + a.y;
            poly1[v] = make_d2(x1, y1);
            
            // Update AABB
            min1_x = fmin(min1_x, x1);
            max1_x = fmax(max1_x, x1);
            min1_y = fmin(min1_y, y1);
            max1_y = fmax(max1_y, y1);
        }

        // Loop over convex pieces of tree 2
        for (int j = 0; j < num_pieces; ++j) {
            int n2 = piece_nverts[j];
            d2 poly2[MAX_VERTS_PER_PIECE];
            double min2_x = 1e30, max2_x = -1e30;
            double min2_y = 1e30, max2_y = -1e30;

            for (int v = 0; v < n2; ++v) {
                int idx = j * MAX_VERTS_PER_PIECE + v;
                int base = 2 * idx;
                double x = piece_xy[base + 0];
                double y = piece_xy[base + 1];

                // Apply tree 2 transform
                double x2 = c2 * x - s2 * y + b.x;
                double y2 = s2 * x + c2 * y + b.y;
                poly2[v] = make_d2(x2, y2);
                
                // Update AABB
                min2_x = fmin(min2_x, x2);
                max2_x = fmax(max2_x, x2);
                min2_y = fmin(min2_y, y2);
                max2_y = fmax(max2_y, y2);
            }

            // AABB overlap test - early exit if no overlap
            if (max1_x < min2_x || max2_x < min1_x ||
                max1_y < min2_y || max2_y < min1_y) {
                continue;  // No AABB overlap, skip expensive intersection
            }

            total += convex_intersection_area(poly1, n1, poly2, n2);
        }
    }

    return total;
}

// Compute sum of overlap areas between a reference tree `ref` and a list
// of other trees provided as a flattened 3xN array (row-major: row0=x, row1=y, row2=theta).
// Optionally computes gradient via double-sided finite differences.
__device__ double overlap_ref_with_list(
    const double3 ref,
    const double* __restrict__ xyt_3xN, // flattened row-major: 3 rows, N cols
    const int n,
    const double* __restrict__ piece_xy,
    const int* __restrict__ piece_nverts,
    const int num_pieces,
    double* __restrict__ grad_out) // if non-NULL, write gradient to grad_out[3]: [dx, dy, dtheta]
{
    const double eps = 1e-6;
    double sum = 0.0;

    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;

    // Initialize gradients if requested
    if (grad_out != NULL) {
        grad_out[0] = 0.0;
        grad_out[1] = 0.0;
        grad_out[2] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        double3 other;
        other.x = row_x[i];
        other.y = row_y[i];
        other.z = row_t[i];

        if (!(other.x == ref.x && other.y == ref.y && other.z == ref.z)) {
            double overlap = overlap_two_trees(ref, other, piece_xy, piece_nverts, num_pieces);
            sum += overlap;

            // Compute gradient if requested
            if (grad_out != NULL && overlap > 0.0) {
                // Gradient w.r.t. ref.x
                double3 ref_px = ref; ref_px.x += eps;
                double3 ref_mx = ref; ref_mx.x -= eps;
                double f_px = overlap_two_trees(ref_px, other, piece_xy, piece_nverts, num_pieces);
                double f_mx = overlap_two_trees(ref_mx, other, piece_xy, piece_nverts, num_pieces);
                grad_out[0] += (f_px - f_mx) / (2.0 * eps);

                // Gradient w.r.t. ref.y
                double3 ref_py = ref; ref_py.y += eps;
                double3 ref_my = ref; ref_my.y -= eps;
                double f_py = overlap_two_trees(ref_py, other, piece_xy, piece_nverts, num_pieces);
                double f_my = overlap_two_trees(ref_my, other, piece_xy, piece_nverts, num_pieces);
                grad_out[1] += (f_py - f_my) / (2.0 * eps);

                // Gradient w.r.t. ref.z (theta)
                double3 ref_pz = ref; ref_pz.z += eps;
                double3 ref_mz = ref; ref_mz.z -= eps;
                double f_pz = overlap_two_trees(ref_pz, other, piece_xy, piece_nverts, num_pieces);
                double f_mz = overlap_two_trees(ref_mz, other, piece_xy, piece_nverts, num_pieces);
                grad_out[2] += (f_pz - f_mz) / (2.0 * eps);
            }
        }
    }

    return sum;
}

// Sum overlaps for each tree in the provided list against the entire list.
// This does NOT exploit symmetry; each unordered pair will be counted twice
// (ref vs other, and other vs ref).
// Computes gradients for all trees if out_grads is non-NULL.
__device__ void overlap_list_total(
    const double* __restrict__ xyt_3xN,
    const int n,
    const double* __restrict__ piece_xy,
    const int* __restrict__ piece_nverts,
    const int num_pieces,
    double* __restrict__ out_total,
    double* __restrict__ out_grads) // if non-NULL, write gradients to out_grads[n*3]
{
    // Single-block version: assume gridDim.x == 1 and no cross-block striding.
    // Each thread computes at most one reference index (its threadIdx.x) and
    // we reduce across the block into shared memory, then thread 0 writes the
    // final total to out_total[0]. This avoids atomics.
    int tid = threadIdx.x;
    const double* row_x = xyt_3xN + 0 * n;
    const double* row_y = xyt_3xN + 1 * n;
    const double* row_t = xyt_3xN + 2 * n;

    double local_sum = 0.0;
    double local_grad[3] = {0.0, 0.0, 0.0};

    if (tid < n) {
        double3 ref;
        ref.x = row_x[tid];
        ref.y = row_y[tid];
        ref.z = row_t[tid];

        local_sum = overlap_ref_with_list(ref, xyt_3xN, n, piece_xy, piece_nverts, num_pieces, 
                                          out_grads != NULL ? local_grad : NULL);
        
        // Write per-tree gradient to output
        if (out_grads != NULL) {
            out_grads[tid * 3 + 0] = local_grad[0];
            out_grads[tid * 3 + 1] = local_grad[1];
            out_grads[tid * 3 + 2] = local_grad[2];
        }
    }

    // Shared-memory reduction (assumes blockDim.x <= 1024)
    __shared__ double sdata[1024];
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce only the first n elements (not blockDim.x)
    for (int stride = 1; stride < n; stride *= 2) {
        int index = 2 * stride * tid;
        if (index + stride < n) {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_total[0] = sdata[0]/2; // each pair counted twice
    }
    
}

// Kernel: one thread per reference element computes sum(ref vs all others)
__global__ void overlap_list_total_kernel(
    const double* __restrict__ xyt_3xN, // flattened row-major: 3 rows, N cols
    const int n,
    const double* __restrict__ piece_xy,
    const int* __restrict__ piece_nverts,
    const int num_pieces,
    double* __restrict__ out_total,
    double* __restrict__ out_grads) // if non-NULL, write gradients
{
    // Delegate to the threaded device helper which accumulates
    // per-thread partial sums into out_total[0] and computes gradients.
    overlap_list_total(xyt_3xN, n, piece_xy, piece_nverts, num_pieces, out_total, out_grads);
}

__global__ void overlap_two_trees_kernel(
    const double tx1, const double ty1, const double th1,
    const double tx2, const double ty2, const double th2,
    const double* __restrict__ piece_xy,      // length num_pieces * MAX_VERTS_PER_PIECE * 2
    const int*    __restrict__ piece_nverts,  // length num_pieces
    const int num_pieces,
    double* __restrict__ out_area)
{
    // Accept scalar inputs from the host, but construct double3 PODs
    double3 a; a.x = tx1; a.y = ty1; a.z = th1;
    double3 b; b.x = tx2; b.y = ty2; b.z = th2;

    out_area[0] = overlap_two_trees(a, b, piece_xy, piece_nverts, num_pieces);
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
_overlap_two_trees_kernel: cp.RawKernel | None = None
_overlap_list_total_kernel: cp.RawKernel | None = None

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
    - Uploads the convex pieces to device memory.
    - Compiles the CUDA source and fetches the overlap kernel.

    Subsequent calls are no-ops, so you can safely call this at the start
    of public API functions.
    """
    global _initialized, _piece_xy_d, _piece_nverts_d
    global _num_pieces, _raw_module, _overlap_two_trees_kernel, _overlap_list_total_kernel

    if _initialized:
        return

    piece_xy_flat, piece_nverts = _build_convex_piece_arrays(CONVEX_PIECES)

    _num_pieces = piece_nverts.shape[0]

    _piece_xy_d = cp.asarray(piece_xy_flat, dtype=cp.float64)
    _piece_nverts_d = cp.asarray(piece_nverts, dtype=cp.int32)

    _raw_module = cp.RawModule(code=_CUDA_SRC, options=("--std=c++11",))
    _overlap_two_trees_kernel = _raw_module.get_function("overlap_two_trees_kernel")
    _overlap_list_total_kernel = _raw_module.get_function("overlap_list_total_kernel")

    _initialized = True


# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------

def overlap_two_trees(xyt1, xyt2) -> cp.ndarray:
    """Compute overlap area between two poses of the SAME polygon on the GPU.

    Parameters
    ----------
    xyt1, xyt2 : array-like (length 3)
        Each is a sequence of (x, y, theta) for a single pose. Matrices/batches
        are not supported by this function; pass single 3-element sequences.

    Returns
    -------
    area : cp.float64
        A scalar CuPy float containing the computed overlap area.
    """
    _ensure_initialized()

    # Convert to sequences and validate length
    a = list(xyt1)
    b = list(xyt2)
    if len(a) != 3 or len(b) != 3:
        raise ValueError("xyt1 and xyt2 must be length-3 sequences: (x, y, theta)")

    areas_arr = cp.empty(1, dtype=cp.float64)

    threads_per_block = 1
    blocks = 1

    _overlap_two_trees_kernel(
        (blocks,),
        (threads_per_block,),
        (
            float(a[0]), float(a[1]), float(a[2]),
            float(b[0]), float(b[1]), float(b[2]),
            _piece_xy_d,
            _piece_nverts_d,
            np.int32(_num_pieces),
            areas_arr,
        ),
    )

    return areas_arr[0]


def overlap_list_total(xyt, compute_grad: bool = True):
    """Compute total (non-symmetric) overlap sum for a list of poses.

    Parameters
    ----------
    xyt : array-like, shape (N,3)
        List of poses (x, y, theta).
    compute_grad : bool, optional
        If True, compute and return gradients. Default is True.

    Returns
    -------
    total : float
        Sum over all reference elements of sum(overlap(ref, other)). Each
        unordered pair is counted twice (ref vs other, and other vs ref).
    grads : cp.ndarray, shape (N,3), optional
        Gradients with respect to each input pose (x, y, theta).
        Only returned if compute_grad=True.
    """
    _ensure_initialized()

    xyt_arr = cp.asarray(xyt, dtype=cp.float64)
    if xyt_arr.ndim != 2 or xyt_arr.shape[1] != 3:
        raise ValueError("xyt must be shape (N,3)")

    n = int(xyt_arr.shape[0])

    # Flatten to 3xN row-major so rows are x,y,theta
    xyt_3xN = cp.ascontiguousarray(xyt_arr.T).ravel()

    # Allocate a single-element output to hold the accumulated total
    out_total = cp.zeros(1, dtype=cp.float64)
    
    # Allocate gradient output if requested
    out_grads = cp.zeros(n * 3, dtype=cp.float64) if compute_grad else None

    threads_per_block = n
    blocks = 1

    _overlap_list_total_kernel(
        (blocks,),
        (threads_per_block,),
        (
            xyt_3xN,
            np.int32(n),
            _piece_xy_d,
            _piece_nverts_d,
            np.int32(_num_pieces),
            out_total,
            out_grads if out_grads is not None else cp.cuda.memory.MemoryPointer(cp.cuda.UnownedMemory(0, 0, None), 0),
        ),
    )

    # Ensure kernel finished and results are visible
    cp.cuda.Stream.null.synchronize()

    total = float(out_total[0])
    
    if compute_grad:
        # Reshape gradients back to (N, 3)
        grads = out_grads.reshape(n, 3)
        return total, grads
    else:
        return total

