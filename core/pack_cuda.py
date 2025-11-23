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
    
    // Precompute ref tree transform ONCE (outside the loop over trees)
    double c1 = 0.0;
    double s1 = 0.0;
    sincos(ref.z, &s1, &c1);
    
    // Precompute and cache transformed polygons for ref tree ONCE
    d2 ref_polys[MAX_PIECES][MAX_VERTS_PER_PIECE];
    double ref_aabb_min_x[MAX_PIECES];
    double ref_aabb_max_x[MAX_PIECES];
    double ref_aabb_min_y[MAX_PIECES];
    double ref_aabb_max_y[MAX_PIECES];
    
    for (int pi = 0; pi < MAX_PIECES; ++pi) {
        int n1 = const_piece_nverts[pi];
        double min1_x = 1e30, max1_x = -1e30;
        double min1_y = 1e30, max1_y = -1e30;

        // Load and transform piece with ref tree's pose, compute AABB
        for (int v = 0; v < n1; ++v) {
            int idx = pi * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            double x = const_piece_xy[base + 0];
            double y = const_piece_xy[base + 1];
            
            // Apply ref tree transform
            double x1 = c1 * x - s1 * y + ref.x;
            double y1 = s1 * x + c1 * y + ref.y;
            ref_polys[pi][v] = make_d2(x1, y1);
            
            // Update AABB
            min1_x = fmin(min1_x, x1);
            max1_x = fmax(max1_x, x1);
            min1_y = fmin(min1_y, y1);
            max1_y = fmax(max1_y, y1);
        }
        
        ref_aabb_min_x[pi] = min1_x;
        ref_aabb_max_x[pi] = max1_x;
        ref_aabb_min_y[pi] = min1_y;
        ref_aabb_max_y[pi] = max1_y;
    }
    
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
        
        // Precompute other tree transform
        double c2 = 0.0;
        double s2 = 0.0;
        sincos(other.z, &s2, &c2);
        double total = 0.0;

        // Loop over convex pieces of ref tree (now use cached transforms)
        for (int pi = 0; pi < MAX_PIECES; ++pi) {
            int n1 = const_piece_nverts[pi];

            // Loop over convex pieces of other tree
            for (int pj = 0; pj < MAX_PIECES; ++pj) {
                int n2 = const_piece_nverts[pj];
                d2 poly2[MAX_VERTS_PER_PIECE];
                double min2_x = 1e30, max2_x = -1e30;
                double min2_y = 1e30, max2_y = -1e30;

                for (int v = 0; v < n2; ++v) {
                    int idx = pj * MAX_VERTS_PER_PIECE + v;
                    int base = 2 * idx;
                    double x = const_piece_xy[base + 0];
                    double y = const_piece_xy[base + 1];

                    // Apply other tree transform
                    double x2 = c2 * x - s2 * y + other.x;
                    double y2 = s2 * x + c2 * y + other.y;
                    poly2[v] = make_d2(x2, y2);
                    
                    // Update AABB
                    min2_x = fmin(min2_x, x2);
                    max2_x = fmax(max2_x, x2);
                    min2_y = fmin(min2_y, y2);
                    max2_y = fmax(max2_y, y2);
                }

                // AABB overlap test - early exit if no overlap
                if (ref_aabb_max_x[pi] < min2_x || max2_x < ref_aabb_min_x[pi] ||
                    ref_aabb_max_y[pi] < min2_y || max2_y < ref_aabb_min_y[pi]) {
                    continue;  // No AABB overlap, skip expensive intersection
                }

                total += convex_intersection_area(ref_polys[pi], n1, poly2, n2);
            }
        }
        
        sum += total;
    }

    return sum;
}

// Sum overlaps between trees in xyt1 and trees in xyt2.
// Each tree in xyt1 is compared against all trees in xyt2.
// Identical poses are automatically skipped.
// Result is divided by 2 since each pair is counted twice.
// Computes gradients for all trees in xyt1 if out_grads is non-NULL.
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
    const double* row_x2 = xyt2_3xN + 0 * n2;
    const double* row_y2 = xyt2_3xN + 1 * n2;
    const double* row_t2 = xyt2_3xN + 2 * n2;

    double local_sum = 0.0;
    double local_grad[3] = {0.0, 0.0, 0.0};

    if (tid < n1) {
        double3 ref;
        ref.x = row_x[tid];
        ref.y = row_y[tid];
        ref.z = row_t[tid];

        // Compute overlap sum; gradients computed below if requested.
        local_sum = overlap_ref_with_list(ref, xyt2_3xN, n2);

        if (out_grads != NULL) {
            const double eps = 1e-6;

            // Zero local gradient
            local_grad[0] = 0.0;
            local_grad[1] = 0.0;
            local_grad[2] = 0.0;

            for (int i = 0; i < n2; ++i) {
                double3 other;
                other.x = row_x2[i];
                other.y = row_y2[i];
                other.z = row_t2[i];

                // Skip identical poses
                if (other.x == ref.x && other.y == ref.y && other.z == ref.z) {
                    continue;
                }

                // Build single-element flattened 3x1 array [x, y, theta]
                double other_xyt[3];
                other_xyt[0] = other.x;
                other_xyt[1] = other.y;
                other_xyt[2] = other.z;

                double overlap = overlap_ref_with_list(ref, other_xyt, 1);
                if (overlap <= 0.0) continue;

                // x
                double3 ref_px = ref; ref_px.x += eps;
                double3 ref_mx = ref; ref_mx.x -= eps;
                double f_px = overlap_ref_with_list(ref_px, other_xyt, 1);
                double f_mx = overlap_ref_with_list(ref_mx, other_xyt, 1);
                local_grad[0] += (f_px - f_mx) / (2.0 * eps);

                // y
                double3 ref_py = ref; ref_py.y += eps;
                double3 ref_my = ref; ref_my.y -= eps;
                double f_py = overlap_ref_with_list(ref_py, other_xyt, 1);
                double f_my = overlap_ref_with_list(ref_my, other_xyt, 1);
                local_grad[1] += (f_py - f_my) / (2.0 * eps);

                // theta
                double3 ref_pz = ref; ref_pz.z += eps;
                double3 ref_mz = ref; ref_mz.z -= eps;
                double f_pz = overlap_ref_with_list(ref_pz, other_xyt, 1);
                double f_mz = overlap_ref_with_list(ref_mz, other_xyt, 1);
                local_grad[2] += (f_pz - f_mz) / (2.0 * eps);
            }

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
    global _num_pieces, _raw_module,  _overlap_list_total_kernel

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

    # Overwrite the file each time to ensure it matches the compiled source.
    # Let any IO errors propagate so callers see a clear failure.
    with open(persist_path, 'w', encoding='utf-8') as _f:
        _f.write(_CUDA_SRC)

    # Compile CUDA module from the in-memory source string. This keeps
    # behavior compatible across CuPy versions that may not accept
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH','')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH; please install the CUDA toolkit or add nvcc to PATH")

    ptx_path = os.path.join(persist_dir, 'pack_cuda_saved.ptx')
    cmd = [nvcc_path, "-lineinfo", "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed (exit {proc.returncode}):\n{proc.stderr.decode(errors='ignore')}")

    # Load compiled PTX into a CuPy RawModule
    _raw_module = cp.RawModule(path=ptx_path)
    #_raw_module = cp.RawModule(path='/mnt/d/packing/temp/pack_cuda_saved.ptx', backend='nvcc', options=('-lineinfo',))
    _overlap_list_total_kernel = _raw_module.get_function("overlap_list_total_kernel")

    # Copy polygon data to constant memory (cached on-chip, broadcast to all threads)
    const_piece_xy_ptr = _raw_module.get_global('const_piece_xy')
    const_piece_nverts_ptr = _raw_module.get_global('const_piece_nverts')
    # Use memcpyHtoD to copy to device constant memory
    cp.cuda.runtime.memcpy(const_piece_xy_ptr.ptr, piece_xy_flat.ctypes.data, piece_xy_flat.nbytes, cp.cuda.runtime.memcpyHostToDevice)
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

    xyt1_arr = cp.asarray(xyt1, dtype=cp.float64)
    if xyt1_arr.ndim != 2 or xyt1_arr.shape[1] != 3:
        raise ValueError("xyt1 must be shape (N,3)")

    xyt2_arr = cp.asarray(xyt2, dtype=cp.float64)
    if xyt2_arr.ndim != 2 or xyt2_arr.shape[1] != 3:
        raise ValueError("xyt2 must be shape (N,3)")

    n1 = int(xyt1_arr.shape[0])
    n2 = int(xyt2_arr.shape[0])

    # Flatten to 3xN row-major so rows are x,y,theta
    xyt1_3xN = cp.ascontiguousarray(xyt1_arr.T).ravel()
    xyt2_3xN = cp.ascontiguousarray(xyt2_arr.T).ravel()

    # Allocate a single-element output to hold the accumulated total
    out_total = cp.zeros(1, dtype=cp.float64)
    
    # Allocate gradient output if requested (gradients are w.r.t. xyt1)
    out_grads = cp.zeros(n1 * 3, dtype=cp.float64) if compute_grad else None

    threads_per_block = n1
    blocks = 1

    # Dummy empty array to use when gradient output is not requested.
    # Passing an empty CuPy array is safer than constructing an unowned
    # memory pointer (some CuPy versions reject a zero-sized UnownedMemory).
    null_ptr = cp.asarray([], dtype=cp.float64)

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
        return out_total

