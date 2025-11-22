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
    double c1 = cos(a.z);
    double s1 = sin(a.z);
    double c2 = cos(b.z);
    double s2 = sin(b.z);
    double total = 0.0;

    // Loop over convex pieces of tree 1
    for (int i = 0; i < num_pieces; ++i) {
        int n1 = piece_nverts[i];
        d2 poly1[MAX_VERTS_PER_PIECE];

        // Load and transform piece with tree 1's pose
        for (int v = 0; v < n1; ++v) {
            int idx = i * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            double x = piece_xy[base + 0];
            double y = piece_xy[base + 1];
            
            // Apply tree 1 transform
            double x1 = c1 * x - s1 * y + a.x;
            double y1 = s1 * x + c1 * y + a.y;
            poly1[v] = make_d2(x1, y1);
        }

        // Loop over convex pieces of tree 2
        for (int j = 0; j < num_pieces; ++j) {
            int n2 = piece_nverts[j];
            d2 poly2[MAX_VERTS_PER_PIECE];

            for (int v = 0; v < n2; ++v) {
                int idx = j * MAX_VERTS_PER_PIECE + v;
                int base = 2 * idx;
                double x = piece_xy[base + 0];
                double y = piece_xy[base + 1];

                // Apply tree 2 transform
                double x2 = c2 * x - s2 * y + b.x;
                double y2 = s2 * x + c2 * y + b.y;
                poly2[v] = make_d2(x2, y2);
            }

            total += convex_intersection_area(poly1, n1, poly2, n2);
        }
    }

    return total;
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
    global _num_pieces, _raw_module, _overlap_two_trees_kernel

    if _initialized:
        return

    piece_xy_flat, piece_nverts = _build_convex_piece_arrays(CONVEX_PIECES)

    _num_pieces = piece_nverts.shape[0]

    _piece_xy_d = cp.asarray(piece_xy_flat, dtype=cp.float64)
    _piece_nverts_d = cp.asarray(piece_nverts, dtype=cp.int32)

    _raw_module = cp.RawModule(code=_CUDA_SRC, options=("--std=c++11",))
    _overlap_two_trees_kernel = _raw_module.get_function("overlap_two_trees_kernel")

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

