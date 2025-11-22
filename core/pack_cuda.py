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
    double dx, double dy, double dtheta,
    const double* __restrict__ piece_xy,
    const int* __restrict__ piece_nverts,
    int num_pieces)
{
    // Compute overlap area between two identical trees given the relative
    // transform (dx, dy, dtheta) between them. This is the core per-sample
    // routine used by the kernel. All work happens with scalar inputs.
    double c = cos(dtheta);
    double s = sin(dtheta);
    double total = 0.0;

    // Loop over convex pieces of base polygon (pose 1 = identity)
    for (int i = 0; i < num_pieces; ++i) {
        int n1 = piece_nverts[i];
        d2 poly1[MAX_VERTS_PER_PIECE];

        // Load base piece in world coords (identity transform)
        for (int v = 0; v < n1; ++v) {
            int idx = i * MAX_VERTS_PER_PIECE + v;
            int base = 2 * idx;
            double x = piece_xy[base + 0];
            double y = piece_xy[base + 1];
            poly1[v] = make_d2(x, y);
        }

        // Loop over convex pieces of moved polygon (pose 2 = relative dx,dy,theta)
        for (int j = 0; j < num_pieces; ++j) {
            int n2 = piece_nverts[j];
            d2 poly2[MAX_VERTS_PER_PIECE];

            for (int v = 0; v < n2; ++v) {
                int idx = j * MAX_VERTS_PER_PIECE + v;
                int base = 2 * idx;
                double x = piece_xy[base + 0];
                double y = piece_xy[base + 1];

                // Rotate + translate by relative transform
                double xr = c * x - s * y + dx;
                double yr = s * x + c * y + dy;
                poly2[v] = make_d2(xr, yr);
            }

            total += convex_intersection_area(poly1, n1, poly2, n2);
        }
    }

    return total;
}

__global__ void overlap_kernel(
    const double* __restrict__ dx,
    const double* __restrict__ dy,
    const double* __restrict__ dtheta,
    const double* __restrict__ piece_xy,      // length num_pieces * MAX_VERTS_PER_PIECE * 2
    const int*    __restrict__ piece_nverts,  // length num_pieces
    int num_pieces,
    int n_samples,
    double* __restrict__ out_area)
{
    // Each thread handles one sample (one relative transform between trees).
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;

    double tx = dx[tid];
    double ty = dy[tid];
    double th = dtheta[tid];

    out_area[tid] = overlap_two_trees(
        tx, ty, th,
        piece_xy, piece_nverts, num_pieces);
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
_overlap_kernel: cp.RawKernel | None = None

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
    global _num_pieces, _raw_module, _overlap_kernel

    if _initialized:
        return

    piece_xy_flat, piece_nverts = _build_convex_piece_arrays(CONVEX_PIECES)

    _num_pieces = piece_nverts.shape[0]

    _piece_xy_d = cp.asarray(piece_xy_flat, dtype=cp.float64)
    _piece_nverts_d = cp.asarray(piece_nverts, dtype=cp.int32)

    _raw_module = cp.RawModule(code=_CUDA_SRC, options=("--std=c++11",))
    _overlap_kernel = _raw_module.get_function("overlap_kernel")

    _initialized = True


# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------

def compute_overlap_area_gpu(
    x1: float,
    y1: float,
    theta1: float,
    x2: float,
    y2: float,
    theta2: float,
) -> cp.ndarray:
    """Compute overlap area between two poses of the SAME polygon on the GPU.

    The base polygon is defined implicitly by CONVEX_PIECES (4 convex pieces).

    Pose 1: (x1, y1, theta1)
    Pose 2: (x2, y2, theta2)

    Because the polygons are identical, the overlap depends only on the
    relative transform, so internally we use:
        dx     = x2 - x1
        dy     = y2 - y1
        dtheta = theta2 - theta1

    Parameters
    ----------
    x1, y1, theta1, x2, y2, theta2 : float
        Scalar poses of the two polygon copies (Python floats or NumPy scalars).

    Returns
    -------
    area : cp.ndarray
        A 0-d CuPy array (cp.float64) containing the overlap area.
    """
    _ensure_initialized()

    # Compute relative transform as Python floats, then wrap into small
    # 1-element CuPy arrays to satisfy the kernel's pointer-based interface.
    dx_val = float(x2) - float(x1)
    dy_val = float(y2) - float(y1)
    dtheta_val = float(theta2) - float(theta1)

    dx_arr = cp.asarray([dx_val], dtype=cp.float64)
    dy_arr = cp.asarray([dy_val], dtype=cp.float64)
    dtheta_arr = cp.asarray([dtheta_val], dtype=cp.float64)

    n_samples = 1
    areas_arr = cp.empty(1, dtype=cp.float64)

    threads_per_block = 1
    blocks = 1

    _overlap_kernel(
        (blocks,),
        (threads_per_block,),
        (
            dx_arr,
            dy_arr,
            dtheta_arr,
            _piece_xy_d,
            _piece_nverts_d,
            np.int32(_num_pieces),
            np.int32(n_samples),
            areas_arr,
        ),
    )

    # Return as a 0-d CuPy array for easy use in CuPy/NumPy code.
    return areas_arr[0]

