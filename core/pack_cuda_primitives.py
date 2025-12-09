"""
pack_cuda_primitives.py

Primitive CUDA device functions for polygon operations.
These are low-level geometric operations used by higher-level functions.
"""

PRIMITIVE_SRC = r"""
// Use CUDA's built-in double2 type (more idiomatic and potentially better optimized)
typedef double2 d2;

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

    return make_double2(p1.x + t * rx, p1.y + t * ry);
}

// Compute signed polygon area via the shoelace formula (absolute value returned).
// Also computes gradients w.r.t. each vertex (x,y) coordinate.
// Shoelace: A = 0.5 * sum_i (x_i*y_{i+1} - x_{i+1}*y_i)
// ∂A/∂x_k = 0.5 * (y_{k+1} - y_{k-1})
// ∂A/∂y_k = 0.5 * (x_{k-1} - x_{k+1})
__device__ __forceinline__ double polygon_area(const d2* v, int n, d2* d_v) {
    if (n < 3) {
        for (int i = 0; i < n; ++i) {
            d_v[i] = make_double2(0.0, 0.0);
        }
        return 0.0;
    }
    
    // Compute signed area
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        sum += v[i].x * v[j].y - v[j].x * v[i].y;
    }
    double signed_area = 0.5 * sum;
    double area = (signed_area >= 0.0) ? signed_area : -signed_area;
    
    // Skip backward if area is zero
    if (area == 0.0) {
        for (int i = 0; i < n; ++i) {
            d_v[i] = make_double2(0.0, 0.0);
        }
        return 0.0;
    }
    
    // Compute gradients: gradient of |A| = sign(A) * ∂A/∂v
    // Implicit d_area = 1.0
    double sign = (signed_area >= 0.0) ? 1.0 : -1.0;
    double factor = sign * 1.0;
    
    for (int k = 0; k < n; ++k) {
        int k_prev = (k - 1 + n) % n;
        int k_next = (k + 1) % n;
        
        d_v[k].x = factor * 0.5 * (v[k_next].y - v[k_prev].y);
        d_v[k].y = factor * 0.5 * (v[k_prev].x - v[k_next].x);
    }
    
    return area;
}

// ============================================================================
// BACKWARD PASS PRIMITIVES
// ============================================================================

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
    d2 dt_dp1 = make_double2((dnum_dp1_x * denom - num * ddenom_dp1_x) / denom2,
                        (dnum_dp1_y * denom - num * ddenom_dp1_y) / denom2);
    
    // ∂num/∂p2 = 0
    // ∂denom/∂p2.x = sy,  ∂denom/∂p2.y = -sx
    d2 dt_dp2 = make_double2((-num * sy) / denom2, (num * sx) / denom2);
    
    // ∂num/∂q1.x = sy + (q1.y - p1.y) (from ∂sy/∂q1.x = 0 and ∂sx/∂q1.x = -1)
    // ∂num/∂q1.y = -sx - (q1.x - p1.x) (from ∂sy/∂q1.y = -1 and ∂sx/∂q1.y = 0)
    // ∂denom/∂q1.x = ry (from ∂sy/∂q1.x = 0 and ∂sx/∂q1.x = -1)
    // ∂denom/∂q1.y = -rx (from ∂sy/∂q1.y = -1 and ∂sx/∂q1.y = 0)
    double dnum_dq1_x = sy + (q1.y - p1.y);
    double dnum_dq1_y = -sx - (q1.x - p1.x);
    double ddenom_dq1_x = ry;
    double ddenom_dq1_y = -rx;
    d2 dt_dq1 = make_double2((dnum_dq1_x * denom - num * ddenom_dq1_x) / denom2,
                        (dnum_dq1_y * denom - num * ddenom_dq1_y) / denom2);
    
    // ∂num/∂q2.x = -(q1.y - p1.y) (from ∂sy/∂q2.x = 0 and ∂sx/∂q2.x = +1)
    // ∂num/∂q2.y = (q1.x - p1.x) (from ∂sy/∂q2.y = +1 and ∂sx/∂q2.y = 0)
    // ∂denom/∂q2.x = -ry (from ∂sy/∂q2.x = 0 and ∂sx/∂q2.x = +1)
    // ∂denom/∂q2.y = rx (from ∂sy/∂q2.y = +1 and ∂sx/∂q2.y = 0)
    double dnum_dq2_x = -(q1.y - p1.y);
    double dnum_dq2_y = (q1.x - p1.x);
    double ddenom_dq2_x = -ry;
    double ddenom_dq2_y = rx;
    d2 dt_dq2 = make_double2((dnum_dq2_x * denom - num * ddenom_dq2_x) / denom2,
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

// Clip against edge with metadata for backward pass
__device__ __forceinline__ int clip_against_edge(
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
        d_in_pts[i] = make_double2(0.0, 0.0);
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

// Compute the area of intersection between two convex polygons using
// Sutherland-Hodgman clipping of "subj" against all edges of "clip".
// If d_subj and d_clip are non-NULL, also computes gradients w.r.t. both input polygons.
// If they are NULL, skips the backward pass for faster forward-only computation.
__device__ double convex_intersection_area(
    const d2* subj, int n_subj,
    const d2* clip, int n_clip,
    d2* d_subj,     // output: gradient w.r.t. subject vertices (can be NULL)
    d2* d_clip)     // output: gradient w.r.t. clip vertices (can be NULL)
{
    const int compute_grads = (d_subj != NULL && d_clip != NULL);
    
    // Early exit for empty input
    if (n_subj == 0 || n_clip == 0) {
        if (compute_grads) {
            for (int i = 0; i < n_subj; ++i) {
                d_subj[i] = make_double2(0.0, 0.0);
            }
            for (int i = 0; i < n_clip; ++i) {
                d_clip[i] = make_double2(0.0, 0.0);
            }
        }
        return 0.0;
    }

    // Forward pass: perform clipping
    // Only save metadata and intermediate polygons if computing gradients
    d2 forward_polys[MAX_VERTS_PER_PIECE + 1][MAX_INTERSECTION_VERTS];
    int forward_counts[MAX_VERTS_PER_PIECE + 1];
    ClipMetadata metadata[MAX_VERTS_PER_PIECE];
    
    // For forward-only mode, we just need current and next buffers
    d2 current_poly[MAX_INTERSECTION_VERTS];
    d2 next_poly[MAX_INTERSECTION_VERTS];
    int current_count;
    
    if (compute_grads) {
        // Initialize with subject polygon (save for backward pass)
        forward_counts[0] = n_subj;
        for (int i = 0; i < n_subj; ++i) {
            forward_polys[0][i] = subj[i];
        }
    } else {
        // Initialize current buffer
        current_count = n_subj;
        for (int i = 0; i < n_subj; ++i) {
            current_poly[i] = subj[i];
        }
    }

    // Apply each clipping edge
    int clip_count = 0;
    if (compute_grads) {
        for (int e = 0; e < n_clip && forward_counts[e] > 0; ++e) {
            d2 A = clip[e];
            d2 B = clip[(e + 1) % n_clip];

            int n_out = clip_against_edge(
                forward_polys[e], forward_counts[e], 
                A, B, 
                forward_polys[e + 1], &metadata[e]);
            forward_counts[e + 1] = n_out;
            clip_count++;
        }
    } else {
        // Forward-only: no metadata needed
        for (int e = 0; e < n_clip && current_count > 0; ++e) {
            d2 A = clip[e];
            d2 B = clip[(e + 1) % n_clip];

            int out_count = 0;
            d2 S = current_poly[current_count - 1];
            double S_side = cross3(A, B, S);

            for (int i = 0; i < current_count; ++i) {
                d2 E = current_poly[i];
                double E_side = cross3(A, B, E);

                bool S_inside = (S_side >= 0.0);
                bool E_inside = (E_side >= 0.0);

                if (S_inside && E_inside) {
                    next_poly[out_count++] = E;
                } else if (S_inside && !E_inside) {
                    next_poly[out_count++] = line_intersection(S, E, A, B);
                } else if (!S_inside && E_inside) {
                    next_poly[out_count++] = line_intersection(S, E, A, B);
                    next_poly[out_count++] = E;
                }

                S = E;
                S_side = E_side;
            }
            
            // Swap buffers
            current_count = out_count;
            for (int i = 0; i < out_count; ++i) {
                current_poly[i] = next_poly[i];
            }
            clip_count++;
        }
    }

    // Get final polygon and compute area
    int final_n;
    d2* final_poly;
    if (compute_grads) {
        final_n = forward_counts[clip_count];
        final_poly = forward_polys[clip_count];
    } else {
        final_n = current_count;
        final_poly = current_poly;
    }
    
    // Compute area (and gradients if needed)
    if (final_n < 3) {
        if (compute_grads) {
            for (int i = 0; i < n_subj; ++i) {
                d_subj[i] = make_double2(0.0, 0.0);
            }
            for (int i = 0; i < n_clip; ++i) {
                d_clip[i] = make_double2(0.0, 0.0);
            }
        }
        return 0.0;
    }
    
    double area;
    if (compute_grads) {
        d2 d_polyFinal[MAX_INTERSECTION_VERTS];
        area = polygon_area(final_poly, final_n, d_polyFinal);
        
        // Skip backward pass if area is zero
        if (area == 0.0) {
            for (int i = 0; i < n_subj; ++i) {
                d_subj[i] = make_double2(0.0, 0.0);
            }
            for (int i = 0; i < n_clip; ++i) {
                d_clip[i] = make_double2(0.0, 0.0);
            }
            return 0.0;
        }
        
        // Initialize clip gradients to zero before accumulation
        for (int i = 0; i < n_clip; ++i) {
            d_clip[i] = make_double2(0.0, 0.0);
        }
        
        // Backward pass: backprop through clipping stages
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
    } else {
        // Forward-only: just compute area using shoelace formula
        double sum = 0.0;
        for (int i = 0; i < final_n; ++i) {
            int j = (i + 1) % final_n;
            sum += final_poly[i].x * final_poly[j].y - final_poly[j].x * final_poly[i].y;
        }
        area = fabs(0.5 * sum);
    }
    
    return area;
}

__device__ inline double dot2(const double2& a, const double2& b) {
    return a.x * b.x + a.y * b.y;
}

__device__ void sat_separation_with_grad_pose(
    const double2* __restrict__ verts1_local,
    int n1,
    const double2* __restrict__ verts2_world,
    int n2,
    double x1,
    double y1,
    double cos_th1,
    double sin_th1,
    double* sep,
    double* dsep_dx,      // can be NULL to skip gradient computation
    double* dsep_dy,      // can be NULL to skip gradient computation
    double* dsep_dtheta)  // can be NULL to skip gradient computation
{
    const int compute_grads = (dsep_dx != NULL && dsep_dy != NULL && dsep_dtheta != NULL);
    
    // Default outputs: no penetration
    *sep = 0.0;
    if (compute_grads) {
        *dsep_dx     = 0.0;
        *dsep_dy     = 0.0;
        *dsep_dtheta = 0.0;
    }

    if (n1 < 3 || n2 < 3) {
        return;
    }
    if (n1 > MAX_VERTS_PER_PIECE || n2 > MAX_VERTS_PER_PIECE) {
        return;
    }

    // Precompute world-space coords for poly1 from local coords + pose.
    double2 world1[MAX_VERTS_PER_PIECE];
    for (int i = 0; i < n1; ++i) {
        double lx = verts1_local[i].x;
        double ly = verts1_local[i].y;
        double wx =  cos_th1 * lx - sin_th1 * ly + x1;
        double wy =  sin_th1 * lx + cos_th1 * ly + y1;
        world1[i] = make_double2(wx, wy);
    }

    const double INF = 1.0e30;

    double min_penetration = INF;
    bool   found_overlap_axis = false;

    // Data for the active axis
    double2 best_normal     = make_double2(0.0, 0.0);
    int     best_case       = -1;  // 1 = pen1, 2 = pen2
    int     best_v_idx      = -1;
    double2 best_w_world    = make_double2(0.0, 0.0);
    bool    axis_from_poly1 = false;

    double proj1[MAX_VERTS_PER_PIECE];
    double proj2[MAX_VERTS_PER_PIECE];

    // ---- Test axes from poly1 edges ----
    for (int i = 0; i < n1; ++i) {
        double2 p1 = world1[i];
        double2 p2 = world1[(i + 1) % n1];
        double2 edge = make_double2(p2.x - p1.x, p2.y - p1.y);

        double2 normal = make_double2(-edge.y, edge.x);
        double len = sqrt(normal.x * normal.x + normal.y * normal.y);
        if (len < 1.0e-12) {
            continue;
        }
        normal.x /= len;
        normal.y /= len;

        // Project both polygons onto this axis
        for (int k = 0; k < n1; ++k) {
            proj1[k] = dot2(world1[k], normal);
        }
        for (int k = 0; k < n2; ++k) {
            proj2[k] = dot2(verts2_world[k], normal);
        }

        // Compute intervals
        double min1 = proj1[0], max1 = proj1[0];
        for (int k = 1; k < n1; ++k) {
            if (proj1[k] < min1) min1 = proj1[k];
            if (proj1[k] > max1) max1 = proj1[k];
        }
        double min2 = proj2[0], max2 = proj2[0];
        for (int k = 1; k < n2; ++k) {
            if (proj2[k] < min2) min2 = proj2[k];
            if (proj2[k] > max2) max2 = proj2[k];
        }

        // *** SAT CONDITION: if there is ANY separating axis, polygons do not overlap ***
        if (max1 < min2 || max2 < min1) {
            // Separated on this axis -> SAT says NO overlap at all.
            return;  // sep, grads already zero
        }

        // If you reach here, intervals overlap on this axis => candidate penetration
        found_overlap_axis = true;

        double pen1 = max1 - min2;  // move poly1 along -normal
        double pen2 = max2 - min1;  // move poly1 along +normal
        double penetration = (pen1 < pen2 ? pen1 : pen2);

        if (penetration < min_penetration) {
            min_penetration = penetration;
            best_normal     = normal;
            axis_from_poly1 = true;

            if (pen1 <= pen2) {
                best_case = 1; // pen1
                int idx_v = 0;
                for (int k = 1; k < n1; ++k) {
                    if (proj1[k] > proj1[idx_v]) {
                        idx_v = k;
                    }
                }
                best_v_idx = idx_v;
                int idx_w = 0;
                for (int k = 1; k < n2; ++k) {
                    if (proj2[k] < proj2[idx_w]) {
                        idx_w = k;
                    }
                }
                best_w_world = verts2_world[idx_w];
            } else {
                best_case = 2; // pen2
                int idx_v = 0;
                for (int k = 1; k < n1; ++k) {
                    if (proj1[k] < proj1[idx_v]) {
                        idx_v = k;
                    }
                }
                best_v_idx = idx_v;
                int idx_w = 0;
                for (int k = 1; k < n2; ++k) {
                    if (proj2[k] > proj2[idx_w]) {
                        idx_w = k;
                    }
                }
                best_w_world = verts2_world[idx_w];
            }
        }
    }

    // ---- Test axes from poly2 edges ----
    for (int i = 0; i < n2; ++i) {
        double2 p1 = verts2_world[i];
        double2 p2 = verts2_world[(i + 1) % n2];
        double2 edge = make_double2(p2.x - p1.x, p2.y - p1.y);

        double2 normal = make_double2(-edge.y, edge.x);
        double len = sqrt(normal.x * normal.x + normal.y * normal.y);
        if (len < 1.0e-12) {
            continue;
        }
        normal.x /= len;
        normal.y /= len;

        for (int k = 0; k < n1; ++k) {
            proj1[k] = dot2(world1[k], normal);
        }
        for (int k = 0; k < n2; ++k) {
            proj2[k] = dot2(verts2_world[k], normal);
        }

        double min1 = proj1[0], max1 = proj1[0];
        for (int k = 1; k < n1; ++k) {
            if (proj1[k] < min1) min1 = proj1[k];
            if (proj1[k] > max1) max1 = proj1[k];
        }
        double min2 = proj2[0], max2 = proj2[0];
        for (int k = 1; k < n2; ++k) {
            if (proj2[k] < min2) min2 = proj2[k];
            if (proj2[k] > max2) max2 = proj2[k];
        }

        // *** SAT separating axis check again ***
        if (max1 < min2 || max2 < min1) {
            return;  // separated -> no penetration
        }

        found_overlap_axis = true;

        double pen1 = max1 - min2;
        double pen2 = max2 - min1;
        double penetration = (pen1 < pen2 ? pen1 : pen2);

        if (penetration < min_penetration) {
            min_penetration = penetration;
            best_normal     = normal;
            axis_from_poly1 = false;

            if (pen1 <= pen2) {
                best_case = 1; // pen1
                int idx_v = 0;
                for (int k = 1; k < n1; ++k) {
                    if (proj1[k] > proj1[idx_v]) {
                        idx_v = k;
                    }
                }
                best_v_idx = idx_v;
                int idx_w = 0;
                for (int k = 1; k < n2; ++k) {
                    if (proj2[k] < proj2[idx_w]) {
                        idx_w = k;
                    }
                }
                best_w_world = verts2_world[idx_w];
            } else {
                best_case = 2; // pen2
                int idx_v = 0;
                for (int k = 1; k < n1; ++k) {
                    if (proj1[k] < proj1[idx_v]) {
                        idx_v = k;
                    }
                }
                best_v_idx = idx_v;
                int idx_w = 0;
                for (int k = 1; k < n2; ++k) {
                    if (proj2[k] > proj2[idx_w]) {
                        idx_w = k;
                    }
                }
                best_w_world = verts2_world[idx_w];
            }
        }
    }

    // If we never saw overlapping intervals on any axis, treat as separated.
    // (Shouldn't happen for valid convex pairs, but keep this guard.)
    if (!found_overlap_axis || min_penetration <= 1.0e-12) {
        return;
    }

    *sep = min_penetration;
    
    // Skip gradient computation if not requested
    if (!compute_grads) {
        return;
    }

    // ---- Gradients ----
    double2 grad_v;
    if (best_case == 1) {
        grad_v = best_normal;
    } else {
        grad_v = make_double2(-best_normal.x, -best_normal.y);
    }

    double d_dx = grad_v.x;
    double d_dy = grad_v.y;

    double vx0 = verts1_local[best_v_idx].x;
    double vy0 = verts1_local[best_v_idx].y;

    double dvx_dt = -sin_th1 * vx0 - cos_th1 * vy0;
    double dvy_dt =  cos_th1 * vx0 - sin_th1 * vy0;

    double term_vertex = grad_v.x * dvx_dt + grad_v.y * dvy_dt;

    double term_axis = 0.0;
    if (axis_from_poly1) {
        double2 v_world = world1[best_v_idx];
        double2 w_world = best_w_world;

        double2 grad_n;
        if (best_case == 1) {
            grad_n = make_double2(v_world.x - w_world.x,
                                  v_world.y - w_world.y);
        } else {
            grad_n = make_double2(w_world.x - v_world.x,
                                  w_world.y - v_world.y);
        }

        double2 dn_dt = make_double2(-best_normal.y, best_normal.x);

        term_axis = grad_n.x * dn_dt.x + grad_n.y * dn_dt.y;
    }

    double d_dtheta = term_vertex + term_axis;

    *dsep_dx     = d_dx;
    *dsep_dy     = d_dy;
    *dsep_dtheta = d_dtheta;
}
"""