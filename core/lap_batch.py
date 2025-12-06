"""
Batched Linear Assignment Problem (LAP) solver using GPU Hungarian Algorithm.

All computation stays on GPU - no CPU transfers.
Single kernel launch solves all problems in parallel.

Usage:
    from lap_batch import solve_lap_batch
    
    # cost_matrices: cupy array of shape (batch_size, N, N)
    assignments, costs = solve_lap_batch(cost_matrices)
"""

import cupy as cp

# Hungarian algorithm kernel - one block per problem, one thread per block
# (Hungarian is sequential; parallelism comes from solving all 800 problems simultaneously)
_hungarian_kernel = cp.RawKernel(r'''
extern "C" __global__ void hungarian_kernel(
    const float* __restrict__ costs_in,  // (batch_size, N, N)
    float* __restrict__ costs,            // (batch_size, N, N) working copy
    int* __restrict__ row_match,          // (batch_size, N) row->col matching
    int* __restrict__ col_match,          // (batch_size, N) col->row matching
    float* __restrict__ u,                // (batch_size, N) row potentials
    float* __restrict__ v,                // (batch_size, N) col potentials
    const int batch_size,
    const int N
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    
    // Single thread per block - Hungarian is inherently sequential
    if (threadIdx.x != 0) return;
    
    const float* my_costs_in = costs_in + bid * N * N;
    float* my_costs = costs + bid * N * N;
    int* my_row_match = row_match + bid * N;
    int* my_col_match = col_match + bid * N;
    float* my_u = u + bid * N;
    float* my_v = v + bid * N;
    
    // Copy costs and initialize
    for (int i = 0; i < N; i++) {
        my_row_match[i] = -1;
        my_col_match[i] = -1;
        my_u[i] = 0.0f;
        my_v[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            my_costs[i * N + j] = my_costs_in[i * N + j];
        }
    }
    
    // Kuhn-Munkres algorithm - process each row
    for (int i = 0; i < N; i++) {
        // Arrays for path finding (on stack, assuming N <= 128)
        int p[128];      // parent pointers for augmenting path
        float minv[128]; // minimum slack to each column
        bool used[128];  // columns used in current search
        
        for (int j = 0; j < N; j++) {
            minv[j] = 1e30f;
            used[j] = false;
            p[j] = -1;
        }
        
        int cur_row = i;
        int cur_col = -1;
        
        // Dijkstra-style search for augmenting path
        while (true) {
            int best_col = -1;
            float best_val = 1e30f;
            
            for (int j = 0; j < N; j++) {
                if (used[j]) continue;
                
                float reduced = my_costs[cur_row * N + j] - my_u[cur_row] - my_v[j];
                if (reduced < minv[j]) {
                    minv[j] = reduced;
                    p[j] = cur_row;
                }
                if (minv[j] < best_val) {
                    best_val = minv[j];
                    best_col = j;
                }
            }
            
            // Update potentials
            for (int j = 0; j < N; j++) {
                if (used[j]) {
                    my_u[my_col_match[j]] += best_val;
                    my_v[j] -= best_val;
                } else {
                    minv[j] -= best_val;
                }
            }
            my_u[i] += best_val;
            
            used[best_col] = true;
            cur_col = best_col;
            cur_row = my_col_match[cur_col];
            
            if (cur_row < 0) break;  // Found augmenting path
        }
        
        // Trace back and flip the path
        while (cur_col >= 0) {
            int prev_row = p[cur_col];
            int prev_col = my_row_match[prev_row];
            my_row_match[prev_row] = cur_col;
            my_col_match[cur_col] = prev_row;
            cur_col = prev_col;
        }
    }
}
''', 'hungarian_kernel')


_compute_costs_kernel = cp.RawKernel(r'''
extern "C" __global__ void compute_costs(
    const float* __restrict__ costs,
    const int* __restrict__ assignments,
    float* __restrict__ total_costs,
    const int batch_size,
    const int N
) {
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;
    
    const float* my_costs = costs + bid * N * N;
    const int* my_assign = assignments + bid * N;
    
    float total = 0.0f;
    for (int row = 0; row < N; row++) {
        int col = my_assign[row];
        if (col >= 0 && col < N) {
            total += my_costs[row * N + col];
        }
    }
    total_costs[bid] = total;
}
''', 'compute_costs')


def solve_lap_batch(cost_matrices_gpu: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Solve batched LAP using GPU Hungarian Algorithm (exact).
    
    Single kernel launch solves all problems in parallel.
    All computation stays on GPU - no CPU transfers.
    
    Parameters
    ----------
    cost_matrices_gpu : cp.ndarray
        Shape (batch_size, N, N). Cost matrices on GPU.
        N must be <= 128.
        
    Returns
    -------
    assignments : cp.ndarray
        Shape (batch_size, N). Row-to-column assignments (optimal).
    costs : cp.ndarray
        Shape (batch_size,). Total assignment cost for each problem.
    """
    batch_size, N, _ = cost_matrices_gpu.shape
    assert N <= 128, f"N={N} exceeds maximum of 128"
    
    if cost_matrices_gpu.dtype != cp.float32:
        cost_matrices_gpu = cost_matrices_gpu.astype(cp.float32)
    cost_matrices_gpu = cp.ascontiguousarray(cost_matrices_gpu)
    
    costs_work = cp.empty_like(cost_matrices_gpu)
    row_match = cp.full((batch_size, N), -1, dtype=cp.int32)
    col_match = cp.full((batch_size, N), -1, dtype=cp.int32)
    u = cp.zeros((batch_size, N), dtype=cp.float32)
    v = cp.zeros((batch_size, N), dtype=cp.float32)
    total_costs = cp.empty(batch_size, dtype=cp.float32)
    
    # One block per problem, ONE thread per block
    _hungarian_kernel(
        (batch_size,), (1,),
        (cost_matrices_gpu, costs_work, row_match, col_match, u, v, batch_size, N)
    )
    
    blocks = (batch_size + 255) // 256
    _compute_costs_kernel(
        (blocks,), (256,),
        (cost_matrices_gpu, row_match, total_costs, batch_size, N)
    )
    
    return row_match, total_costs
