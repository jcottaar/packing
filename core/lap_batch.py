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
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LAPConfig:
    algorithm: Literal['hungarian', 'auction'] = 'hungarian'

    # Auction algorithm hyperparameters (Bertsekas auction w/ optional epsilon scaling)
    # Notes:
    # - Smaller epsilon_final is closer to exact but slower.
    # - For many use-cases (binary "similar vs not"), larger eps can be fine.
    auction_epsilon_init: float = 0.1
    auction_epsilon_final: float = 1e-3
    auction_epsilon_decay: float = 0.8
    auction_max_rounds: int = 3
    auction_max_iters: int = 0  # 0 => choose based on N on the host

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
        // Arrays for path finding (on stack, assuming N <= 200)
        int p[200];      // parent pointers for augmenting path
        float minv[200]; // minimum slack to each column
        bool used[200];  // columns used in current search
        
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


# Auction algorithm kernel - one block per problem, one thread per block
# Parallelism comes from solving many independent problems in the batch.
_auction_kernel = cp.RawKernel(r'''
extern "C" __global__ void auction_kernel(
    const float* __restrict__ costs_in,  // (batch_size, N, N)
    int* __restrict__ row_match,         // (batch_size, N) row->col
    int* __restrict__ col_match,         // (batch_size, N) col->row
    float* __restrict__ prices,          // (batch_size, N) prices for columns
    const int batch_size,
    const int N,
    const float epsilon_init,
    const float epsilon_final,
    const float epsilon_decay,
    const int max_rounds,
    const int max_iters
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    if (threadIdx.x != 0) return;

    const float* my_costs = costs_in + bid * N * N;
    int* my_row_match = row_match + bid * N;
    int* my_col_match = col_match + bid * N;
    float* my_prices = prices + bid * N;

    // Initialize
    for (int i = 0; i < N; i++) {
        my_row_match[i] = -1;
        my_col_match[i] = -1;
        my_prices[i] = 0.0f;
    }

    float eps = epsilon_init;
    for (int round = 0; round < max_rounds; round++) {
        // Epsilon scaling refinement: keep prices but re-run assignment so later
        // rounds can improve the solution.
        if (round > 0) {
            for (int i = 0; i < N; i++) {
                my_row_match[i] = -1;
                my_col_match[i] = -1;
            }
        }

        // Run auction iterations for current epsilon
        for (int it = 0; it < max_iters; it++) {
            bool all_assigned = true;
            for (int i = 0; i < N; i++) {
                if (my_row_match[i] < 0) { all_assigned = false; break; }
            }
            if (all_assigned) break;

            for (int i = 0; i < N; i++) {
                if (my_row_match[i] >= 0) continue;  // already assigned

                float best_val = -1e30f;
                float second_val = -1e30f;
                int best_j = -1;

                // Find best and second best column for bidder i
                const float* row_costs = my_costs + i * N;
                for (int j = 0; j < N; j++) {
                    // We solve min-cost assignment by maximizing (-cost - price)
                    float val = -row_costs[j] - my_prices[j];
                    if (val > best_val) {
                        second_val = best_val;
                        best_val = val;
                        best_j = j;
                    } else if (val > second_val) {
                        second_val = val;
                    }
                }

                // If N==1, second_val can stay at -inf-ish
                float increment = (best_val - second_val) + eps;
                if (best_j < 0) continue;

                my_prices[best_j] += increment;

                // Assign bidder i to best_j (possibly unassign previous owner)
                int prev_i = my_col_match[best_j];
                my_col_match[best_j] = i;
                my_row_match[i] = best_j;
                if (prev_i >= 0) {
                    my_row_match[prev_i] = -1;
                }
            }
        }

        // Epsilon scaling schedule
        if (eps <= epsilon_final) break;
        eps = eps * epsilon_decay;
        if (eps < epsilon_final) eps = epsilon_final;
    }
}
''', 'auction_kernel')


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


def solve_lap_batch(cost_matrices_gpu: cp.ndarray, config: LAPConfig | None = None) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Solve batched LAP using GPU Hungarian (exact) or GPU Auction algorithm.
    
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
    if config is None:
        config = LAPConfig()

    batch_size, N, _ = cost_matrices_gpu.shape
    assert N <= 200, f"N={N} exceeds maximum of 200"
    
    if cost_matrices_gpu.dtype != cp.float32:
        cost_matrices_gpu = cost_matrices_gpu.astype(cp.float32)
    cost_matrices_gpu = cp.ascontiguousarray(cost_matrices_gpu)
    
    row_match = cp.full((batch_size, N), -1, dtype=cp.int32)
    col_match = cp.full((batch_size, N), -1, dtype=cp.int32)
    total_costs = cp.empty(batch_size, dtype=cp.float32)

    if config.algorithm == 'hungarian':
        costs_work = cp.empty_like(cost_matrices_gpu)
        u = cp.zeros((batch_size, N), dtype=cp.float32)
        v = cp.zeros((batch_size, N), dtype=cp.float32)

        # One block per problem, ONE thread per block
        _hungarian_kernel(
            (batch_size,), (1,),
            (cost_matrices_gpu, costs_work, row_match, col_match, u, v, batch_size, N)
        )
    elif config.algorithm == 'auction':
        prices = cp.zeros((batch_size, N), dtype=cp.float32)

        eps_init = float(config.auction_epsilon_init)
        eps_final = float(config.auction_epsilon_final)
        eps_decay = float(config.auction_epsilon_decay)
        max_rounds = int(config.auction_max_rounds)
        max_iters = int(config.auction_max_iters)
        if max_iters <= 0:
            # Practical upper bound for convergence; tuned for float32 and typical dense problems.
            max_iters = max(10, 5 * N * N)

        _auction_kernel(
            (batch_size,), (1,),
            (cost_matrices_gpu, row_match, col_match, prices, batch_size, N,
             cp.float32(eps_init), cp.float32(eps_final), cp.float32(eps_decay),
             max_rounds, max_iters)
        )
    else:
        raise ValueError(f"Unknown LAPConfig.algorithm={config.algorithm!r}")
    
    blocks = (batch_size + 255) // 256
    _compute_costs_kernel(
        (blocks,), (256,),
        (cost_matrices_gpu, row_match, total_costs, batch_size, N)
    )
    
    return row_match, total_costs
