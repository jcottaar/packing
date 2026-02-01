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
import os
import subprocess
import shutil


@dataclass
class LAPConfig:
    algorithm: Literal['hungarian', 'auction', 'min_cost_row', 'min_cost_col'] = 'hungarian'

    # Auction algorithm hyperparameters (Bertsekas auction w/ optional epsilon scaling)
    # Notes:
    # - Smaller epsilon_final is closer to exact but slower.
    # - For many use-cases (binary "similar vs not"), larger eps can be fine.
    auction_epsilon_init: float = 0.1
    auction_epsilon_final: float = 1e-3
    auction_epsilon_decay: float = 0.8
    auction_max_rounds: int = 1
    auction_max_iters: int = 0  # 0 => choose based on N on the host

    # Diversity shortcut kernel option (for min_cost_row/min_cost_col)
    use_diversity_kernel: bool = True


# ---------------------------------------------------------------------------
# CUDA source code - all kernels in one compilation unit
# ---------------------------------------------------------------------------

_CUDA_SRC = r'''
extern "C" {

// Hungarian algorithm kernel - one block per problem, one thread per block
// (Hungarian is sequential; parallelism comes from solving all problems simultaneously)
__global__ void hungarian_kernel(
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


// Auction algorithm kernel - one block per problem, one thread per block
// Parallelism comes from solving many independent problems in the batch.
__global__ void auction_kernel(
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


__global__ void compute_costs(
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


// Diversity shortcut kernel - fuses pairwise distance computation with min-cost-row reduction
// One block per (pop_idx, ref_idx) pair, N_trees threads per block
// Applies transformation (rotation + mirror) to reference coordinates inside the kernel
// Only supports min_cost_row (for each pop tree, find min dist to any ref tree)
//
// Multithreaded design:
// Phase 1: Each thread cooperatively transforms one ref tree -> shared memory
// Phase 2: Each thread handles one pop tree, finds min over all refs
// Phase 3: Tree reduction to sum partial results
__global__ void diversity_shortcut_kernel(
    const float* __restrict__ pop_xyt,    // (N_pop, N_trees, 3) - contiguous
    const float* __restrict__ ref_xyt,    // (N_ref, N_trees, 3) - contiguous, untransformed
    float* __restrict__ costs_out,        // (N_pop * N_ref,)
    const int N_pop,
    const int N_ref,
    const int N_trees,
    const float cos_a,                    // cos(rotation_angle)
    const float sin_a,                    // sin(rotation_angle)
    const int do_mirror                   // 1 to mirror across x-axis, 0 otherwise
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_pairs = N_pop * N_ref;
    if (bid >= total_pairs) return;
    
    const int pop_idx = bid / N_ref;
    const int ref_idx = bid % N_ref;
    
    // Stride is 3 for xyt layout
    const float* my_pop = pop_xyt + pop_idx * N_trees * 3;
    const float* my_ref = ref_xyt + ref_idx * N_trees * 3;
    
    const float PI = 3.14159265358979323846f;
    const float TWO_PI = 2.0f * PI;
    
    // Compute rotation angle from sin/cos for theta offset
    float rot_angle = atan2f(sin_a, cos_a);
    
    // Shared memory for transformed ref coordinates and partial sums
    // Layout: ref_x[N_trees], ref_y[N_trees], ref_t[N_trees], partial_sums[N_trees]
    extern __shared__ float shared_mem[];
    float* ref_x = shared_mem;
    float* ref_y = shared_mem + N_trees;
    float* ref_t = shared_mem + 2 * N_trees;
    float* partial_sums = shared_mem + 3 * N_trees;
    
    // Phase 1: Cooperative transform - each thread transforms one ref tree
    if (tid < N_trees) {
        float rx = my_ref[tid * 3 + 0];
        float ry = my_ref[tid * 3 + 1];
        float rt = my_ref[tid * 3 + 2];
        
        // Mirror if needed
        if (do_mirror) {
            ry = -ry;
            rt = PI - rt;
        }
        
        // Rotate
        float rx_rot = rx * cos_a - ry * sin_a;
        float ry_rot = rx * sin_a + ry * cos_a;
        float rt_rot = rt + rot_angle;
        
        // Wrap rt_rot to [-pi, pi]
        rt_rot = fmodf(rt_rot + PI, TWO_PI);
        if (rt_rot < 0) rt_rot += TWO_PI;
        rt_rot -= PI;
        
        ref_x[tid] = rx_rot;
        ref_y[tid] = ry_rot;
        ref_t[tid] = rt_rot;
    }
    
    __syncthreads();
    
    // Phase 2: Each thread handles one pop tree, finds min distance to any ref tree
    float my_min_dist = 0.0f;
    if (tid < N_trees) {
        float px = my_pop[tid * 3 + 0];
        float py = my_pop[tid * 3 + 1];
        float pt = my_pop[tid * 3 + 2];
        
        float min_sq = 1e30f;
        
        for (int j = 0; j < N_trees; j++) {
            float dx = px - ref_x[j];
            float dy = py - ref_y[j];
            float dt = pt - ref_t[j];
            
            // Wrap dt to [-pi, pi]
            dt = fmodf(dt + PI, TWO_PI);
            if (dt < 0) dt += TWO_PI;
            dt -= PI;
            
            float sq = dx*dx + dy*dy + dt*dt;
            if (sq < min_sq) min_sq = sq;
        }
        
        my_min_dist = sqrtf(min_sq);
    }
    
    // Store in shared memory for reduction
    partial_sums[tid] = my_min_dist;
    
    __syncthreads();
    
    // Phase 3: Sequential sum by thread 0 (simpler and correct for any N_trees)
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < N_trees; i++) {
            total += partial_sums[i];
        }
        costs_out[bid] = total;
    }
}

}  // extern "C"
'''

# ---------------------------------------------------------------------------
# Compiled CUDA module and kernels (lazy initialization)
# ---------------------------------------------------------------------------

_module: cp.RawModule | None = None
_hungarian_kernel: cp.RawKernel | None = None
_auction_kernel: cp.RawKernel | None = None
_compute_costs_kernel: cp.RawKernel | None = None
_diversity_shortcut_kernel: cp.RawKernel | None = None
_initialized: bool = False


def _ensure_initialized() -> None:
    """Lazy initialization hook.
    
    On first call, this:
    - Persists the CUDA source to a .cu file for profiler correlation.
    - Compiles the CUDA source with nvcc using optimized flags.
    - Loads the compiled CUBIN and extracts kernel functions.
    
    Subsequent calls are no-ops, so you can safely call this at the start
    of public API functions.
    """
    global _initialized, _module
    global _hungarian_kernel, _auction_kernel, _compute_costs_kernel, _diversity_shortcut_kernel
    
    if _initialized:
        return
    
    import kaggle_support as kgs
    
    print('init LAP CUDA')
    
    # Persist the CUDA source to a stable .cu file inside kgs.temp_dir
    # and compile from that file so profilers can correlate source lines.
    # Use PID to avoid conflicts when multiple processes compile simultaneously
    persist_dir = os.fspath(kgs.temp_dir)
    pid = os.getpid()
    persist_path = os.path.join(persist_dir, f'lap_batch_saved_{pid}.cu')
    cubin_path = os.path.join(persist_dir, f'lap_batch_{pid}.cubin')
    
    # Overwrite the file each time to ensure it matches the compiled source.
    with open(persist_path, 'w', encoding='utf-8') as f:
        f.write(_CUDA_SRC)
    
    # Find nvcc
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH; please install the CUDA toolkit or add nvcc to PATH")
    
    # Detect GPU compute capability
    device = cp.cuda.Device()
    compute_capability_str = device.compute_capability  # e.g., "89" or "120"
    
    # Parse compute capability string into major.minor
    if len(compute_capability_str) == 2:
        major = int(compute_capability_str[0])
        minor = int(compute_capability_str[1])
    else:
        major = int(compute_capability_str[:-1])
        minor = int(compute_capability_str[-1])
    
    sm_arch = f"sm_{compute_capability_str}"
    max_threads_per_block = device.attributes['MaxThreadsPerBlock']
    print(f"Detected GPU compute capability: {major}.{minor} (arch={sm_arch})")
    print(f"GPU max threads per block: {max_threads_per_block}")
    
    # Compile with nvcc
    # Performance flags:
    # -O3: Maximum optimization
    # -use_fast_math: Aggressive math optimizations
    # --extra-device-vectorization: Enable additional vectorization passes
    # --ptxas-options=-v: Verbose register/memory usage output
    # --ptxas-options=--warn-on-spills: Warn if registers spill to local memory
    cmd_cubin = [
        nvcc_path,
        "-O3",
        "-use_fast_math",
        "--extra-device-vectorization",
        "--ptxas-options=-v,--warn-on-spills",
        f"-arch={sm_arch}",
        "-cubin", persist_path,
        "-o", cubin_path
    ]
    
    print("=== Compiling LAP kernels ===")
    print(f"Command: {' '.join(cmd_cubin)}")
    proc = subprocess.run(cmd_cubin, text=True, capture_output=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc cubin compilation failed (exit {proc.returncode})\n{proc.stderr}")
    
    # Print ptxas register usage info (goes to stderr with --ptxas-options=-v)
    if proc.stderr:
        print(proc.stderr)
    if proc.stdout:
        print(proc.stdout)
    
    # Load compiled CUBIN into a CuPy RawModule
    _module = cp.RawModule(path=cubin_path)
    
    # Extract kernel functions
    _hungarian_kernel = _module.get_function("hungarian_kernel")
    _auction_kernel = _module.get_function("auction_kernel")
    _compute_costs_kernel = _module.get_function("compute_costs")
    _diversity_shortcut_kernel = _module.get_function("diversity_shortcut_kernel")
    
    # Print kernel attributes
    def print_kernel_attributes(kernel: cp.RawKernel, name: str):
        """Print diagnostic information about a compiled kernel."""
        print(f"\n--- Kernel: {name} ---")
        print(f"  Max threads per block (kernel): {kernel.max_threads_per_block}")
        print(f"  Num registers: {kernel.num_regs}")
        print(f"  Shared memory (bytes): {kernel.shared_size_bytes}")
        print(f"  Const memory (bytes): {kernel.const_size_bytes}")
        print(f"  Local memory (bytes): {kernel.local_size_bytes}")
    
    print_kernel_attributes(_hungarian_kernel, "hungarian_kernel")
    print_kernel_attributes(_auction_kernel, "auction_kernel")
    print_kernel_attributes(_compute_costs_kernel, "compute_costs")
    print_kernel_attributes(_diversity_shortcut_kernel, "diversity_shortcut_kernel")
    
    _initialized = True


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
    _ensure_initialized()
    import kaggle_support as kgs
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

        if kgs.profiling:
            cp.cuda.Device().synchronize()
        # One block per problem, ONE thread per block
        _hungarian_kernel(
            (batch_size,), (1,),
            (cost_matrices_gpu, costs_work, row_match, col_match, u, v, batch_size, N)
        )

        if kgs.profiling:
            cp.cuda.Device().synchronize()
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

        if kgs.profiling:
            cp.cuda.Device().synchronize()

        _auction_kernel(
            (batch_size,), (1,),
            (cost_matrices_gpu, row_match, col_match, prices, batch_size, N,
             cp.float32(eps_init), cp.float32(eps_final), cp.float32(eps_decay),
             max_rounds, max_iters)
        )
        
        if kgs.profiling:
            cp.cuda.Device().synchronize()
    elif config.algorithm == 'min_cost_row':
        # For each row take min over columns, then sum rows -> total cost.
        min_per_row = cp.amin(cost_matrices_gpu, axis=2)  # shape (batch_size, N)
        total_costs = cp.sum(min_per_row, axis=1).astype(cp.float32)
    elif config.algorithm == 'min_cost_col':
        # For each column take min over rows, then sum cols -> total cost.
        min_per_col = cp.amin(cost_matrices_gpu, axis=1)  # shape (batch_size, N)
        total_costs = cp.sum(min_per_col, axis=1).astype(cp.float32)
    else:
        raise ValueError(f"Unknown LAPConfig.algorithm={config.algorithm!r}")

    # If we ran Hungarian/Auction, compute costs from assignments; otherwise
    # `total_costs` was computed directly above.
    if config.algorithm in ('hungarian', 'auction'):
        blocks = (batch_size + 255) // 256
        _compute_costs_kernel(
            (blocks,), (256,),
            (cost_matrices_gpu, row_match, total_costs, batch_size, N)
        )

    return row_match, total_costs


def compute_diversity_shortcut_kernel(
    pop_xyt: cp.ndarray,
    ref_xyt: cp.ndarray,
    cos_a: float,
    sin_a: float,
    do_mirror: bool
) -> cp.ndarray:
    """
    Compute pairwise diversity costs using fused CUDA kernel (min_cost_row only).
    
    Applies transformation (rotation + mirror) to reference coordinates
    inside the kernel. Uses min_cost_row algorithm: for each pop tree,
    finds minimum distance to any ref tree, then sums.
    
    For min_cost_col, use a different approach (this kernel does not support it).
    
    Parameters
    ----------
    pop_xyt : cp.ndarray
        Shape (N_pop, N_trees, 3) - population coordinates (x, y, theta), must be contiguous
    ref_xyt : cp.ndarray
        Shape (N_ref, N_trees, 3) - reference coordinates (untransformed), must be contiguous
    cos_a : float
        Cosine of the rotation angle
    sin_a : float
        Sine of the rotation angle
    do_mirror : bool
        If True, mirror reference across x-axis before rotation
        
    Returns
    -------
    cp.ndarray
        Shape (N_pop, N_ref) - assignment costs (min_cost_row metric)
    """
    _ensure_initialized()
    N_pop, N_trees, _ = pop_xyt.shape
    N_ref = ref_xyt.shape[0]
    
    # Store original dtype for output
    orig_dtype = pop_xyt.dtype
    
    # Only convert if float64, otherwise assume float32 and contiguous
    if orig_dtype == cp.float64:
        pop_xyt_f32 = cp.ascontiguousarray(pop_xyt.astype(cp.float32))
        ref_xyt_f32 = cp.ascontiguousarray(ref_xyt.astype(cp.float32))
    else:
        # Assume already float32 and contiguous
        pop_xyt_f32 = pop_xyt
        ref_xyt_f32 = ref_xyt
    
    # Output array
    total_pairs = N_pop * N_ref
    costs_out = cp.empty(total_pairs, dtype=cp.float32)

    import kaggle_support as kgs
    if kgs.profiling:
        cp.cuda.Device().synchronize()
    
    # Launch kernel: 1 block per pair, N_trees threads per block
    # Shared memory: 4 * N_trees floats (ref_x, ref_y, ref_t, partial_sums)
    shared_mem_bytes = 4 * N_trees * 4  # 4 arrays * N_trees * sizeof(float)
    
    _diversity_shortcut_kernel(
        (total_pairs,), (N_trees,),
        (pop_xyt_f32, ref_xyt_f32, costs_out,
         N_pop, N_ref, N_trees, 
         cp.float32(cos_a), cp.float32(sin_a),
         1 if do_mirror else 0),
        shared_mem=shared_mem_bytes
    )

    if kgs.profiling:
        cp.cuda.Device().synchronize()
    
    # Reshape and cast back to original dtype
    result = costs_out.reshape(N_pop, N_ref)
    if orig_dtype == cp.float64:
        result = result.astype(orig_dtype)
    
    return result