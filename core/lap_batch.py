"""
Batched Linear Assignment Problem (LAP) Solver

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt it 
(including commercially), but must give credit to the original author (Jeroen 
Cottaar) and keep it under this license.

GPU-accelerated Hungarian algorithm for solving multiple linear assignment problems
in parallel. All computation stays on GPU with no CPU transfers. Single kernel launch
solves all problems simultaneously.

Key Features:
- Batched Hungarian algorithm (optimal LAP solution)
- Min-cost heuristics (min_cost_row, min_cost_col) for faster approximations
- Fused diversity kernel for genetic algorithm distance computations
- NVCC compilation with optimization flags for peak performance
"""

import cupy as cp
from dataclasses import dataclass
from typing import Literal
import os
import subprocess
import shutil    
import kaggle_support as kgs


@dataclass
class LAPConfig:
    """
    Configuration for LAP solver algorithm selection.
    
    Attributes:
        algorithm: Solution method - 'hungarian' (optimal), 'min_cost_row', or 'min_cost_col'
        use_diversity_kernel: Use fused CUDA kernel for diversity computations (min_cost only)
    """
    algorithm: Literal['hungarian', 'min_cost_row', 'min_cost_col'] = 'hungarian'
    use_diversity_kernel: bool = True


# ---------------------------------------------------------------------------
# CUDA source code - all kernels in one compilation unit
# ---------------------------------------------------------------------------

_CUDA_SRC = r'''
extern "C" {

// =============================================================================
// Hungarian Algorithm Kernel
// =============================================================================
// Solves optimal linear assignment using Kuhn-Munkres algorithm.
// Parallelism: One block per problem (batch), single thread per block.
// Hungarian is inherently sequential; we parallelize across the batch dimension.
//
// Algorithm:
// 1. Initialize potentials (u, v) and matches to zero/-1
// 2. For each row, find augmenting path using Dijkstra-like search
// 3. Update dual potentials to maintain reduced costs >= 0
// 4. Trace back and flip alternating path to increase matching size
//
// Memory: Uses stack arrays (size 200) for path finding - limits N <= 200
__global__ void hungarian_kernel(
    const float* __restrict__ costs_in,  // (batch_size, N, N) - input cost matrices
    float* __restrict__ costs,           // (batch_size, N, N) - working copy
    int* __restrict__ row_match,         // (batch_size, N) - row->col assignments
    int* __restrict__ col_match,         // (batch_size, N) - col->row assignments
    float* __restrict__ u,               // (batch_size, N) - row dual potentials
    float* __restrict__ v,               // (batch_size, N) - column dual potentials
    const int batch_size,
    const int N
) {
    const int bid = blockIdx.x;
    if (bid >= batch_size) return;
    
    // Single thread per block - Hungarian is inherently sequential
    if (threadIdx.x != 0) return;
    
    // Get pointers to this problem's data
    const float* my_costs_in = costs_in + bid * N * N;
    float* my_costs = costs + bid * N * N;
    int* my_row_match = row_match + bid * N;
    int* my_col_match = col_match + bid * N;
    float* my_u = u + bid * N;
    float* my_v = v + bid * N;
    
    // Initialize: copy costs, set matches to -1, potentials to 0
    for (int i = 0; i < N; i++) {
        my_row_match[i] = -1;
        my_col_match[i] = -1;
        my_u[i] = 0.0f;
        my_v[i] = 0.0f;
        for (int j = 0; j < N; j++) {
            my_costs[i * N + j] = my_costs_in[i * N + j];
        }
    }
    
    // Kuhn-Munkres: augment matching for each row
    for (int i = 0; i < N; i++) {
        // Stack arrays for augmenting path search (limits N <= 200)
        int p[200];       // Parent pointers (backtracking array)
        float minv[200];  // Minimum slack (reduced cost) to each column
        bool used[200];   // Columns already in alternating tree
        
        // Initialize search state
        for (int j = 0; j < N; j++) {
            minv[j] = 1e30f;
            used[j] = false;
            p[j] = -1;
        }
        
        int cur_row = i;
        int cur_col = -1;
        
        // Dijkstra-style search for augmenting path from unmatched row i
        while (true) {
            int best_col = -1;
            float best_val = 1e30f;
            
            // Find column with minimum slack among unused columns
            for (int j = 0; j < N; j++) {
                if (used[j]) continue;
                
                // Compute reduced cost: c[cur_row][j] - u[cur_row] - v[j]
                float reduced = my_costs[cur_row * N + j] - my_u[cur_row] - my_v[j];
                if (reduced < minv[j]) {
                    minv[j] = reduced;
                    p[j] = cur_row;  // Track parent for path reconstruction
                }
                if (minv[j] < best_val) {
                    best_val = minv[j];
                    best_col = j;
                }
            }
            
            // Update dual potentials to maintain feasibility
            // Matched rows/cols: increase potential by best_val
            // Unmatched cols: decrease slack by best_val
            for (int j = 0; j < N; j++) {
                if (used[j]) {
                    my_u[my_col_match[j]] += best_val;  // Matched row
                    my_v[j] -= best_val;                 // Matched column
                } else {
                    minv[j] -= best_val;                 // Unmatched column slack
                }
            }
            my_u[i] += best_val;  // Current unmatched row
            
            // Add best column to alternating tree
            used[best_col] = true;
            cur_col = best_col;
            cur_row = my_col_match[cur_col];
            
            if (cur_row < 0) break;  // Found unmatched column - augmenting path complete
        }
        
        // Flip alternating path to increase matching size by 1
        while (cur_col >= 0) {
            int prev_row = p[cur_col];
            int prev_col = my_row_match[prev_row];
            my_row_match[prev_row] = cur_col;
            my_col_match[cur_col] = prev_row;
            cur_col = prev_col;
        }
    }
}


// =============================================================================
// Cost Computation Kernel
// =============================================================================
// Computes total assignment cost from cost matrices and assignments.
// Parallelism: Multiple threads process different problems in parallel.
__global__ void compute_costs(
    const float* __restrict__ costs,      // (batch_size, N, N) - cost matrices
    const int* __restrict__ assignments,  // (batch_size, N) - row->col assignments
    float* __restrict__ total_costs,      // (batch_size,) - output: sum of assigned costs
    const int batch_size,
    const int N
) {
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= batch_size) return;
    
    // Get pointers to this problem's data
    const float* my_costs = costs + bid * N * N;
    const int* my_assign = assignments + bid * N;
    
    // Sum costs for all assigned pairs
    float total = 0.0f;
    for (int row = 0; row < N; row++) {
        int col = my_assign[row];
        if (col >= 0 && col < N) {
            total += my_costs[row * N + col];
        }
    }
    total_costs[bid] = total;
}


// =============================================================================
// Diversity Shortcut Kernel (Fused Transform + Min-Cost-Row)
// =============================================================================
// Computes pairwise diversity metric by fusing coordinate transformation with
// min-cost-row LAP approximation. Significantly faster than separate operations.
//
// Parallelism: One block per (pop_idx, ref_idx) pair, N_trees threads per block
//
// Algorithm:
// Phase 1: Cooperatively transform reference coordinates (rotation + mirror)
// Phase 2: Each thread finds min distance from one pop tree to all ref trees
// Phase 3: Thread 0 sums all minimum distances
//
// Only supports min_cost_row metric (for each pop tree, min dist to any ref tree).
// For min_cost_col or Hungarian, use separate transformation + solve_lap_batch.
//
// Memory: Uses 4 * N_trees floats of shared memory per block
__global__ void diversity_shortcut_kernel(
    const float* __restrict__ pop_xyt,    // (N_pop, N_trees, 3) - population [x,y,theta]
    const float* __restrict__ ref_xyt,    // (N_ref, N_trees, 3) - reference (untransformed)
    float* __restrict__ costs_out,        // (N_pop * N_ref,) - output diversity costs
    const int N_pop,
    const int N_ref,
    const int N_trees,
    const float cos_a,                    // cos(rotation_angle)
    const float sin_a,                    // sin(rotation_angle)
    const int do_mirror                   // 1=mirror across x-axis, 0=no mirror
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int total_pairs = N_pop * N_ref;
    if (bid >= total_pairs) return;
    
    // Decode block index into (pop_idx, ref_idx)
    const int pop_idx = bid / N_ref;
    const int ref_idx = bid % N_ref;
    
    // Get pointers to this pair's coordinates (stride 3 for xyt layout)
    const float* my_pop = pop_xyt + pop_idx * N_trees * 3;
    const float* my_ref = ref_xyt + ref_idx * N_trees * 3;
    
    const float PI = 3.14159265358979323846f;
    const float TWO_PI = 2.0f * PI;
    
    // Compute rotation angle from sin/cos for theta offset
    float rot_angle = atan2f(sin_a, cos_a);
    
    // Shared memory layout: [ref_x, ref_y, ref_theta, partial_sums]
    // Each array has N_trees elements
    extern __shared__ float shared_mem[];
    float* ref_x = shared_mem;
    float* ref_y = shared_mem + N_trees;
    float* ref_t = shared_mem + 2 * N_trees;
    float* partial_sums = shared_mem + 3 * N_trees;
    
    // -------------------------------------------------------------------------
    // Phase 1: Cooperative transformation - each thread transforms one ref tree
    // -------------------------------------------------------------------------
    if (tid < N_trees) {
        float rx = my_ref[tid * 3 + 0];  // x coordinate
        float ry = my_ref[tid * 3 + 1];  // y coordinate
        float rt = my_ref[tid * 3 + 2];  // theta (rotation angle)
        
        // Apply mirror transformation if requested
        if (do_mirror) {
            ry = -ry;        // Flip across x-axis
            rt = PI - rt;    // Adjust angle for mirror
        }
        
        // Apply 2D rotation
        float rx_rot = rx * cos_a - ry * sin_a;
        float ry_rot = rx * sin_a + ry * cos_a;
        float rt_rot = rt + rot_angle;
        
        // Normalize angle to [-pi, pi]
        rt_rot = fmodf(rt_rot + PI, TWO_PI);
        if (rt_rot < 0) rt_rot += TWO_PI;
        rt_rot -= PI;
        
        // Store transformed coordinates in shared memory
        ref_x[tid] = rx_rot;
        ref_y[tid] = ry_rot;
        ref_t[tid] = rt_rot;
    }
    
    __syncthreads();  // Wait for all transformations to complete
    
    // -------------------------------------------------------------------------
    // Phase 2: Each thread finds minimum distance from its pop tree to all refs
    // -------------------------------------------------------------------------
    float my_min_dist = 0.0f;
    if (tid < N_trees) {
        float px = my_pop[tid * 3 + 0];
        float py = my_pop[tid * 3 + 1];
        float pt = my_pop[tid * 3 + 2];
        
        float min_sq = 1e30f;
        
        // Find minimum squared distance to any reference tree
        for (int j = 0; j < N_trees; j++) {
            float dx = px - ref_x[j];
            float dy = py - ref_y[j];
            float dt = pt - ref_t[j];
            
            // Wrap angle difference to [-pi, pi]
            dt = fmodf(dt + PI, TWO_PI);
            if (dt < 0) dt += TWO_PI;
            dt -= PI;
            
            // Euclidean distance in (x, y, theta) space
            float sq = dx*dx + dy*dy + dt*dt;
            if (sq < min_sq) min_sq = sq;
        }
        
        my_min_dist = sqrtf(min_sq);
    }
    
    // Store in shared memory for reduction
    partial_sums[tid] = my_min_dist;
    
    __syncthreads();  // Wait for all minimum distances
    
    // -------------------------------------------------------------------------
    // Phase 3: Thread 0 sums all minimum distances (simple sequential reduction)
    // -------------------------------------------------------------------------
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
_compute_costs_kernel: cp.RawKernel | None = None
_diversity_shortcut_kernel: cp.RawKernel | None = None
_initialized: bool = False


def _ensure_initialized() -> None:
    """
    Initialize CUDA kernels on first use (lazy compilation).
    
    This function:
    1. Persists CUDA source to .cu file for profiler source correlation
    2. Detects GPU compute capability and compiles with nvcc
    3. Loads compiled CUBIN and extracts kernel functions
    4. Prints kernel resource usage (registers, shared memory)
    
    Subsequent calls are no-ops. Safe to call at start of public API functions.
    Uses PID in filenames to avoid conflicts in multiprocess scenarios.
    
    Raises:
        RuntimeError: If nvcc not found or compilation fails
    """
    global _initialized, _module
    global _hungarian_kernel, _compute_costs_kernel, _diversity_shortcut_kernel
    
    if _initialized:
        return
    
    print('init LAP CUDA')
    
    # Persist CUDA source for profiler correlation
    persist_dir = os.fspath(kgs.temp_dir)
    pid = os.getpid()
    persist_path = os.path.join(persist_dir, f'lap_batch_saved_{pid}.cu')
    cubin_path = os.path.join(persist_dir, f'lap_batch_{pid}.cubin')
    
    with open(persist_path, 'w', encoding='utf-8') as f:
        f.write(_CUDA_SRC)
    
    # Locate nvcc compiler
    os.environ['PATH'] = '/usr/local/cuda/bin:' + os.environ.get('PATH', '')
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError(
            "nvcc not found in PATH; install CUDA toolkit or add nvcc to PATH"
        )
    
    # Detect GPU compute capability
    device = cp.cuda.Device()
    compute_capability_str = device.compute_capability  # e.g., "89" or "120"
    
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
    
    # Compile with aggressive optimization flags
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
        raise RuntimeError(
            f"nvcc cubin compilation failed (exit {proc.returncode})\n{proc.stderr}"
        )
    
    # Print compilation diagnostics
    if proc.stderr:
        print(proc.stderr)
    if proc.stdout:
        print(proc.stdout)
    
    # Load compiled CUBIN
    _module = cp.RawModule(path=cubin_path)
    
    # Extract kernel functions
    _hungarian_kernel = _module.get_function("hungarian_kernel")
    _compute_costs_kernel = _module.get_function("compute_costs")
    _diversity_shortcut_kernel = _module.get_function("diversity_shortcut_kernel")
    
    # Print kernel resource usage
    def print_kernel_attributes(kernel: cp.RawKernel, name: str):
        """Print diagnostic information about compiled kernel resources."""
        print(f"\n--- Kernel: {name} ---")
        print(f"  Max threads per block (kernel): {kernel.max_threads_per_block}")
        print(f"  Num registers: {kernel.num_regs}")
        print(f"  Shared memory (bytes): {kernel.shared_size_bytes}")
        print(f"  Const memory (bytes): {kernel.const_size_bytes}")
        print(f"  Local memory (bytes): {kernel.local_size_bytes}")
    
    print_kernel_attributes(_hungarian_kernel, "hungarian_kernel")
    print_kernel_attributes(_compute_costs_kernel, "compute_costs")
    print_kernel_attributes(_diversity_shortcut_kernel, "diversity_shortcut_kernel")
    
    _initialized = True


def solve_lap_batch(
    cost_matrices_gpu: cp.ndarray, 
    config: LAPConfig | None = None
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Solve batched linear assignment problems using GPU Hungarian algorithm.
    
    Single kernel launch solves all problems in parallel with no CPU transfers.
    Supports exact Hungarian algorithm or faster min-cost heuristics.
    
    Args:
        cost_matrices_gpu: Cost matrices, shape (batch_size, N, N), dtype float32 or float64
                          N must be <= 200 (Hungarian algorithm stack limit)
        config: Algorithm configuration (default: Hungarian with diversity kernel)
    
    Returns:
        tuple containing:
            - assignments: Shape (batch_size, N), row-to-column assignments (dtype int32)
            - costs: Shape (batch_size,), total assignment cost per problem (dtype float32)
    
    Notes:
        - Hungarian: Optimal solution, slower
        - min_cost_row: For each row, pick min column; fast approximation
        - min_cost_col: For each column, pick min row; fast approximation
    """
    _ensure_initialized()
    
    if config is None:
        config = LAPConfig()

    batch_size, N, _ = cost_matrices_gpu.shape
    assert N <= 200, f"N={N} exceeds maximum of 200"
    
    # Ensure float32 and contiguous layout
    if cost_matrices_gpu.dtype != cp.float32:
        cost_matrices_gpu = cost_matrices_gpu.astype(cp.float32)
    cost_matrices_gpu = cp.ascontiguousarray(cost_matrices_gpu)
    
    # Allocate output arrays
    row_match = cp.full((batch_size, N), -1, dtype=cp.int32)  # Shape: (batch_size, N)
    col_match = cp.full((batch_size, N), -1, dtype=cp.int32)  # Shape: (batch_size, N)
    total_costs = cp.empty(batch_size, dtype=cp.float32)      # Shape: (batch_size,)

    # Execute selected algorithm
    if config.algorithm == 'hungarian':
        # Allocate working memory for Hungarian algorithm
        costs_work = cp.empty_like(cost_matrices_gpu)
        u = cp.zeros((batch_size, N), dtype=cp.float32)  # Row potentials
        v = cp.zeros((batch_size, N), dtype=cp.float32)  # Column potentials
        
        # Launch Hungarian kernel: 1 block per problem, 1 thread per block
        _hungarian_kernel(
            (batch_size,), (1,),
            (cost_matrices_gpu, costs_work, row_match, col_match, u, v, batch_size, N)
        )
        
    elif config.algorithm == 'min_cost_row':
        # Fast heuristic: for each row, take minimum column
        min_per_row = cp.amin(cost_matrices_gpu, axis=2)  # Shape: (batch_size, N)
        total_costs = cp.sum(min_per_row, axis=1).astype(cp.float32)
        
    elif config.algorithm == 'min_cost_col':
        # Fast heuristic: for each column, take minimum row
        min_per_col = cp.amin(cost_matrices_gpu, axis=1)  # Shape: (batch_size, N)
        total_costs = cp.sum(min_per_col, axis=1).astype(cp.float32)
        
    else:
        raise ValueError(f"Unknown LAPConfig.algorithm={config.algorithm!r}")

    # Compute costs from assignments if using Hungarian
    if config.algorithm == 'hungarian':
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
    Compute pairwise diversity costs using fused CUDA kernel.
    
    Applies transformation (rotation + optional mirror) to reference coordinates
    inside the kernel. Uses min_cost_row algorithm: for each population tree,
    finds minimum distance to any reference tree, then sums across trees.
    
    This is significantly faster than separate transformation + LAP solve.
    Only supports min_cost_row metric (not min_cost_col or Hungarian).
    
    Args:
        pop_xyt: Population coordinates, shape (N_pop, N_trees, 3), dtype float32/float64
                 Must be contiguous. Layout: [x, y, theta] per tree
        ref_xyt: Reference coordinates, shape (N_ref, N_trees, 3), dtype float32/float64
                 Must be contiguous. Will be transformed inside kernel
        cos_a: Cosine of rotation angle to apply to reference
        sin_a: Sine of rotation angle to apply to reference
        do_mirror: If True, mirror reference across x-axis before rotation
    
    Returns:
        cp.ndarray: Diversity costs, shape (N_pop, N_ref), same dtype as inputs
                    Each entry is sum of min distances for min_cost_row metric
    
    Notes:
        - Uses shared memory for transformed coordinates (4 * N_trees floats)
        - Launches N_pop * N_ref blocks with N_trees threads each
        - Converts to float32 internally if inputs are float64
    """
    _ensure_initialized()
    
    N_pop, N_trees, _ = pop_xyt.shape
    N_ref = ref_xyt.shape[0]
    
    orig_dtype = pop_xyt.dtype
    
    # Convert to float32 for kernel if needed
    if orig_dtype == cp.float64:
        pop_xyt_f32 = cp.ascontiguousarray(pop_xyt.astype(cp.float32))
        ref_xyt_f32 = cp.ascontiguousarray(ref_xyt.astype(cp.float32))
    else:
        pop_xyt_f32 = pop_xyt
        ref_xyt_f32 = ref_xyt
    
    # Allocate output
    total_pairs = N_pop * N_ref
    costs_out = cp.empty(total_pairs, dtype=cp.float32)  # Shape: (N_pop * N_ref,)

    # Launch kernel configuration
    shared_mem_bytes = 4 * N_trees * 4  # 4 arrays * N_trees * sizeof(float)
    
    _diversity_shortcut_kernel(
        (total_pairs,), (N_trees,),
        (pop_xyt_f32, ref_xyt_f32, costs_out,
         N_pop, N_ref, N_trees, 
         cp.float32(cos_a), cp.float32(sin_a),
         1 if do_mirror else 0),
        shared_mem=shared_mem_bytes
    )
    
    # Reshape and convert back to original dtype
    result = costs_out.reshape(N_pop, N_ref)
    if orig_dtype == cp.float64:
        result = result.astype(orig_dtype)
    
    return result