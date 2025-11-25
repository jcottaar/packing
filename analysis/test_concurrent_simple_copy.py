# Simple kernel concurrency test - block size scaling and multi-stream parallelism
# Supports both simple kernel and overlap kernel from pack_cuda

import time
import math
import numpy as np
import cupy as cp
import sys
import os
import subprocess
import shutil
sys.path.insert(0, os.path.join(os.getcwd(), '../core'))
sys.path.append('/mnt/d/packing/code/core/')

import pack_cuda
import kaggle_support as kgs
kgs.profiling=False

# ============================================================================
# KERNEL SETUP
# ============================================================================

# Define simple work kernel with multi-ensemble support (like pack_cuda.py)
simple_kernel_code = r'''
extern "C" __global__
void multi_simple_work(
    const float** input_list,
    float** output_list,
    const int* n_list,
    int work_factor,
    int num_ensembles
) {
    // Each block handles one ensemble (same as pack_cuda.py)
    int ensemble_idx = blockIdx.x;
    if (ensemble_idx >= num_ensembles) return;
    
    const float* input = input_list[ensemble_idx];
    float* output = output_list[ensemble_idx];
    int n = n_list[ensemble_idx];
    
    int idx = threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // Do some arithmetic work to make it non-trivial
        for (int i = 0; i < work_factor; i++) {
            val = val * 1.001f + 0.001f;
            val = sqrtf(val * val + 1.0f);
        }
        output[idx] = val;
    }
}
'''

# Module-level variables (same pattern as pack_cuda.py)
_simple_raw_module = None
_simple_kernel = None
_simple_initialized = False

# Compile the simple kernel using the SAME method as pack_cuda.py:
# Write to file, compile with nvcc to PTX, load with cp.RawModule
def _ensure_simple_initialized():
    global _simple_raw_module, _simple_kernel, _simple_initialized
    
    if _simple_initialized:
        return
    
    persist_dir = '/mnt/d/packing/temp'
    os.makedirs(persist_dir, exist_ok=True)
    
    persist_path = os.path.join(persist_dir, 'simple_kernel.cu')
    with open(persist_path, 'w') as f:
        f.write(simple_kernel_code)
    
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError("nvcc not found in PATH")
    
    ptx_path = os.path.join(persist_dir, 'simple_kernel.ptx')
    
    # First compile to cubin to get ptxas verbose output (same as pack_cuda.py)
    cubin_path = os.path.join(persist_dir, 'simple_kernel.cubin')
    cmd_cubin = [nvcc_path, "-O3", "-use_fast_math", "--ptxas-options=-v", "-arch=sm_89", "-cubin", persist_path, "-o", cubin_path]
    
    print("=== Compiling simple kernel (cubin for ptxas info) ===")
    print(f"Command: {' '.join(cmd_cubin)}")
    proc = subprocess.run(cmd_cubin, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc cubin compilation failed (exit {proc.returncode})")
    
    # Now compile to PTX for actual use
    cmd = [nvcc_path, "-O3", "-use_fast_math", "-arch=sm_89", "-ptx", persist_path, "-o", ptx_path]
    proc = subprocess.run(cmd, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc PTX compilation failed (exit {proc.returncode})")
    
    # Load compiled PTX into a CuPy RawModule (same as pack_cuda.py)
    _simple_raw_module = cp.RawModule(path=ptx_path)
    _simple_kernel = _simple_raw_module.get_function("multi_simple_work")
    _simple_initialized = True

# Public API function (same pattern as pack_cuda.py)
def simple_work_multi_ensemble(input_list, n_list, work_factor, stream=None):
    """Compute simple work for multiple ensembles in parallel.
    
    This launches one GPU block per ensemble, matching pack_cuda.py's pattern.
    
    Parameters
    ----------
    input_list : list of cp.ndarray
        List of input arrays, each shape (n_i,) for ensemble i.
    n_list : list of int
        List of array sizes.
    work_factor : int
        Number of work iterations per element.
    stream : cp.cuda.Stream or None
        CUDA stream for async execution.
    
    Returns
    -------
    output_list : list of cp.ndarray
        List of output arrays, each shape (n_i,).
    """
    _ensure_simple_initialized()
    
    num_ensembles = len(input_list)
    
    # Allocate output arrays
    output_list = [cp.zeros_like(inp) for inp in input_list]
    
    # Create device arrays of pointers (same as pack_cuda.py)
    input_ptrs = cp.array([inp.data.ptr for inp in input_list], dtype=cp.uint64)
    output_ptrs = cp.array([out.data.ptr for out in output_list], dtype=cp.uint64)
    n_array = cp.array(n_list, dtype=cp.int32)
    
    # Launch kernel: one block per ensemble, threads per block = max(n_list)
    max_n = max(n_list)
    threads_per_block = min(max_n, 1024)  # CUDA limit
    
    _simple_kernel(
        (num_ensembles,),  # grid: one block per ensemble
        (threads_per_block,),  # block size
        (input_ptrs, output_ptrs, n_array, work_factor, num_ensembles),
        stream=stream
    )
    
    return output_list

# Initialize kernels
_ensure_simple_initialized()
pack_cuda.USE_FLOAT32=True
pack_cuda._ensure_initialized()

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def run_trial_multi_stream(num_streams, iters_per_stream, num_blocks_per_stream, n_threads, work_factor, kernel_type):
    """Test: launch multiple kernels in parallel via streams (each kernel has multiple blocks)"""
    
    if kernel_type == 'simple':
        # Use multi-ensemble interface like pack_cuda
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Create ensemble data lists (each ensemble has n_threads elements)
        input_list = [cp.random.randn(n_threads, dtype=cp.float32) for _ in range(total_ensembles)]
        n_list = [n_threads] * total_ensembles
        
        # Warmup
        simple_work_multi_ensemble(input_list, n_list, work_factor)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            simple_work_multi_ensemble(input_list, n_list, work_factor)
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    elif kernel_type == 'simple_direct':
        # Call kernel directly (bypass wrapper) - like old cp.RawKernel approach
        _ensure_simple_initialized()
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Create ensemble data lists (each ensemble has n_threads elements)
        input_list = [cp.random.randn(n_threads, dtype=cp.float32) for _ in range(total_ensembles)]
        output_list = [cp.zeros(n_threads, dtype=cp.float32) for _ in range(total_ensembles)]
        n_list = [n_threads] * total_ensembles
        
        # Create device arrays of pointers
        input_ptrs = cp.array([inp.data.ptr for inp in input_list], dtype=cp.uint64)
        output_ptrs = cp.array([out.data.ptr for out in output_list], dtype=cp.uint64)
        n_array = cp.array(n_list, dtype=cp.int32)
        
        # Warmup
        _simple_kernel(
            (total_ensembles,),
            (n_threads,),
            (input_ptrs, output_ptrs, n_array, work_factor, total_ensembles)
        )
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            _simple_kernel(
                (total_ensembles,),
                (n_threads,),
                (input_ptrs, output_ptrs, n_array, work_factor, total_ensembles)
            )
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
        
    elif kernel_type == 'overlap':
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Determine dtype
        dtype = cp.float32 if pack_cuda.USE_FLOAT32 else cp.float64
        
        # Create ensemble data as 3D arrays (n_ensembles, n_trees, 3)
        xyt1 = cp.ascontiguousarray(cp.random.randn(total_ensembles, num_trees, 3).astype(dtype))
        xyt2 = cp.ascontiguousarray(cp.random.randn(total_ensembles, num_trees, 3).astype(dtype))
        
        # Warmup
        pack_cuda.overlap_multi_ensemble(xyt1, xyt2, compute_grad=True)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda.overlap_multi_ensemble(xyt1, xyt2, compute_grad=True)
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    elif kernel_type == 'boundary':
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Determine dtype
        dtype = cp.float32 if pack_cuda.USE_FLOAT32 else cp.float64
        
        # Create ensemble data as 3D array (n_ensembles, n_trees, 3)
        xyt = cp.ascontiguousarray(cp.random.randn(total_ensembles, num_trees, 3).astype(dtype))
        h = cp.full(total_ensembles, 10.0, dtype=dtype)
        
        # Warmup
        pack_cuda.boundary_multi_ensemble(xyt, h, compute_grad=True)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda.boundary_multi_ensemble(xyt, h, compute_grad=True)
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    elif kernel_type == 'boundary_direct':
        # Call boundary kernel directly (bypass wrapper) - new interface with 3D arrays
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        dtype = cp.float32 if pack_cuda.USE_FLOAT32 else cp.float64
        
        # Pre-allocate single 3D array (n_ensembles, n_trees, 3) in C-contiguous layout
        xyt = cp.ascontiguousarray(cp.random.randn(total_ensembles, num_trees, 3).astype(dtype))
        h_array = cp.full(total_ensembles, 10.0, dtype=dtype)
        
        # Pre-allocate outputs
        out_totals = cp.zeros(total_ensembles, dtype=dtype)
        out_grads = cp.zeros((total_ensembles, num_trees, 3), dtype=dtype)
        out_grad_h = cp.zeros(total_ensembles, dtype=dtype)
        
        threads_per_block = num_trees * 4
        
        # Warmup
        pack_cuda._multi_boundary_list_total_kernel(
            (total_ensembles,),
            (threads_per_block,),
            (xyt, np.int32(num_trees), h_array, out_totals, out_grads, out_grad_h, np.int32(total_ensembles))
        )
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda._multi_boundary_list_total_kernel(
                (total_ensembles,),
                (threads_per_block,),
                (xyt, np.int32(num_trees), h_array, out_totals, out_grads, out_grad_h, np.int32(total_ensembles))
            )
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    elif kernel_type == 'simple_dummy':
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Create ensemble data lists (same as boundary)
        xyt_list = [cp.random.randn(num_trees, 3) for _ in range(total_ensembles)]
        h_list = [10.0] * total_ensembles
        
        # Warmup
        pack_cuda.simple_dummy_multi_ensemble(xyt_list, h_list, compute_grad=True)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda.simple_dummy_multi_ensemble(xyt_list, h_list, compute_grad=True)
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
    elapsed = end - start
    total_kernel_calls = iters_per_stream * num_streams
    calls_per_sec = total_kernel_calls / elapsed
    return calls_per_sec, elapsed

# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Test parameters
    # For simple kernel: n_threads is threads per block
    # For overlap kernel: n_threads is trees per ensemble (kernel uses n_threads * 4 actual threads)
    n_threads = 20  # Thread count per block (simple) or trees per ensemble (overlap)
    work_factor = 1000000  # Work iterations per thread
    iters = 100

    # KERNEL TYPE FLAG: 'simple', 'simple_direct', 'overlap', 'boundary', 'boundary_direct', or 'simple_dummy'
    for KERNEL_TYPE in ['boundary']:#'simple_dummy', 'simple', 'simple_direct']:

        print("=" * 60)
        print(f"KERNEL TYPE: {KERNEL_TYPE.upper()}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("TEST 1: Block Count Scaling (Single Stream, Multiple Blocks)")
        print(f"Config: {n_threads} threads/block, {work_factor} work iterations")
        print("=" * 60)
        print("Blocks\tKernels/sec\tElapsed(s)")
        block_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        for b in block_counts:
            # Use run_trial_multi_stream with 1 stream for single stream test
            kps, t = run_trial_multi_stream(1, iters, b, n_threads, work_factor, kernel_type=KERNEL_TYPE)
            cp.cuda.Device().synchronize()
            print(f"{b}\t{int(kps)}\t\t{t:.3f}")

        print("\n" + "=" * 60)
        print("TEST 2: Parallel Kernel Execution (Multiple Streams, Each with Multiple Blocks)")
        print(f"Config: {n_threads} threads/block, {work_factor} work iterations")
        print("=" * 60)
        print("Streams\tBlocks/Stream\tKernels/sec\tElapsed(s)")
        num_blocks_per_stream = 60  # Fixed block count per kernel
        stream_counts = [1, 2, 4, 8, 16, 32, 64]
        for num_streams in stream_counts:
            kps, t = run_trial_multi_stream(num_streams, iters, num_blocks_per_stream, n_threads, work_factor, kernel_type=KERNEL_TYPE)
            cp.cuda.Device().synchronize()
            print(f"{num_streams}\t{num_blocks_per_stream}\t\t{int(kps)}\t\t{t:.3f}")

        # print("\n" + "=" * 60)
        # print("GPU Info")
        # print("=" * 60)
        # props = cp.cuda.runtime.getDeviceProperties(0)
        # print(f"GPU: {props['name'].decode()}")
        # print(f"Number of SMs: {props['multiProcessorCount']}")
