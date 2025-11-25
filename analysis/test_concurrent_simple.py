# Simple kernel concurrency test - block size scaling and multi-stream parallelism
# Supports both simple kernel and overlap kernel from pack_cuda

import time
import math
import numpy as np
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), '../core'))
sys.path.append('/mnt/d/packing/code/core/')

import pack_cuda
import kaggle_support as kgs

# ============================================================================
# KERNEL SETUP
# ============================================================================

# Define simple work kernel with block indexing for multi-block execution
simple_kernel_code = r'''
extern "C" __global__
void simple_work(const float* input, float* output, int n, int work_factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

# Compile the simple kernel
simple_kernel = cp.RawKernel(simple_kernel_code, 'simple_work')

# Initialize pack_cuda
pack_cuda.USE_FLOAT32=True
pack_cuda._ensure_initialized()

# ============================================================================
# HELPER FUNCTIONS FOR DATA PREPARATION
# ============================================================================


def prepare_simple_data_multi_stream(num_streams, num_blocks_per_stream, n_threads):
    """Prepare data for simple kernel with multiple streams"""
    total_size_per_stream = num_blocks_per_stream * n_threads
    inputs = [cp.random.randn(total_size_per_stream, dtype=cp.float32) for _ in range(num_streams)]
    outputs = [cp.zeros(total_size_per_stream, dtype=cp.float32) for _ in range(num_streams)]
    return inputs, outputs, total_size_per_stream

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def run_trial_multi_stream(num_streams, iters_per_stream, num_blocks_per_stream, n_threads, work_factor, kernel_type):
    """Test: launch multiple kernels in parallel via streams (each kernel has multiple blocks)"""
    
    if kernel_type == 'simple':
        inputs, outputs, total_size_per_stream = prepare_simple_data_multi_stream(num_streams, num_blocks_per_stream, n_threads)
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
        
        # Warmup
        for s_idx, stream in enumerate(streams):
            simple_kernel(
                (num_blocks_per_stream,), (n_threads,),
                (inputs[s_idx], outputs[s_idx], total_size_per_stream, work_factor),
                stream=stream
            )
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            for s_idx, stream in enumerate(streams):
                simple_kernel(
                    (num_blocks_per_stream,), (n_threads,),
                    (inputs[s_idx], outputs[s_idx], total_size_per_stream, work_factor),
                    stream=stream
                )
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
        
    elif kernel_type == 'overlap':
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Create ensemble data lists
        xyt1_list = [cp.random.randn(num_trees, 3) for _ in range(total_ensembles)]
        xyt2_list = [cp.random.randn(num_trees, 3) for _ in range(total_ensembles)]
        
        # Warmup
        pack_cuda.overlap_multi_ensemble(xyt1_list, xyt2_list, compute_grad=True)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda.overlap_multi_ensemble(xyt1_list, xyt2_list, compute_grad=True)
        cp.cuda.Device().synchronize()
        end = time.perf_counter()
    
    elif kernel_type == 'boundary':
        num_trees = n_threads // 4
        total_ensembles = num_streams * num_blocks_per_stream
        
        # Create ensemble data lists
        xyt_list = [cp.random.randn(num_trees, 3) for _ in range(total_ensembles)]
        h_list = [10.0] * total_ensembles  # boundary size
        
        # Warmup
        pack_cuda.boundary_multi_ensemble(xyt_list, h_list, compute_grad=True)
        cp.cuda.Device().synchronize()
        
        # Timed run
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for i in range(iters_per_stream):
            pack_cuda.boundary_multi_ensemble(xyt_list, h_list, compute_grad=True)
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
    work_factor = 10000  # Work iterations per thread
    iters = 100

    # KERNEL TYPE FLAG: 'simple', 'overlap', 'boundary', or 'simple_dummy'
    for KERNEL_TYPE in ['simple_dummy', 'simple']:#, 'simple', 'overlap']:

        print("=" * 60)
        print(f"KERNEL TYPE: {KERNEL_TYPE.upper()}")
        print("=" * 60)

        # print("\n" + "=" * 60)
        # print("TEST 1: Block Count Scaling (Single Stream, Multiple Blocks)")
        # print(f"Config: {n_threads} threads/block, {work_factor} work iterations")
        # print("=" * 60)
        # print("Blocks\tKernels/sec\tElapsed(s)")
        # block_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # for b in block_counts:
        #     # Use run_trial_multi_stream with 1 stream for single stream test
        #     kps, t = run_trial_multi_stream(1, iters, b, n_threads, work_factor, kernel_type=KERNEL_TYPE)
        #     cp.cuda.Device().synchronize()
        #     print(f"{b}\t{int(kps)}\t\t{t:.3f}")

        print("\n" + "=" * 60)
        print("TEST 2: Parallel Kernel Execution (Multiple Streams, Each with Multiple Blocks)")
        print(f"Config: {n_threads} threads/block, {work_factor} work iterations")
        print("=" * 60)
        print("Streams\tBlocks/Stream\tKernels/sec\tElapsed(s)")
        num_blocks_per_stream = 64  # Fixed block count per kernel
        stream_counts = [1, 2, 4, 8]
        for num_streams in stream_counts:
            kps, t = run_trial_multi_stream(num_streams, iters, num_blocks_per_stream, n_threads, work_factor, kernel_type=KERNEL_TYPE)
            cp.cuda.Device().synchronize()
            print(f"{num_streams}\t{num_blocks_per_stream}\t\t{int(kps)}\t\t{t:.3f}")

        print("\n" + "=" * 60)
        print("GPU Info")
        print("=" * 60)
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"GPU: {props['name'].decode()}")
        print(f"Number of SMs: {props['multiProcessorCount']}")
