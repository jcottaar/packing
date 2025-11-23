
# tools/kernel_concurrency_test.py
# Usage: python3 tools/kernel_concurrency_test.py
# Make sure you run this in the Python environment that has CuPy and your repo available.

import time
import math
import numpy as np
import cupy as cp
import sys
sys.path.append('/mnt/d/packing/code/core/')
import pack_cuda

def make_input(N):
    # small random poses in sensible range
    rng = np.random.RandomState(123)
    x = rng.uniform(-5.0, 5.0, size=N)
    y = rng.uniform(-5.0, 5.0, size=N)
    t = rng.uniform(-math.pi, math.pi, size=N)
    xyt = np.stack([x, y, t], axis=1).astype(np.float64)
    return xyt

def run_trial(num_streams, iters_per_stream, xyt1_np, xyt2_np):
    pack_cuda._ensure_initialized()  # make sure module is initialized
    n1 = xyt1_np.shape[0]
    n2 = xyt2_np.shape[0]
    # Flatten 3xN row-major as kernel expects
    xyt1_3xN = cp.ascontiguousarray(cp.asarray(xyt1_np).T).ravel()
    xyt2_3xN = cp.ascontiguousarray(cp.asarray(xyt2_np).T).ravel()
    # dynamic shared memory size (3 rows * n2 cols * 8 bytes/double)
    shared_mem = 3 * n2 * 8

    # Grab raw kernel and device arrays from pack_cuda (uses private names)
    kernel = pack_cuda._overlap_list_total_kernel
    piece_xy = pack_cuda._piece_xy_d
    piece_nverts = pack_cuda._piece_nverts_d
    num_pieces = np.int32(pack_cuda._num_pieces)

    # Precreate per-stream outputs
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    out_totals = [cp.zeros(1, dtype=cp.float64) for _ in range(num_streams)]

    # Warmup single call per stream to get JIT/compilation out of the way
    for s_idx, stream in enumerate(streams):
        with stream:
            kernel(
                (1,), (n1,),
                (xyt1_3xN, np.int32(n1), xyt2_3xN, np.int32(n2), out_totals[s_idx], cp.zeros(1)),
                stream=stream,
                shared_mem=shared_mem
            )
    # Ensure warmup finished
    for s in streams:
        s.synchronize()

    # Timed run: launch iters_per_stream kernels on each stream (back-to-back)
    start = time.time()
    for k in range(iters_per_stream):
        for s_idx, stream in enumerate(streams):
            with stream:
                kernel(
                    (1,), (n1,),
                    (xyt1_3xN, np.int32(n1), xyt2_3xN, np.int32(n2), out_totals[s_idx], cp.zeros(1)),
                    stream=stream,
                    shared_mem=shared_mem
                )
    # Wait for all streams to finish
    for s in streams:
        s.synchronize()
    end = time.time()

    elapsed = end - start
    total_kernels = num_streams * iters_per_stream
    kernels_per_sec = total_kernels / elapsed
    return kernels_per_sec, elapsed

if __name__ == "__main__":
    # Parameters to tune
    N = 1024                       # number of trees (threads per block)
    iters = 1                    # iterations per stream
    xyt = make_input(N)
    pack_cuda._ensure_initialized()
    packs = [1]  # number of concurrent streams to test
    print("Streams\tKernels/sec\tElapsed(s)")
    for s in packs:
        kps, t = run_trial(s, iters, xyt[:200], xyt[:200])
        print(f"{s}\t{int(kps)}\t\t{t:.3f}")