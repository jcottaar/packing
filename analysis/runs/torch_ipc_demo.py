# torch_ipc_stdlib_mp.py
import os
import torch
import multiprocessing as mp

def worker(conn, worker_id: int):
    # Receive the LUT tensor (CUDA IPC happens here)
    lut = conn.recv()
    assert lut.is_cuda
    print(f"[worker {worker_id} pid={os.getpid()}] got lut: ptr={lut.data_ptr()} first={int(lut[0].item())}", flush=True)

    import cupy as cp
    gen = cp.random.default_rng(seed=42)
    xx=int(gen.integers(100,2000).get().item())

    # Do some operation that uses LUT. To prove sharing, mutate in-place.
    # In your real code you'd treat LUT as read-only.
    lut[0] += xx
    torch.cuda.synchronize()

    

    conn.send(("done", lut.data_ptr(), int(lut[0].item())))
    conn.close()

def main():
    # IMPORTANT: spawn (CUDA-safe + enables proper IPC behavior)
    mp.set_start_method("spawn", force=True)

    # Create LUT once on GPU in parent
    lut = torch.arange(10, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()
    print(f"[parent pid={os.getpid()}] LUT before: {lut.cpu().tolist()} ptr={lut.data_ptr()}", flush=True)

    # Spawn a couple workers
    n = 3
    procs = []
    conns = []
    for i in range(1, n + 1):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        p = mp.Process(target=worker, args=(child_conn, i))
        p.start()
        procs.append(p)
        conns.append(parent_conn)

    # Send the SAME CUDA tensor object to all workers (shared storage via IPC)
    for c in conns:
        c.send(lut)

    # Collect replies
    for i, (p, c) in enumerate(zip(procs, conns), start=1):
        msg, ptr, first_val = c.recv()
        print(f"[parent] worker {i} says: {msg}, worker_ptr={ptr}, lut[0]={first_val}", flush=True)
        c.close()
        p.join()

    torch.cuda.synchronize()
    print(f"[parent] LUT after:  {lut.cpu().tolist()} ptr={lut.data_ptr()}", flush=True)
    print("[parent] If LUT[0] changed, workers wrote into shared storage.", flush=True)

if __name__ == "__main__":
    main()