#!/usr/bin/env python3
"""
PyTorch IPC with GPU: Parallel Computation Example
Demonstrates shared CPU memory with parallel GPU processing across multiple processes
"""

import torch
import torch.multiprocessing as mp


def gpu_worker(shared_tensor, start_idx, end_idx, gpu_id):
    """Worker process that processes a slice of the tensor on GPU"""
    device = torch.device(f'cuda:{gpu_id % torch.cuda.device_count()}')
    print(f"Worker {gpu_id}: Processing indices [{start_idx}:{end_idx}] on {device}")
    
    # Copy slice to GPU, compute, copy results back to shared CPU tensor
    slice_data = shared_tensor[start_idx:end_idx].to(device)
    result = slice_data ** 2 + slice_data * 3  # Some computation
    shared_tensor[start_idx:end_idx] = result.cpu()
    
    print(f"Worker {gpu_id}: Completed")


if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    print("PyTorch GPU Multiprocessing Demo")
    print("="*60)
    
    # Create shared tensor in CPU memory
    size = 1000
    shared_tensor = torch.arange(size, dtype=torch.float32)
    shared_tensor.share_memory_()  # Enable sharing across processes
    
    print(f"Initial tensor sum: {shared_tensor.sum().item():.0f}")
    print(f"Initial first 10: {shared_tensor[:10].tolist()}")
    
    # Split work among processes
    num_workers = 4
    chunk_size = size // num_workers
    processes = []
    
    print(f"\nLaunching {num_workers} GPU workers...")
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else size
        p = mp.Process(target=gpu_worker, args=(shared_tensor, start, end, i))
        p.start()
        processes.append(p)
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    print(f"\nFinal tensor sum: {shared_tensor.sum().item():.0f}")
    print(f"Final first 10: {shared_tensor[:10].tolist()}")
    print("="*60)
    print("Completed!")
