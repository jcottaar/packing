#!/usr/bin/env python3
"""
PyTorch IPC: Alternative approaches to sharing tensors
Demonstrates different ways to share tensors without passing as function arguments
"""

import torch
import torch.multiprocessing as mp

# =============================================================================
# Approach 1: Module-level variable (doesn't work with spawn)
# =============================================================================

SHARED_TENSOR = None  # This won't be visible to child processes with 'spawn'

def worker_with_global(start_idx, end_idx, gpu_id):
    """This will fail with spawn because SHARED_TENSOR is None in child process"""
    print(f"Worker {gpu_id}: SHARED_TENSOR is {SHARED_TENSOR}")
    if SHARED_TENSOR is None:
        print(f"Worker {gpu_id}: ERROR - SHARED_TENSOR is None!")
        return


# =============================================================================
# Approach 2: Using multiprocessing.Manager (works but slower)
# =============================================================================

def worker_with_dict(shared_dict, start_idx, end_idx, gpu_id):
    """Worker accesses tensor via managed dict"""
    device = torch.device(f'cuda:{gpu_id % torch.cuda.device_count()}')
    print(f"Worker {gpu_id}: Processing indices [{start_idx}:{end_idx}] on {device}")
    
    # Get tensor from shared dict
    shared_tensor = shared_dict['tensor']
    
    # Process on GPU
    slice_data = shared_tensor[start_idx:end_idx].to(device)
    result = slice_data ** 2 + slice_data * 3
    shared_tensor[start_idx:end_idx] = result.cpu()
    
    # Update dict (not strictly necessary since tensor is already shared)
    shared_dict['tensor'] = shared_tensor
    
    print(f"Worker {gpu_id}: Completed")


# =============================================================================
# Approach 3: Using Queue to send reference (works)
# =============================================================================

def worker_with_queue(queue, start_idx, end_idx, gpu_id):
    """Worker receives tensor reference via queue"""
    device = torch.device(f'cuda:{gpu_id % torch.cuda.device_count()}')
    
    # Get tensor from queue
    shared_tensor = queue.get()
    print(f"Worker {gpu_id}: Received tensor, processing [{start_idx}:{end_idx}] on {device}")
    
    slice_data = shared_tensor[start_idx:end_idx].to(device)
    result = slice_data ** 2 + slice_data * 3
    shared_tensor[start_idx:end_idx] = result.cpu()
    
    print(f"Worker {gpu_id}: Completed")


def demo_approach1():
    """Demo: Global variable - FAILS with spawn"""
    global SHARED_TENSOR
    print("\nApproach 1: Module-level variable (spawn method)")
    print("="*60)
    
    SHARED_TENSOR = torch.arange(100, dtype=torch.float32)
    SHARED_TENSOR.share_memory_()
    
    print("Main process: SHARED_TENSOR created")
    
    p = mp.Process(target=worker_with_global, args=(0, 100, 0))
    p.start()
    p.join()
    
    print("Result: Global variable approach doesn't work with spawn!")


def demo_approach2():
    """Demo: Manager dict"""
    print("\nApproach 2: Manager dict")
    print("="*60)
    
    # Create manager and shared dict
    manager = mp.Manager()
    shared_dict = manager.dict()
    
    # Create and share tensor via dict
    size = 1000
    shared_tensor = torch.arange(size, dtype=torch.float32)
    shared_tensor.share_memory_()
    shared_dict['tensor'] = shared_tensor
    
    print(f"Initial tensor sum: {shared_tensor.sum().item():.0f}")
    
    # Launch workers
    num_workers = 4
    chunk_size = size // num_workers
    processes = []
    
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else size
        p = mp.Process(target=worker_with_dict, args=(shared_dict, start, end, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    final_tensor = shared_dict['tensor']
    print(f"Final tensor sum: {final_tensor.sum().item():.0f}")
    print("Result: Works, but Manager adds overhead")


def demo_approach3():
    """Demo: Queue to distribute tensor reference"""
    print("\nApproach 3: Queue to send tensor reference")
    print("="*60)
    
    size = 1000
    shared_tensor = torch.arange(size, dtype=torch.float32)
    shared_tensor.share_memory_()
    
    print(f"Initial tensor sum: {shared_tensor.sum().item():.0f}")
    
    # Create queue and send tensor to all workers
    queue = mp.Queue()
    
    # Launch workers
    num_workers = 4
    chunk_size = size // num_workers
    processes = []
    
    for i in range(num_workers):
        # Put tensor in queue for each worker
        queue.put(shared_tensor)
        
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else size
        p = mp.Process(target=worker_with_queue, args=(queue, start, end, i))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print(f"Final tensor sum: {shared_tensor.sum().item():.0f}")
    print("Result: Works! Each worker gets tensor via queue")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print("PyTorch IPC: Alternative Sharing Approaches")
    print("="*60)
    
    demo_approach1()  # Shows why globals don't work
    demo_approach2()  # Manager dict - works but overhead
    demo_approach3()  # Queue - works well
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Passing as argument: Most efficient (direct pickling)")
    print("- Queue approach: Works, minimal overhead")
    print("- Manager dict: Works, but adds synchronization overhead")
    print("- Global variable: Doesn't work with spawn method")
    print("="*60)
