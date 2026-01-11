"""Test script for visualizing analytical edge_spacer sampling in SolutionCollectionSquare."""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import kaggle_support as kgs

def test_edge_spacer_sampling():
    # Create edge spacer with specific parameters
    edge_spacer = kgs.EdgeSpacerBasic(dist_x=0.5, dist_y=0.5, dist_corner=0.0)
    
    # Create solution collection with edge spacer filtering enabled
    sol = kgs.SolutionCollectionSquare(
        edge_spacer=edge_spacer,
        filter_move_locations_with_edge_spacer=True,
        filter_move_locations_margin=0.25
    )
    
    # Set up 2 individuals with different h values (size, x_offset, y_offset)
    sol.h = cp.array([
        [4.0, 0.0, 0.0],   # Individual 0: size=4, centered
        [6.0, 0.5, -0.5],  # Individual 1: size=6, offset
    ], dtype=kgs.dtype_cp)
    
    # We need some dummy xyt to make the solution valid
    sol.xyt = cp.zeros((2, 1, 3), dtype=kgs.dtype_cp)
    
    # Create generator with fixed seed for reproducibility
    generator = cp.random.Generator(cp.random.XORWOW(seed=42))
    
    # Sample move locations for each individual
    N_samples = 10000
    
    # We'll call _generate_move_centers N_samples times with 2 indices
    # Actually, let's call it once per sample to accumulate
    all_x = [[], []]
    all_y = [[], []]
    
    for _ in range(N_samples):
        inds = cp.arange(2)
        x, y = sol._generate_move_centers(edge_clearance=None, inds_to_do=inds, generator=generator)
        all_x[0].append(float(x[0]))
        all_x[1].append(float(x[1]))
        all_y[0].append(float(y[0]))
        all_y[1].append(float(y[1]))
    
    all_x = [np.array(ax) for ax in all_x]
    all_y = [np.array(ay) for ay in all_y]
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, ax in enumerate(axes):
        h = sol.h[i].get()
        size, offset_x, offset_y = h[0], h[1], h[2]
        half_size = size / 2
        margin = sol.filter_move_locations_margin
        x_threshold = half_size - edge_spacer.dist_x - margin
        y_threshold = half_size - edge_spacer.dist_y - margin
        
        # Plot sampled points
        ax.scatter(all_x[i], all_y[i], s=1, alpha=0.3, label='Sampled points')
        
        # Draw the square boundary
        rect_x = [-half_size + offset_x, half_size + offset_x, half_size + offset_x, -half_size + offset_x, -half_size + offset_x]
        rect_y = [-half_size + offset_y, -half_size + offset_y, half_size + offset_y, half_size + offset_y, -half_size + offset_y]
        ax.plot(rect_x, rect_y, 'b-', linewidth=2, label='Boundary')
        
        # Draw the invalid region (center rectangle where points should NOT be)
        invalid_x = [-x_threshold + offset_x, x_threshold + offset_x, x_threshold + offset_x, -x_threshold + offset_x, -x_threshold + offset_x]
        invalid_y = [-y_threshold + offset_y, -y_threshold + offset_y, y_threshold + offset_y, y_threshold + offset_y, -y_threshold + offset_y]
        ax.fill(invalid_x, invalid_y, color='red', alpha=0.2, label='Invalid region')
        ax.plot(invalid_x, invalid_y, 'r--', linewidth=1)
        
        ax.set_aspect('equal')
        ax.set_title(f'Individual {i}: size={size}, offset=({offset_x}, {offset_y})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('edge_spacer_sampling_test.png', dpi=150)
    plt.show()
    print("Saved plot to edge_spacer_sampling_test.png")

if __name__ == '__main__':
    test_edge_spacer_sampling()
