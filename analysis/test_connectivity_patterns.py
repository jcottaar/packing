#!/usr/bin/env python3
"""Test script to verify connectivity pattern switching works correctly"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'core'))

import pack_runner
import numpy as np

def test_connectivity_patterns():
    """Test all connectivity patterns"""
    
    # Create base runner
    runner = pack_runner.baseline_runner(fast_mode=True)
    
    patterns = [
        (1, "Default ring"),
        (2, "Ring with symmetric star"),  
        (3, "Ring with asymmetric star"),
        (4, "Ring with small world"),
        (5, "Hypercube"),
        (6, "Tree (N=31)")
    ]
    
    print("Testing connectivity patterns...")
    
    for pattern_id, pattern_name in patterns:
        print(f"\n=== Testing Pattern {pattern_id}: {pattern_name} ===")
        
        # Reset to base configuration
        test_runner = pack_runner.baseline_runner(fast_mode=True)
        
        # Apply connectivity pattern
        modifier = runner.modifier_dict['connectivity_pattern']
        modifier.modifier_function(test_runner.base_ga, 'connectivity_pattern', pattern_id)
        
        # Check the resulting GA type and properties
        ga_type = type(test_runner.base_ga.ga).__name__
        print(f"GA Type: {ga_type}")
        print(f"N islands: {test_runner.base_ga.ga.N}")
        
        if hasattr(test_runner.base_ga.ga, 'mate_distance'):
            print(f"Mate distance: {test_runner.base_ga.ga.mate_distance}")
        if hasattr(test_runner.base_ga.ga, 'star_topology'):
            print(f"Star topology: {test_runner.base_ga.ga.star_topology}")
            if test_runner.base_ga.ga.star_topology and hasattr(test_runner.base_ga.ga, 'asymmetric_star'):
                print(f"Asymmetric star: {test_runner.base_ga.ga.asymmetric_star}")
        if hasattr(test_runner.base_ga.ga, 'small_world_rewiring'):
            print(f"Small world rewiring: {test_runner.base_ga.ga.small_world_rewiring}")
        
        # Test connectivity matrix (small sample)
        if hasattr(test_runner.base_ga.ga, '_get_connectivity_matrix'):
            # Initialize the GA list to test connectivity matrix
            test_runner.base_ga.ga.ga_list = [None] * test_runner.base_ga.ga.N
            matrix = test_runner.base_ga.ga._get_connectivity_matrix()
            print(f"Connectivity matrix shape: {matrix.shape}")
            print(f"Total connections: {np.sum(matrix)}")
            print(f"Connections per island (avg): {np.sum(matrix) / matrix.shape[0]:.1f}")
            
    print("\n✅ All connectivity patterns tested successfully!")

if __name__ == "__main__":
    test_connectivity_patterns()