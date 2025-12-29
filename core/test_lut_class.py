#!/usr/bin/env python3
"""Test script for new LookupTable class and CollisionCost integration."""

import sys
import numpy as np
import cupy as cp

import kaggle_support as kgs
import pack_cost
import pack_cuda_lut

# Use float32 for consistency with test_lut.ipynb
kgs.USE_FLOAT32 = True
pack_cuda_lut.USE_TEXTURE = True

def test_lookup_table_class():
    """Test LookupTable class creation and properties."""
    print("=" * 60)
    print("Test 1: LookupTable class creation")
    print("=" * 60)

    # Create a small test lookup table manually
    X = np.linspace(-1, 1, 10, dtype=np.float32)
    Y = np.linspace(-1, 1, 10, dtype=np.float32)
    theta = np.linspace(-np.pi, np.pi, 10, dtype=np.float32)
    vals = np.random.rand(10, 10, 10).astype(np.float32)

    lut = pack_cuda_lut.LookupTable(X=X, Y=Y, theta=theta, vals=vals)

    print(f"Created LookupTable:")
    print(f"  Shape: {lut.N_x} x {lut.N_y} x {lut.N_theta}")
    print(f"  X range: [{lut.X_min:.3f}, {lut.X_max:.3f}], dx={lut.grid_dx:.3f}")
    print(f"  Y range: [{lut.Y_min:.3f}, {lut.Y_max:.3f}], dy={lut.grid_dy:.3f}")
    print(f"  Theta range: [{lut.theta_min:.3f}, {lut.theta_max:.3f}], dtheta={lut.grid_dtheta:.3f}")
    print("✓ LookupTable class works!\n")


def test_lut_build_from_cost():
    """Test building LUT from cost function."""
    print("=" * 60)
    print("Test 2: Building LUT from cost function (small grid)")
    print("=" * 60)

    # Create cost function
    cost_fn = pack_cost.CollisionCostOverlappingArea()

    # Build small lookup table
    lut = pack_cuda_lut.LookupTable.build_from_cost_function(
        cost_fn=cost_fn,
        N_x=50,
        N_y=50,
        N_theta=50,
        trim_zeros=True,
        verbose=True
    )

    print(f"\n✓ Built lookup table with shape: {lut.N_x} x {lut.N_y} x {lut.N_theta}")
    print(f"  Cost range: [{lut.vals.min():.6f}, {lut.vals.max():.6f}]\n")

    return lut


def test_collision_cost_with_lut():
    """Test CollisionCost with use_lookup_table option."""
    print("=" * 60)
    print("Test 3: CollisionCost with use_lookup_table=True")
    print("=" * 60)

    # Create test solution
    N_individuals = 100
    N_trees = 20

    np.random.seed(42)
    xyt = np.random.uniform(-1, 1, size=(N_individuals, N_trees, 3)).astype(np.float32)
    xyt[:, :, 2] = np.random.uniform(-np.pi, np.pi, size=(N_individuals, N_trees))
    xyt_cp = cp.asarray(xyt)

    sol = kgs.SolutionCollectionSquare()
    sol.xyt = xyt_cp
    sol.h = cp.tile(cp.array([[10., 0., 0.]], dtype=cp.float32), (N_individuals, 1))
    sol.check_constraints()

    print(f"Created test solution: {N_individuals} ensembles x {N_trees} trees")

    # Test with lookup table (small grid for speed)
    print("\nComputing cost with LUT (50x50x50)...")
    cost_lut = pack_cost.CollisionCostOverlappingArea(
        use_lookup_table=True,
        lut_N_x=10,
        lut_N_y=10,
        lut_N_theta=10
    )

    cost_vals_lut, grad_lut, _ = cost_lut.compute_cost_allocate(sol, evaluate_gradient=True)
    print(f"  Cost range: [{cost_vals_lut.get().min():.6f}, {cost_vals_lut.get().max():.6f}]")
    print(f"  Gradient shape: {grad_lut.shape}")

    # Test without lookup table for comparison
    print("\nComputing cost with standard method...")
    cost_standard = pack_cost.CollisionCostOverlappingArea(use_lookup_table=False)
    cost_vals_std, grad_std, _ = cost_standard.compute_cost_allocate(sol, evaluate_gradient=True)
    print(f"  Cost range: [{cost_vals_std.get().min():.6f}, {cost_vals_std.get().max():.6f}]")

    # Compare
    cost_diff = cp.abs(cost_vals_lut - cost_vals_std).get()
    rel_error = cost_diff / (cp.abs(cost_vals_std).get() + 1e-10)

    print(f"\nComparison:")
    print(f"  Max absolute error: {cost_diff.max():.6e}")
    print(f"  Mean absolute error: {cost_diff.mean():.6e}")
    print(f"  Max relative error: {rel_error[cost_vals_std.get() > 0.01].max():.4f}")
    print(f"  Mean relative error: {rel_error[cost_vals_std.get() > 0.01].mean():.4f}")

    print(cost_vals_lut[:10].get())
    print(grad_lut[:2,:5,:].get())

    raise 'stop'

    print("\n✓ CollisionCost with LUT works!\n")


def test_lut_swapping():
    """Test fast LUT swapping between different cost functions."""
    print("=" * 60)
    print("Test 4: Fast LUT swapping between cost functions")
    print("=" * 60)

    # Create test solution
    N_individuals = 50
    N_trees = 15

    np.random.seed(123)
    xyt = np.random.uniform(-1, 1, size=(N_individuals, N_trees, 3)).astype(np.float32)
    xyt[:, :, 2] = np.random.uniform(-np.pi, np.pi, size=(N_individuals, N_trees))
    xyt_cp = cp.asarray(xyt)

    sol = kgs.SolutionCollectionSquare()
    sol.xyt = xyt_cp
    sol.h = cp.tile(cp.array([[10., 0., 0.]], dtype=cp.float32), (N_individuals, 1))
    sol.check_constraints()

    print(f"Created test solution: {N_individuals} ensembles x {N_trees} trees\n")

    # Create two different cost functions with different LUT grids
    print("Creating CollisionCostOverlappingArea with coarse LUT (30x30x30)...")
    cost_overlap = pack_cost.CollisionCostOverlappingArea(
        use_lookup_table=True,
        lut_N_x=30,
        lut_N_y=30,
        lut_N_theta=30
    )

    print("\nCreating CollisionCostSeparation with fine LUT (60x60x60)...")
    cost_separation = pack_cost.CollisionCostSeparation(
        use_lookup_table=True,
        lut_N_x=60,
        lut_N_y=60,
        lut_N_theta=60
    )

    # Compute with first cost function
    print("\n--- Computing with CollisionCostOverlappingArea ---")
    cost1, _, _ = cost_overlap.compute_cost_allocate(sol, evaluate_gradient=False)
    print(f"  Cost range: [{cost1.get().min():.6f}, {cost1.get().max():.6f}]")

    # Compute with second cost function (demonstrates fast LUT swap)
    print("\n--- Computing with CollisionCostSeparation (fast swap!) ---")
    cost2, _, _ = cost_separation.compute_cost_allocate(sol, evaluate_gradient=False)
    print(f"  Cost range: [{cost2.get().min():.6f}, {cost2.get().max():.6f}]")

    # Swap back to first (demonstrates repeated swapping is fast)
    print("\n--- Computing with CollisionCostOverlappingArea again (fast swap!) ---")
    cost3, _, _ = cost_overlap.compute_cost_allocate(sol, evaluate_gradient=False)
    print(f"  Cost range: [{cost3.get().min():.6f}, {cost3.get().max():.6f}]")

    # Verify consistency
    assert cp.allclose(cost1, cost3), "Costs should be identical on repeated calls"
    print("\n✓ LUT swapping works correctly!")
    print("✓ No kernel recompilation needed - just updated constant memory!\n")


def test_zero_trimming():
    """Test zero edge trimming."""
    print("=" * 60)
    print("Test 4: Zero edge trimming")
    print("=" * 60)

    # Create array with zeros on edges
    X = np.linspace(-2, 2, 100, dtype=np.float32)
    Y = np.linspace(-2, 2, 100, dtype=np.float32)
    theta = np.linspace(-np.pi, np.pi, 100, dtype=np.float32)

    vals = np.zeros((100, 100, 100), dtype=np.float32)
    # Add some non-zero values in the center
    vals[30:70, 30:70, 30:70] = 1.0

    print(f"Original shape: {vals.shape}")

    X_trim, Y_trim, theta_trim, vals_trim = pack_cuda_lut.LookupTable._trim_zero_edges(
        X, Y, theta, vals, verbose=True
    )

    print(f"\n✓ Zero trimming works!")
    print(f"  Trimmed shape: {vals_trim.shape}")
    print(f"  X range: [{X_trim[0]:.3f}, {X_trim[-1]:.3f}]")
    print(f"  Y range: [{Y_trim[0]:.3f}, {Y_trim[-1]:.3f}]\n")


if __name__ == "__main__":
    try:
        test_lookup_table_class()
        test_zero_trimming()
        test_lut_build_from_cost()
        test_collision_cost_with_lut()
        test_lut_swapping()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
