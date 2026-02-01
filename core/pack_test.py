"""Test Suite for Packing Optimization Framework

This module provides comprehensive testing for cost functions, gradients, genetic algorithm,
and GPU kernels. Tests validate correctness through finite difference checks, vectorization
consistency, and reference comparisons.

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt it
(including commercially), but must give credit to the original author (Jeroen Cottaar)
and keep it under this license.
"""
import numpy as np
import cupy as cp
import kaggle_support as kgs
import pack_cost
import pack_cuda
import pack_cuda_primitives_test
import pack_ga3
import matplotlib.pyplot as plt
import pack_vis_sol
import copy


# Global test configuration
CUDA_float32 = True

# Initialize framework
kgs.set_float32(CUDA_float32)
pack_cuda._ensure_initialized()



def run_all_tests(regenerate_reference=False):
    """Execute complete test suite for GA, cost functions, and GPU kernels.
    
    Args:
        regenerate_reference: If True, regenerate reference outputs for GA tests.
                            Use with caution - only when algorithm changes are intentional.
    """
    # Enable detailed debugging output
    kgs.debugging_mode = 2
    
    # Test genetic algorithm with different configurations
    test_ga(regenerate_reference, 0)
    test_ga(regenerate_reference, 1)
    
    # Test all cost functions and gradients
    test_costs()
    
    # Test GPU primitives (CUDA kernels)
    pack_cuda_primitives_test.run_all_tests()
    
    print("All tests passed.")
    kgs.set_float32(True)


def test_ga(regenerate_reference, test_case):
    """Test genetic algorithm convergence and reproducibility.
    
    Runs a small-scale GA optimization and compares final fitness values against
    stored reference results to ensure algorithm stability.
    
    Args:
        regenerate_reference: If True, save current results as new reference
        test_case: Which GA configuration to test (0=baseline, 1=symmetric)
    
    Raises:
        AssertionError: If fitness values differ from reference
    """
    # Configure GA based on test case
    match test_case:
        case 0:
            ga = pack_ga3.baseline()
        case 1:
            ga = pack_ga3.baseline_symmetry_180_tesselated()
            # Force axis alignment
            ga.ga.ga_base.initializer.ref_sol_axis2_offset = lambda r: 0.5
    
    # Set test parameters (smaller scale for faster testing)
    ga.n_generations = 5
    ga.ga.ga_base.N_trees_to_do = 20
    ga.rough_relaxers[0].cost.costs[2].lut_N_theta = 50
    ga.ga.ga_base.population_size = 100
    ga.ga.ga_base.search_depth = 0.8
    ga.ga.ga_base.elitism_fraction = 0.5
    ga.ga.ga_base.survival_rate = 0.7
    ga.ga.do_legalize = False
    ga.ga.ga_base.remove_population_after_abbreviate = False
    
    # Run optimization
    ga.run()
    
    # Collect fitness from all GA stages
    res = ga.ga.ga_list[0].population.fitness
    for g in ga.ga.ga_list[1:]:
        res = np.concatenate((res, g.population.fitness))
    
    # Save or compare against reference
    if regenerate_reference:
        kgs.dill_save(
            kgs.code_dir + f'ref_ga_{test_case}.pickle', res
        )
    ref = kgs.dill_load(kgs.code_dir + f'ref_ga_{test_case}.pickle')
    assert np.all(ref == res), (
        "GA test failed: final fitness does not match reference."
    )


def test_costs():
    """Test all cost functions for correctness and gradient accuracy.
    
    Validates:
    - CPU (reference) vs GPU (fast) implementation consistency
    - Gradient correctness via finite differences
    - Vectorized vs individual computation equivalence
    - Multiple solution collection types (Square, Lattice, etc.)
    """
    print('Testing cost computation and gradients')
    
    # Define all cost functions to test
    costs_to_test = [
        pack_cost.CostDummy(),
        pack_cost.AreaCost(),
        pack_cost.CollisionCostSeparation(scaling=5.),
        pack_cost.BoundaryDistanceCost(use_kernel=False),
        pack_cost.BoundaryDistanceCost(use_kernel=True, scaling=5.),
        pack_cost.CollisionCostOverlappingArea(scaling=3.),
        pack_cost.CostCompound(
            scaling=1.5,
            costs=[pack_cost.AreaCost(), pack_cost.CollisionCostSeparation()]
        )
    ]

    # Use high precision for reference computations
    kgs.set_float32(False)
    
    # Create diverse test solutions
    sol_list = []
    
    # 180-degree symmetric square with 3 trees
    sol_list.append(kgs.SolutionCollectionSquareSymmetric180())
    sol_list[-1].xyt = cp.array(
        [place_random(3, 1.5, generator=np.random.default_rng(seed=1)).xyt],
        dtype=cp.float64
    )
    sol_list[-1].h = cp.array([[0.5, 0., 0.]])
    
    # 90-degree symmetric square with 3 trees
    sol_list.append(kgs.SolutionCollectionSquareSymmetric90())
    sol_list[-1].xyt = cp.array(
        [place_random(3, 1.5, generator=np.random.default_rng(seed=1)).xyt],
        dtype=cp.float64
    )
    sol_list[-1].h = cp.array([[0.5, 0., 0.]])
    
    # Standard square with 10 trees (seed 0)
    sol_list.append(kgs.SolutionCollectionSquare())
    sol_list[-1].xyt = cp.array(
        [place_random(10, 1.5, generator=np.random.default_rng(seed=0)).xyt],
        dtype=cp.float64
    )
    sol_list[-1].h = cp.array([[0.5, 0.2, 0.4]])
    sol_list[-1].override_phenotype = True
    
    # Standard square with 10 trees (seed 2, different boundary)
    sol_list.append(kgs.SolutionCollectionSquare())
    sol_list[-1].xyt = cp.array(
        [place_random(10, 1.5, generator=np.random.default_rng(seed=2)).xyt],
        dtype=cp.float64
    )
    sol_list[-1].h = cp.array([[2., -0.1, -0.15]])
    sol_list[-1].override_phenotype = True
    
    # Lattice configuration with 2 trees
    sol_list.append(kgs.SolutionCollectionLattice())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],       # Tree 0 at origin
        [1.5, 0.5, np.pi / 4]  # Tree 1 offset and rotated
    ]]) / 4
    a_length = 2.5 / 3
    b_length = 2.5 / 2
    angle = np.pi / 3  # 60 degrees
    sol_list[-1].h = cp.array([[a_length, b_length, angle]])
    
    # Lattice with negative dimensions
    sol_list.append(kgs.SolutionCollectionLattice())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, np.pi / 4]
    ]]) / 4
    a_length = -2.5 / 6
    b_length = 2.5 / 6
    angle = np.pi / 3
    sol_list[-1].h = cp.array([[a_length, b_length, angle]])
    
    # Rectangular lattice
    sol_list.append(kgs.SolutionCollectionLatticeRectangle())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, np.pi / 4]
    ]]) / 4
    a_length = -2.5 / 6
    b_length = 2.5 / 5
    sol_list[-1].h = cp.array([[a_length, b_length]])
    
    # Fixed aspect ratio lattice
    sol_list.append(kgs.SolutionCollectionLatticeFixed())
    sol_list[-1].xyt = cp.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.5, np.pi / 4]
    ]]) / 4
    sol_list[-1].aspect_ratios = cp.array([2.])
    sol_list[-1].h = cp.array([[0.3]])
    sol_list[-1].N_periodic = 1
    
    # Visualize all test solutions
    for ii in range(len(sol_list)):
        if sol_list[ii].is_phenotype():
            pack_vis_sol.pack_vis_sol(sol_list[ii], solution_idx=0)
        else:
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            pack_vis_sol.pack_vis_sol(
                sol_list[ii], ax=ax[0], solution_idx=0
            )
            pack_vis_sol.pack_vis_sol(
                sol_list[ii].convert_to_phenotype(), ax=ax[1], solution_idx=0
            )
            sol_list[ii].canonicalize()
            _, ax = plt.subplots(1, 2, figsize=(10, 5))
            pack_vis_sol.pack_vis_sol(
                sol_list[ii], ax=ax[0], solution_idx=0
            )
            pack_vis_sol.pack_vis_sol(
                sol_list[ii].convert_to_phenotype(), ax=ax[1], solution_idx=0
            )
    plt.pause(0.001)

    # Test each cost function
    for c in costs_to_test:
        kgs.set_float32(False)
        
        # Storage for vectorized validation
        all_ref_outputs = []
        all_fast_outputs = []
        all_xyt = []
        all_bounds = []

        print(f"\nTesting {c.__class__.__name__}")

        for sol_single in sol_list:
            # Skip boundary cost for periodic solutions
            if (isinstance(c, pack_cost.BoundaryDistanceCost) and
                    sol_single.periodic):
                continue
            
            # Store for vectorized check
            all_xyt.append(sol_single.xyt)
            all_bounds.append(sol_single.h)
            
            # Compare reference (CPU) vs fast (GPU) implementations
            cost_ref, grad_ref, grad_bound_ref = c.compute_cost_ref(sol_single)
            sol_fast = copy.deepcopy(sol_single)
            kgs.set_float32(CUDA_float32)
            sol_fast.xyt = cp.array(sol_single.xyt, dtype=kgs.dtype_cp)
            sol_fast.h = cp.array(sol_single.h, dtype=kgs.dtype_cp)
            cost_fast, grad_fast, grad_bound_fast = c.compute_cost_allocate(
                sol_fast
            )
            cost_fast_no_grad = c.compute_cost_allocate(
                sol_fast, evaluate_gradient=False
            )[0]
            assert cp.all(cost_fast_no_grad == cost_fast)
            
            # Store all outputs for vectorized testing
            all_ref_outputs.append((cost_ref, grad_ref, grad_bound_ref))
            all_fast_outputs.append((cost_fast, grad_fast, grad_bound_fast))
            
            # Verify cost values match
            print(cp.array2string(
                cp.asarray(cost_fast), precision=17, suppress_small=False
            ))
            print(cp.array2string(
                cp.asarray(cost_ref), precision=17, suppress_small=False
            ))
            if not isinstance(c, pack_cost.CostDummy):
                assert cost_ref > 0
            assert cp.allclose(cost_ref, cost_fast, rtol=1e-5), (
                f"Cost mismatch: {cost_ref} vs {cost_fast}"
            )
            
            # Verify gradient values match (skip for float32 periodic)
            if not CUDA_float32 or not sol_single.periodic:
                assert cp.allclose(
                    grad_ref, grad_fast, rtol=1e-4, atol=1e-4
                ), f"Gradient mismatch: {grad_ref} vs {grad_fast}"
                assert cp.allclose(
                    grad_bound_ref, grad_bound_fast, rtol=1e-4, atol=1e-4
                ), f"Bound gradient mismatch: {grad_bound_ref} vs {grad_bound_fast}"

            kgs.set_float32(False)

            # Validate gradients using finite differences
            def _get_cost(obj, xyt_arr):
                """Helper to compute cost for given tree positions."""
                sol_tmp = copy.deepcopy(sol_single)
                sol_tmp.xyt = cp.array(xyt_arr)
                sol_tmp.h = sol_single.h
                if CUDA_float32:
                    return obj.compute_cost_ref(sol_tmp)[0]
                else:
                    return obj.compute_cost_allocate(sol_tmp)[0]

            # Compute numerical gradient w.r.t. tree positions
            x0 = sol_single.xyt.copy()
            shape = x0.shape
            x_flat = x0.ravel()
            n = x_flat.size
            eps = 1e-6  # Finite difference step size
            grad_num = cp.zeros(n, dtype=float)

            for i in range(n):
                x_plus = x_flat.copy()
                x_minus = x_flat.copy()
                x_plus[i] += eps
                x_minus[i] -= eps

                c_plus = _get_cost(c, x_plus.reshape(shape))
                c_minus = _get_cost(c, x_minus.reshape(shape))
                grad_num[i] = (c_plus - c_minus) / (2.0 * eps)

            grad_fast_flat = cp.asarray(grad_fast).ravel()
            max_diff = cp.max(cp.abs(grad_num - grad_fast_flat)).get().item()
            assert cp.allclose(
                grad_num, grad_ref.ravel(), rtol=1e-4, atol=1e-4
            ), f"Finite-diff gradient mismatch (max diff {max_diff})"

            # Validate bound gradients using finite differences
            def _get_cost_bound(obj, bound_arr):
                """Helper to compute cost for given boundary parameters."""
                sol_tmp = copy.deepcopy(sol_single)
                sol_tmp.xyt = sol_single.xyt
                sol_tmp.h = cp.array(bound_arr)
                if CUDA_float32:
                    return obj.compute_cost_ref(sol_tmp)[0]
                else:
                    return obj.compute_cost_allocate(sol_tmp)[0]

            # Compute numerical gradient w.r.t. boundary parameters
            b0 = sol_single.h.copy()
            bound_shape = b0.shape
            bound_flat = b0.ravel()
            n_bound = bound_flat.size
            grad_bound_num = cp.zeros(n_bound, dtype=float)

            for i in range(n_bound):
                b_plus = bound_flat.copy()
                b_minus = bound_flat.copy()
                b_plus[i] += eps
                b_minus[i] -= eps

                c_plus = _get_cost_bound(c, b_plus.reshape(bound_shape))
                c_minus = _get_cost_bound(c, b_minus.reshape(bound_shape))
                grad_bound_num[i] = (c_plus - c_minus) / (2.0 * eps)

            grad_bound_fast_flat = cp.asarray(grad_bound_fast).ravel()
            max_diff_bound = cp.max(
                cp.abs(grad_bound_num - grad_bound_fast_flat)
            ).get().item()
            assert cp.allclose(
                grad_bound_num, grad_bound_ref.ravel(), rtol=1e-4, atol=1e-4
            ), f"Finite-diff bound gradient mismatch (max diff {max_diff_bound})"
        
        # Test vectorized computation consistency
        for todo in (
            [[2, 3]] if (isinstance(c, pack_cost.BoundaryDistanceCost))
            else [[2, 3], [4, 5]]
        ):
            print(f"  Vectorized check for {c.__class__.__name__}")
            
            # Concatenate multiple solutions for batch processing
            full_xyt = cp.concatenate(all_xyt[todo[0]:todo[-1] + 1], axis=0)
            full_bounds = cp.concatenate(
                all_bounds[todo[0]:todo[-1] + 1], axis=0
            )

            # Compute vectorized results
            full_sol = copy.deepcopy(sol_list[todo[0]])
            full_sol.xyt = full_xyt
            full_sol.h = full_bounds
            kgs.set_float32(False)
            print(full_sol.h)
            vec_cost_ref, vec_grad_ref, vec_grad_bound_ref = (
                c.compute_cost_ref(full_sol)
            )
            print(vec_cost_ref)
            
            kgs.set_float32(CUDA_float32)
            full_sol_fast = copy.deepcopy(sol_list[todo[0]])
            full_sol_fast.xyt = cp.array(full_xyt, dtype=kgs.dtype_cp)
            full_sol_fast.h = cp.array(full_bounds, dtype=kgs.dtype_cp)
            vec_cost_fast, vec_grad_fast, vec_grad_bound_fast = (
                c.compute_cost_allocate(full_sol_fast)
            )
            
            # Verify vectorized matches individual calls exactly
            for i, i2 in enumerate(todo):
                stored_ref = all_ref_outputs[i2]
                stored_fast = all_fast_outputs[i2]
                
                # Extract from vectorized results
                vec_ref_cost_i = vec_cost_ref[i]
                vec_fast_cost_i = vec_cost_fast[i]
                vec_ref_grad_i = vec_grad_ref[i]
                vec_fast_grad_i = vec_grad_fast[i]
                vec_ref_grad_bound_i = vec_grad_bound_ref[i]
                vec_fast_grad_bound_i = vec_grad_bound_fast[i]
                
                # Compare - must be exactly identical
                assert cp.array_equal(vec_ref_cost_i, stored_ref[0][0]), (
                    f"Vectorized ref cost mismatch for {c.__class__.__name__} "
                    f"tree {i}: {vec_ref_cost_i} vs {stored_ref[0]}"
                )
                assert cp.array_equal(vec_fast_cost_i, stored_fast[0][0]), (
                    f"Vectorized fast cost mismatch for {c.__class__.__name__} "
                    f"tree {i}: {vec_fast_cost_i} vs {stored_fast[0]}"
                )
                assert cp.array_equal(vec_ref_grad_i, stored_ref[1][0]), (
                    f"Vectorized ref grad mismatch for {c.__class__.__name__} "
                    f"tree {i}"
                )
                assert cp.array_equal(vec_fast_grad_i, stored_fast[1][0]), (
                    f"Vectorized fast grad mismatch for {c.__class__.__name__} "
                    f"tree {i}"
                )
                assert cp.array_equal(
                    vec_ref_grad_bound_i, stored_ref[2][0]
                ), (
                    f"Vectorized ref bound grad mismatch for "
                    f"{c.__class__.__name__} tree {i}"
                )
                assert cp.array_equal(
                    vec_fast_grad_bound_i, stored_fast[2][0]
                ), (
                    f"Vectorized fast bound grad mismatch for "
                    f"{c.__class__.__name__} tree {i}"
                )
            
            print("  ✓ Vectorized results exactly match individual calls")


def place_random(N, inner_size, generator=None):
    """Generate random tree placements within a square region.
    
    Args:
        N: Number of trees to place (int)
        inner_size: Side length of square region (float)
        generator: NumPy random generator (default: seed 42)
    
    Returns:
        TreeList with N trees randomly positioned in [-inner_size/2, inner_size/2]
        squared region with random orientations [0, 2π]
    """
    if generator is None:
        generator = np.random.default_rng(seed=42)
    
    tree_list = kgs.TreeList()
    
    # Random positions centered at origin
    tree_list.x = generator.random(N) * inner_size - inner_size / 2
    tree_list.y = generator.random(N) * inner_size - inner_size / 2
    
    # Random orientations
    tree_list.theta = generator.random(N) * 2 * np.pi
    
    return tree_list