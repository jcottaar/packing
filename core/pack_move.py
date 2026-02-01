"""
Genetic Algorithm Mutation Operators

This module defines mutation moves for the genetic algorithm, including
random perturbations, translations, rotations, and cluster-based operations.
All moves are GPU-accelerated using CuPy for efficient vectorized operations.

This code is released under CC BY-SA 4.0, meaning you can freely use and adapt 
it (including commercially), but must give credit to the original author 
(Jeroen Cottaar) and keep it under this license.
"""
import numpy as np
import cupy as cp
import typing
from typing import TYPE_CHECKING
import kaggle_support as kgs
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import pack_ga3

@dataclass
class Move(kgs.BaseClass):
    """
    Base class for genetic algorithm mutation operators.
    
    All moves operate on GPU arrays (CuPy) for efficient batch processing.
    Subclasses implement _do_move_vec to mutate multiple individuals in parallel.
    """
    respect_edge_spacer_filter: bool = field(init=True, default=True)  # Whether to respect edge spacer constraints

    def do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                    mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                    generator: cp.random.Generator):
        """
        Vectorized move interface - applies mutation to multiple individuals.

        Parameters
        ----------
        population : Population
            Target population where clones are already in place
        inds_to_do : cp.ndarray
            Indices of individuals to modify, shape: (N_moves,)
        mate_sol : kgs.SolutionCollection
            Source solution collection to read from
        inds_mate : cp.ndarray
            Indices of mate individuals in mate_sol, shape: (N_moves,)
        generator : cp.random.Generator
            Random number generator (GPU-based)
        """
        old_val = population.genotype.filter_move_locations_with_edge_spacer
        if not self.respect_edge_spacer_filter:
            population.genotype.filter_move_locations_with_edge_spacer = False        
        self._do_move_vec(population, inds_to_do, mate_sol, inds_mate, generator)
        population.genotype.filter_move_locations_with_edge_spacer = old_val


def _sample_trees_to_mutate(population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                             generator: cp.random.Generator) -> cp.ndarray:
    """
    Sample random tree indices for mutation, respecting edge spacer filtering.
    
    When edge spacer filtering is enabled, only samples from trees that are
    sufficiently close to boundaries. 
        
    Parameters
    ----------
    population : Population
        Population containing solutions and edge spacer constraints
    inds_to_do : cp.ndarray
        Indices of solutions to sample from, shape: (N_moves,)
    generator : cp.random.Generator
        GPU random number generator
        
    Returns
    -------
    cp.ndarray
        Sampled tree indices, shape: (N_moves,)
    """
    xyt = population.genotype.xyt
    N_trees = xyt.shape[1]
    N_moves = int(inds_to_do.shape[0])

    # No filtering - sample uniformly
    if not population.genotype.filter_move_locations_with_edge_spacer:
        return generator.integers(0, N_trees, size=N_moves)

    # Check which trees are valid for each solution
    h_subset = population.genotype.h[inds_to_do]
    valid_mask = population.genotype.edge_spacer.check_valid(
        xyt[inds_to_do], h_subset,
        margin=population.genotype.filter_move_locations_margin
    )  # Shape: (N_moves, N_trees)
    
    # Fallback: for solutions with no valid trees, make all trees valid
    valid_counts = cp.sum(valid_mask, axis=1)  # (N_moves,)
    no_valid_mask = valid_counts == 0
    valid_mask[no_valid_mask, :] = True
    valid_counts = cp.sum(valid_mask, axis=1)
    
    # Generate random indices within the valid tree count for each solution
    random_indices = generator.integers(0, 1, size=N_moves)
    random_indices = (generator.uniform(0, 1, size=N_moves) * 
                      valid_counts).astype(cp.int32)
    
    # Find the tree at the random position within valid trees
    # Use cumulative sum to convert random index to actual tree ID
    cumsum = cp.cumsum(valid_mask, axis=1)  # (N_moves, N_trees)
    target_positions = random_indices[:, cp.newaxis] + 1  # (N_moves, 1)
    
    # Find first tree where cumsum >= target_position
    matches = cumsum >= target_positions  # (N_moves, N_trees)
    tree_indices = cp.argmax(matches, axis=1)  # (N_moves,)
    
    return tree_indices


@dataclass
class MoveSelector(Move):
    """
    Meta-move that randomly selects from multiple move types.
    
    Groups individuals by selected move type and executes moves in batches
    for improved GPU efficiency.
    """
    moves: list = field(init=True, default_factory=list)  # Each element: [Move, name, weight]
    _probabilities: cp.ndarray = field(init=False, default=None)  # Cached normalized probabilities

    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        # Compute normalized probabilities on first call
        if self._probabilities is None:
            total_weight = sum([m[2] for m in self.moves])
            self._probabilities = cp.array(
                [m[2] / total_weight for m in self.moves], dtype=kgs.dtype_cp
            )

        # Select which move to use for each individual
        N_moves = int(inds_mate.shape[0])
        cumulative_probs = cp.cumsum(self._probabilities)
        random_vals = generator.uniform(0, 1, size=N_moves)
        chosen_move_ids_gpu = cp.searchsorted(cumulative_probs, random_vals)

        # Group individuals by chosen move type and execute
        for move_id in range(len(self.moves)):
            mask = chosen_move_ids_gpu == move_id
            if not cp.any(mask):
                continue

            # Get indices for this move type
            batch_inds_to_do = inds_to_do[mask]
            batch_inds_mate = inds_mate[mask]

            # Execute move on batch
            self.moves[move_id][0].do_move_vec(
                population, batch_inds_to_do, mate_sol, batch_inds_mate, generator
            )


@dataclass
class MoveRandomTree(Move):
    """
    Randomly reposition a single tree to a new location.
    
    Selects a random tree and moves it to a random position with random rotation.
    """
    
    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        N_moves = int(inds_to_do.shape[0])

        # Generate random parameters
        trees_to_mutate_gpu = _sample_trees_to_mutate(
            population, inds_to_do, generator
        )
        new_x_gpu, new_y_gpu = population.genotype.generate_move_centers(
            None, inds_to_do, generator
        )
        new_theta_gpu = generator.uniform(-cp.pi, cp.pi, size=N_moves)

        # Apply updates using fancy indexing
        new_xyt = population.genotype.xyt
        new_xyt[inds_to_do, trees_to_mutate_gpu, 0] = new_x_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 1] = new_y_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 2] = new_theta_gpu  


@dataclass
class JiggleRandomTree(Move):
    """
    Perturb a single tree's position and rotation by small random amounts.
    
    Selects a random tree and adds small offsets to its x, y, and theta values.
    """
    max_xy_move: float = field(init=True, default=0.1)  # Maximum position perturbation
    max_theta_move: float = field(init=True, default=np.pi)  # Maximum rotation perturbation
    
    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_xyt = population.genotype.xyt
        N_moves = int(inds_to_do.shape[0])

        # Generate random perturbations
        trees_to_mutate_gpu = _sample_trees_to_mutate(
            population, inds_to_do, generator
        )
        offset_x_gpu = generator.uniform(
            -self.max_xy_move, self.max_xy_move, size=N_moves
        )
        offset_y_gpu = generator.uniform(
            -self.max_xy_move, self.max_xy_move, size=N_moves
        )
        offset_theta_gpu = generator.uniform(
            -self.max_theta_move, self.max_theta_move, size=N_moves
        )

        # Apply updates using fancy indexing
        new_xyt[inds_to_do, trees_to_mutate_gpu, 0] += offset_x_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 1] += offset_y_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 2] += offset_theta_gpu   


@dataclass
class JiggleCluster(Move):
    """
    Perturb a cluster of nearby trees around a random center point.
    
    Selects a random center, finds the N closest trees, and perturbs them
    by small random amounts. The number of trees varies per invocation.
    """
    max_xy_move: float = field(init=True, default=0.1)  # Maximum position perturbation
    max_theta_move: float = field(init=True, default=np.pi)  # Maximum rotation perturbation
    min_N_trees: int = field(init=True, default=2)  # Minimum cluster size
    max_N_trees: int = field(init=True, default=8)  # Maximum cluster size
    
    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])

        # Generate random centers, shape: (N_moves, 1)
        center_x_all, center_y_all = population.genotype.generate_move_centers(
            None, inds_to_do, generator
        )
        center_x_all, center_y_all = center_x_all[:, None], center_y_all[:, None]

        # Generate cluster sizes for all individuals
        max_jiggle = min(self.max_N_trees, N_trees)
        min_jiggle = min(self.min_N_trees, N_trees)
        n_trees_to_jiggle_all = generator.integers(
            min_jiggle, max_jiggle + 1, size=N_moves
        )  # (N_moves,)

        # Pre-generate all random offsets, shape: (N_moves, max_jiggle)
        total_offsets_needed = N_moves * max_jiggle
        all_offset_x = generator.uniform(
            -self.max_xy_move, self.max_xy_move, size=total_offsets_needed
        ).reshape(N_moves, max_jiggle)
        all_offset_y = generator.uniform(
            -self.max_xy_move, self.max_xy_move, size=total_offsets_needed
        ).reshape(N_moves, max_jiggle)
        all_offset_theta = generator.uniform(
            -self.max_theta_move, self.max_theta_move, size=total_offsets_needed
        ).reshape(N_moves, max_jiggle)

        # Get tree positions, shape: (N_moves, N_trees, 2)
        tree_positions = new_xyt[inds_to_do, :, :2]

        # Compute squared distances to centers, shape: (N_moves, N_trees)
        dx = tree_positions[:, :, 0] - center_x_all
        dy = tree_positions[:, :, 1] - center_y_all
        distances = dx**2 + dy**2

        # Sort trees by distance and take closest max_jiggle trees
        sorted_tree_indices = cp.argsort(distances, axis=1)[:, :max_jiggle]

        # Create mask for which trees to actually jiggle
        tree_indices = cp.arange(max_jiggle)[cp.newaxis, :]  # (1, max_jiggle)
        mask = tree_indices < n_trees_to_jiggle_all[:, cp.newaxis]

        # Apply mask to offsets
        all_offset_x = all_offset_x * mask
        all_offset_y = all_offset_y * mask
        all_offset_theta = all_offset_theta * mask

        # Create index arrays for fancy indexing
        individual_indices = inds_to_do[:, cp.newaxis]  # (N_moves, 1)

        # Apply offsets to the closest trees
        new_xyt[individual_indices, sorted_tree_indices, 0] += all_offset_x
        new_xyt[individual_indices, sorted_tree_indices, 1] += all_offset_y
        new_xyt[individual_indices, sorted_tree_indices, 2] += all_offset_theta


@dataclass
class Translate(Move):
    """
    Translate all trees by a random offset with periodic boundary wrapping.
    
    Applies the same random translation to all trees in each solution,
    wrapping coordinates at the boundary using modulo arithmetic.
    """
    
    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt

        # Get boundary sizes, shape: (N_moves, 3)
        h_params = new_h[inds_to_do]

        # Generate random offsets
        h_sizes = h_params[:, 0]
        offset_x_gpu = generator.uniform(
            -h_sizes / 2, h_sizes / 2
        )[:, cp.newaxis]  # (N_moves, 1)
        offset_y_gpu = generator.uniform(
            -h_sizes / 2, h_sizes / 2
        )[:, cp.newaxis]  # (N_moves, 1)
        h_sizes_gpu = h_params[:, 0:1]  # (N_moves, 1)

        # Apply translation with periodic wrapping
        new_xyt[inds_to_do, :, 0] = (cp.mod(
            new_xyt[inds_to_do, :, 0] + offset_x_gpu, h_sizes_gpu
        ) - h_sizes_gpu / 2)
        new_xyt[inds_to_do, :, 1] = (cp.mod(
            new_xyt[inds_to_do, :, 1] + offset_y_gpu, h_sizes_gpu
        ) - h_sizes_gpu / 2)
    

@dataclass
class Twist(Move):
    """
    Rotate trees around a random center with angle decreasing by distance.
    
    Trees closer to the center rotate more, with rotation angle decreasing
    linearly to zero at the specified radius.
    """
    min_radius: float = field(init=True, default=0.5)  # Minimum twist radius
    max_radius: float = field(init=True, default=2.0)  # Maximum twist radius
    
    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_xyt = population.genotype.xyt
        N_moves = int(inds_to_do.shape[0])

        # Generate random twist parameters
        center_x_gpu, center_y_gpu = population.genotype.generate_move_centers(
            None, inds_to_do, generator
        )
        center_x_gpu = center_x_gpu[:, cp.newaxis]  # (N_moves, 1)
        center_y_gpu = center_y_gpu[:, cp.newaxis]  # (N_moves, 1)
        max_twist_angle_gpu = generator.uniform(
            -cp.pi, cp.pi, size=N_moves
        )[:, cp.newaxis]  # (N_moves, 1)
        radius_gpu = generator.uniform(
            self.min_radius, self.max_radius, size=N_moves
        )[:, cp.newaxis]  # (N_moves, 1)

        # Get tree positions, shape: (N_moves, N_trees)
        tree_x = new_xyt[inds_to_do, :, 0]
        tree_y = new_xyt[inds_to_do, :, 1]

        # Compute distances from center
        dx = tree_x - center_x_gpu
        dy = tree_y - center_y_gpu
        distances = cp.sqrt(dx**2 + dy**2)

        # Compute twist angles (linearly decreasing with distance)
        twist_angles = max_twist_angle_gpu * cp.maximum(
            0, 1 - distances / radius_gpu
        )

        # Apply rotation around center point
        cos_angles = cp.cos(twist_angles)
        sin_angles = cp.sin(twist_angles)
        new_x = center_x_gpu + dx * cos_angles - dy * sin_angles
        new_y = center_y_gpu + dx * sin_angles + dy * cos_angles

        # Update positions and angles
        new_xyt[inds_to_do, :, 0] = new_x
        new_xyt[inds_to_do, :, 1] = new_y
        new_xyt[inds_to_do, :, 2] += twist_angles


"""
The crossover classes below are messy and only lightly documented. They are unfortunately
too difficult for me to clean up without potentially altering behavior.
"""


@dataclass
class Crossover(Move):
    """
    Replaces trees near a random point with transformed trees from a mate. This particular one
    is used mostly for non-tesselated problems.
    """
    min_N_trees: int = field(init=True, default=2)  # Minimum trees to replace
    max_N_trees_ratio: float = field(init=True, default=0.5)  # Max ratio of trees to replace
    simple_mate_location: bool = field(init=True, default=True)  # Use simplified mate location logic

    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])

        # Get h parameters (on GPU)
        h_params = new_h[inds_to_do]  # (N_moves, 3) on GPU
        mate_h_params = mate_sol.h[inds_mate]  # (N_moves, 3) on GPU

        # Generate all random values at once (vectorized RNG on GPU)
        h_sizes = h_params[:, 0]
        
        
        # Get all tree positions (N_moves, N_trees, 2) on GPU
        tree_positions_all = new_xyt[inds_to_do, :, :2]  # (N_moves, N_trees, 2)
        mate_positions_all = mate_sol.xyt[inds_mate, :, :2]  # (N_moves, N_trees, 2)

        # Generate n_trees_to_replace, rotation, and mirror for all
        min_trees = min(self.min_N_trees, N_trees)
        max_trees = min(int(np.round(self.max_N_trees_ratio * N_trees)), N_trees)
        if max_trees< min_trees:
            max_trees = min_trees
        n_trees_to_replace_all = generator.integers(min_trees, max_trees + 1, size=N_moves)
        rotation_choice_all = generator.integers(0, 4, size=N_moves)
        do_mirror_all = generator.integers(0, 2, size=N_moves) == 1

        # Generate mate offsets
        if self.simple_mate_location:
            # Estimate how far we have to be from the edge for the selected number of trees
            square_size = population.genotype.h[inds_to_do,0]*np.sqrt(n_trees_to_replace_all / population.genotype.N_trees)
            h_sizes -= square_size
            center_x_all, center_y_all = population.genotype.generate_move_centers(square_size/2, inds_to_do, generator)
            mate_center_x_all, mate_center_y_all = population.genotype.generate_move_centers(square_size/2, inds_to_do, generator)        
        else:          
            center_x_all, center_y_all = population.genotype.generate_move_centers(None, inds_to_do, generator)            
            # CuPy's generator doesn't have choice, use random binary values
            sign_x = generator.integers(0, 2, size=N_moves) * 2 - 1  # 0 or 1 -> -1 or 1
            sign_y = generator.integers(0, 2, size=N_moves) * 2 - 1  # 0 or 1 -> -1 or 1
            mate_h_sizes = mate_h_params[:, 0]            
            mate_offset_x_all = cp.abs(center_x_all - h_params[:, 1]) * (mate_h_sizes / h_sizes) * sign_x
            mate_offset_y_all = cp.abs(center_y_all - h_params[:, 2]) * (mate_h_sizes / h_sizes) * sign_y
            mate_center_x_all = mate_offset_x_all + mate_h_params[:, 1]
            mate_center_y_all = mate_offset_y_all + mate_h_params[:, 2]        


        # Compute L-infinity distances for all individuals at once (vectorized on GPU)
        # Shape: (N_moves, N_trees)
        center_x_all_2d = center_x_all[:, cp.newaxis]  # (N_moves, 1)
        center_y_all_2d = center_y_all[:, cp.newaxis]  # (N_moves, 1)
        mate_center_x_all_2d = mate_center_x_all[:, cp.newaxis]  # (N_moves, 1)
        mate_center_y_all_2d = mate_center_y_all[:, cp.newaxis]  # (N_moves, 1)

        distances_individual_all = cp.maximum(
            cp.abs(tree_positions_all[:, :, 0] - center_x_all_2d),
            cp.abs(tree_positions_all[:, :, 1] - center_y_all_2d)
        )  # (N_moves, N_trees)

        distances_mate_all = cp.maximum(
            cp.abs(mate_positions_all[:, :, 0] - mate_center_x_all_2d),
            cp.abs(mate_positions_all[:, :, 1] - mate_center_y_all_2d)
        )  # (N_moves, N_trees)

        # Sort trees by distance for all individuals (vectorized on GPU)
        sorted_individual_tree_ids = cp.argsort(distances_individual_all, axis=1)  # (N_moves, N_trees)
        sorted_mate_tree_ids = cp.argsort(distances_mate_all, axis=1)  # (N_moves, N_trees)

        # Work with max_trees to enable vectorization - pad with dummy values for variable sizes
        max_n_trees = int(cp.max(n_trees_to_replace_all))

        # Create mask for which trees are actually being replaced (N_moves, max_n_trees)
        tree_idx = cp.arange(max_n_trees)[cp.newaxis, :]  # (1, max_n_trees)
        valid_mask = tree_idx < n_trees_to_replace_all[:, cp.newaxis]  # (N_moves, max_n_trees)

        # Get tree IDs for all moves (take first max_n_trees, will mask invalid ones)
        individual_tree_ids_all = sorted_individual_tree_ids[:, :max_n_trees]  # (N_moves, max_n_trees)
        mate_tree_ids_all = sorted_mate_tree_ids[:, :max_n_trees]  # (N_moves, max_n_trees)

        # Compute centers of mass for selected trees (vectorized with masking)
        # Use valid_mask to only include trees that should be replaced
        # Shape gymnastics: need to gather positions for selected trees

        # For individual trees - gather positions using fancy indexing
        # individual_tree_ids_all has shape (N_moves, max_n_trees)
        # tree_positions_all has shape (N_moves, N_trees, 2)
        # We want (N_moves, max_n_trees, 2)
        move_indices = cp.arange(N_moves)[:, cp.newaxis]  # (N_moves, 1)
        selected_individual_positions = tree_positions_all[move_indices, individual_tree_ids_all]  # (N_moves, max_n_trees, 2)
        selected_mate_positions = mate_positions_all[move_indices, mate_tree_ids_all]  # (N_moves, max_n_trees, 2)

        # Compute centers of mass (with masking for valid trees only)
        # Sum only valid trees and divide by count
        individual_centers_x = cp.sum(selected_individual_positions[:, :, 0] * valid_mask, axis=1) / cp.sum(valid_mask, axis=1)  # (N_moves,)
        individual_centers_y = cp.sum(selected_individual_positions[:, :, 1] * valid_mask, axis=1) / cp.sum(valid_mask, axis=1)  # (N_moves,)
        mate_centers_x = cp.sum(selected_mate_positions[:, :, 0] * valid_mask, axis=1) / cp.sum(valid_mask, axis=1)  # (N_moves,)
        mate_centers_y = cp.sum(selected_mate_positions[:, :, 1] * valid_mask, axis=1) / cp.sum(valid_mask, axis=1)  # (N_moves,)

        # Gather all mate trees to transform (N_moves, max_n_trees, 3)
        # Use fancy indexing: for each move i, get mate_sol.xyt[inds_mate[i], mate_tree_ids_all[i, :], :]
        inds_mate_expanded = inds_mate[:, cp.newaxis]  # (N_moves, 1)
        mate_trees_all = mate_sol.xyt[inds_mate_expanded, mate_tree_ids_all].copy()  # (N_moves, max_n_trees, 3)

        # Apply vectorized transformation to all trees at once
        self._apply_transformation_vectorized(
            mate_trees_all,
            mate_centers_x, mate_centers_y,
            individual_centers_x, individual_centers_y,
            rotation_choice_all, do_mirror_all,
            valid_mask
        )

        # Scatter results back to population (fully vectorized with masking)
        # Use advanced indexing to write all valid trees at once
        # Create index arrays for all valid tree writes
        # Shape: for each (move_i, tree_j) where valid_mask[move_i, tree_j] is True,
        #        write mate_trees_all[move_i, tree_j, :] to new_xyt[inds_to_do[move_i], individual_tree_ids_all[move_i, tree_j], :]

        # Get indices of all valid (move, tree) pairs
        move_indices_flat, tree_indices_flat = cp.where(valid_mask)  # Both shape (N_valid_writes,)

        # For each valid write, get the corresponding individual and tree IDs
        individual_ids_flat = inds_to_do[move_indices_flat]  # (N_valid_writes,)
        tree_ids_flat = individual_tree_ids_all[move_indices_flat, tree_indices_flat]  # (N_valid_writes,)

        # Gather the transformed trees to write (N_valid_writes, 3)
        trees_to_write = mate_trees_all[move_indices_flat, tree_indices_flat, :]  # (N_valid_writes, 3)

        # Scatter write all at once (vectorized)
        new_xyt[individual_ids_flat, tree_ids_flat, :] = trees_to_write

    def _apply_transformation_vectorized(self, trees_all: cp.ndarray,
                                         src_center_x_all, src_center_y_all,
                                         dst_center_x_all, dst_center_y_all,
                                         rotation_choice_all, do_mirror_all,
                                         valid_mask):
        """Vectorized transformation for multiple moves at once.

        Parameters
        ----------
        trees_all : cp.ndarray
            Shape (N_moves, max_n_trees, 3) - trees to transform
        src_center_x_all, src_center_y_all : cp.ndarray
            Shape (N_moves,) - source centers for each move
        dst_center_x_all, dst_center_y_all : cp.ndarray
            Shape (N_moves,) - destination centers for each move
        rotation_choice_all : cp.ndarray
            Shape (N_moves,) - rotation choice (0, 1, 2, or 3) for each move
        do_mirror_all : cp.ndarray
            Shape (N_moves,) - boolean array for mirroring
        valid_mask : cp.ndarray
            Shape (N_moves, max_n_trees) - mask for which trees are actually being replaced
        """
        # Get positions relative to source centers (vectorized)
        # Expand centers to (N_moves, 1) for broadcasting
        src_x = src_center_x_all[:, cp.newaxis]  # (N_moves, 1)
        src_y = src_center_y_all[:, cp.newaxis]  # (N_moves, 1)
        dst_x = dst_center_x_all[:, cp.newaxis]  # (N_moves, 1)
        dst_y = dst_center_y_all[:, cp.newaxis]  # (N_moves, 1)

        dx = trees_all[:, :, 0] - src_x  # (N_moves, max_n_trees)
        dy = trees_all[:, :, 1] - src_y  # (N_moves, max_n_trees)
        theta = trees_all[:, :, 2].copy()  # (N_moves, max_n_trees)

        # Apply mirroring (vectorized with masking)
        mirror_mask = do_mirror_all[:, cp.newaxis]  # (N_moves, 1)
        dy = cp.where(mirror_mask, -dy, dy)
        theta = cp.where(mirror_mask, cp.pi - theta, theta)

        # Apply rotation (vectorized for all 4 rotation choices at once)
        # Compute rotation for all moves
        rot_angles = rotation_choice_all[:, cp.newaxis] * (cp.pi / 2)  # (N_moves, 1)
        cos_angles = cp.cos(rot_angles)  # (N_moves, 1)
        sin_angles = cp.sin(rot_angles)  # (N_moves, 1)

        # Apply rotation (only where rotation_choice != 0)
        rotation_mask = rotation_choice_all[:, cp.newaxis] != 0  # (N_moves, 1)
        dx_rotated = dx * cos_angles - dy * sin_angles
        dy_rotated = dx * sin_angles + dy * cos_angles

        dx = cp.where(rotation_mask, dx_rotated, dx)
        dy = cp.where(rotation_mask, dy_rotated, dy)
        theta = cp.where(rotation_mask, theta + rot_angles, theta)

        # Place at destination centers (vectorized)
        trees_all[:, :, 0] = dst_x + dx
        trees_all[:, :, 1] = dst_y + dy
        trees_all[:, :, 2] = theta

    def _apply_transformation(self, trees: cp.ndarray,
                              src_center_x, src_center_y,
                              dst_center_x, dst_center_y,
                              rotation_choice: int, do_mirror: bool):
        """Apply rotation and mirroring, moving trees from src_center to dst_center.
        All operations on GPU with CuPy."""
        # Get positions relative to source center (mate's center)
        dx = trees[:, 0] - src_center_x
        dy = trees[:, 1] - src_center_y
        theta = trees[:, 2]

        # Apply mirroring (across x-axis through center): y -> -y, theta -> pi - theta
        if do_mirror:
            dy = -dy
            theta = cp.pi - theta

        # Apply rotation (0°, 90°, 180°, or 270°)
        if rotation_choice != 0:
            rot_angle = rotation_choice * (cp.pi / 2)
            cos_a, sin_a = cp.cos(rot_angle), cp.sin(rot_angle)
            dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
            theta = theta + rot_angle

        # Place at destination center (individual's center)
        trees[:, 0] = dst_center_x + dx
        trees[:, 1] = dst_center_y + dy
        trees[:, 2] = theta

@dataclass
class Crossover2(Move):
    """
    Replaces trees near a random point with transformed trees from a mate. This particular one
    is used mostly for tesselated problems.
    """
    min_N_trees: int = field(init=True, default=2)  # Minimum trees to replace
    max_N_trees_ratio: float = field(init=True, default=0.5)  # Max ratio to replace
    distance_function: typing.Literal['stripe', 'square', 'square90', 'square180'] = (
        field(init=True, default='stripe')  # Distance metric
    )
    decouple_mate_location: bool = field(init=True, default=False)  # Independent mate center
    use_edge_clearance_when_decoupled: bool = field(init=True, default=True)  # Edge spacing
    do_90_rotation: bool = field(init=True, default=True)  # Allow 90° rotations

    def _do_move_vec(self, population: 'pack_ga3.Population', inds_to_do: cp.ndarray,
                     mate_sol: kgs.SolutionCollection, inds_mate: cp.ndarray,
                     generator: cp.random.Generator):
        """See do_move_vec."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])
        if N_moves == 0:
            return
        
        # Decide how many trees to replace and what transforms to apply
        min_trees = min(self.min_N_trees, N_trees)
        if self.do_90_rotation:
            rotation_choice_all = generator.integers(0, 4, size=N_moves)
        else:
            rotation_choice_all = 2 * generator.integers(0, 2, size=N_moves)
        do_mirror_all = generator.integers(0, 2, size=N_moves) == 1

        # Gather boundary parameters
        h_params = new_h[inds_to_do]
        mate_h_params = mate_sol.h[inds_mate]

        # Fetch the full set of tree coordinates for both parents involved
        tree_positions_all = new_xyt[inds_to_do, :, :2]
        mate_trees_full = mate_sol.xyt[inds_mate].copy()

        # Apply the random mirror/rotation to the mate data prior to selection
        self._apply_orientation_preselection(
            mate_trees_full,
            rotation_choice_all,
            do_mirror_all,
        )
        mate_sol.canonicalize_xyt(mate_trees_full)

        # Drop the theta channel for distance computations
        mate_positions_all = mate_trees_full[:, :, :2]

        # Build edge-spacer validity masks up front so they can be reused throughout
        if population.genotype.filter_move_locations_with_edge_spacer:
            margin = population.genotype.filter_move_locations_margin
            valid_ind_mask = population.genotype.edge_spacer.check_valid(
                population.genotype.xyt[inds_to_do],
                h_params,
                margin=margin,
            )
            valid_mate_mask = population.genotype.edge_spacer.check_valid(
                mate_trees_full,
                mate_h_params,
                margin=margin,
            )
        else:
            valid_ind_mask = cp.ones((N_moves, N_trees), dtype=bool)
            valid_mate_mask = valid_ind_mask

        # Ensure we never request more trees than are valid in either parent
        usable_counts = cp.minimum(
            valid_ind_mask.sum(axis=1),
            valid_mate_mask.sum(axis=1),
        ).astype(cp.int32)
        usable_counts = cp.maximum(usable_counts, 0)

        # Determine per-move min/max based on filtered availability
        min_trees_available = cp.minimum(min_trees, usable_counts)
        max_trees_available = cp.minimum(
            cp.floor(self.max_N_trees_ratio * usable_counts).astype(cp.int32),
            usable_counts,
        )
        max_trees_available = cp.maximum(max_trees_available, min_trees_available)

        # Sample n_trees uniformly per move within [min, max]
        span = max_trees_available - min_trees_available + 1
        rand_uniform = generator.random(size=N_moves)
        n_trees_to_replace_all = (min_trees_available + cp.floor(rand_uniform * span)).astype(cp.int32)

        # Sample a random point inside each square to anchor the crossover stripe
        if not self.decouple_mate_location:
            line_point_x, line_point_y = population.genotype.generate_move_centers(None, inds_to_do, generator)
            mate_line_point_x = line_point_x - h_params[:, 1] + mate_h_params[:, 1]
            mate_line_point_y = line_point_y - h_params[:, 2] + mate_h_params[:, 2]
        else:
            if self.use_edge_clearance_when_decoupled:
                square_size = population.genotype.h[inds_to_do,0]*cp.sqrt(n_trees_to_replace_all / (mate_sol.N_trees))
                line_point_x, line_point_y = population.genotype.generate_move_centers(square_size/2, inds_to_do, generator)
                mate_line_point_x, mate_line_point_y = population.genotype.generate_move_centers(square_size/2, inds_to_do, generator)
            else:
                line_point_x, line_point_y = population.genotype.generate_move_centers(None, inds_to_do, generator)
                mate_line_point_x, mate_line_point_y = population.genotype.generate_move_centers(None, inds_to_do, generator)

        if self.distance_function == 'stripe':
            # Draw random line orientations and compute their normals for distance tests
            line_angles = generator.uniform(0, 2 * cp.pi, size=N_moves)
            normal_x = -cp.sin(line_angles)
            normal_y = cp.cos(line_angles)

            # Broadcast reference points and line normals for batched projections
            line_point_x_2d = line_point_x[:, cp.newaxis]
            line_point_y_2d = line_point_y[:, cp.newaxis]
            mate_line_point_x_2d = mate_line_point_x[:, cp.newaxis]
            mate_line_point_y_2d = mate_line_point_y[:, cp.newaxis]
            normal_x_2d = normal_x[:, cp.newaxis]
            normal_y_2d = normal_y[:, cp.newaxis]

            tree_pos_jittered = tree_positions_all
            mate_pos_jittered = mate_positions_all

            # Compute absolute distances to each line for the target population
            distances_individual_all = cp.abs(
                (tree_pos_jittered[:, :, 0] - line_point_x_2d) * normal_x_2d +
                (tree_pos_jittered[:, :, 1] - line_point_y_2d) * normal_y_2d
            )

            # Repeat the distance evaluation for the transformed mate coordinates
            distances_mate_all = cp.abs(
                (mate_pos_jittered[:, :, 0] - mate_line_point_x_2d) * normal_x_2d +
                (mate_pos_jittered[:, :, 1] - mate_line_point_y_2d) * normal_y_2d
            )

        elif self.distance_function == 'square':
            # Compute the distances using L-infinity metric for both populations.
            line_point_x_2d = line_point_x[:, cp.newaxis]
            line_point_y_2d = line_point_y[:, cp.newaxis]
            mate_line_point_x_2d = mate_line_point_x[:, cp.newaxis]
            mate_line_point_y_2d = mate_line_point_y[:, cp.newaxis]

            tree_pos_jittered = tree_positions_all
            mate_pos_jittered = mate_positions_all

            distances_individual_all = cp.maximum(
                cp.abs(tree_pos_jittered[:, :, 0] - line_point_x_2d),
                cp.abs(tree_pos_jittered[:, :, 1] - line_point_y_2d)
            )

            distances_mate_all = cp.maximum(
                cp.abs(mate_pos_jittered[:, :, 0] - mate_line_point_x_2d),
                cp.abs(mate_pos_jittered[:, :, 1] - mate_line_point_y_2d)
            )
        elif self.distance_function == 'square90':
            assert isinstance(mate_sol, kgs.SolutionCollectionSquareSymmetric90)
            # Compute L-infinity distance considering all 4 rotational images (0°, 90°, 180°, 270°)
            # Each genotype tree at (x, y) appears at 4 phenotype positions:
            #   0°: (x, y), 90°: (-y, x), 180°: (-x, -y), 270°: (y, -x)
            # We take the minimum distance across all 4 images
            line_point_x_2d = line_point_x[:, cp.newaxis]
            line_point_y_2d = line_point_y[:, cp.newaxis]
            mate_line_point_x_2d = mate_line_point_x[:, cp.newaxis]
            mate_line_point_y_2d = mate_line_point_y[:, cp.newaxis]

            # Apply jitter for distance calculation only  
            tree_pos_jittered = tree_positions_all
            mate_pos_jittered = mate_positions_all

            # Compute L-infinity distances for all 4 rotational images using positions directly
            # Image 0: (x, y)
            dist_0 = cp.maximum(cp.abs(tree_pos_jittered[:, :, 0] - line_point_x_2d), cp.abs(tree_pos_jittered[:, :, 1] - line_point_y_2d))
            # Image 1: (-y, x)
            dist_1 = cp.maximum(cp.abs(-tree_pos_jittered[:, :, 1] - line_point_x_2d), cp.abs(tree_pos_jittered[:, :, 0] - line_point_y_2d))
            # Image 2: (-x, -y)
            dist_2 = cp.maximum(cp.abs(-tree_pos_jittered[:, :, 0] - line_point_x_2d), cp.abs(-tree_pos_jittered[:, :, 1] - line_point_y_2d))
            # Image 3: (y, -x)
            dist_3 = cp.maximum(cp.abs(tree_pos_jittered[:, :, 1] - line_point_x_2d), cp.abs(-tree_pos_jittered[:, :, 0] - line_point_y_2d))
            
            # Take minimum distance across all 4 images
            distances_individual_all = cp.minimum(cp.minimum(dist_0, dist_1), cp.minimum(dist_2, dist_3))

            # For mate population trees - compute distances for all 4 rotational images using jittered positions directly
            # Compute L-infinity distances for all 4 rotational images using jittered positions
            dist_0_mate = cp.maximum(cp.abs(mate_pos_jittered[:, :, 0] - mate_line_point_x_2d), cp.abs(mate_pos_jittered[:, :, 1] - mate_line_point_y_2d))
            dist_1_mate = cp.maximum(cp.abs(-mate_pos_jittered[:, :, 1] - mate_line_point_x_2d), cp.abs(mate_pos_jittered[:, :, 0] - mate_line_point_y_2d))
            dist_2_mate = cp.maximum(cp.abs(-mate_pos_jittered[:, :, 0] - mate_line_point_x_2d), cp.abs(-mate_pos_jittered[:, :, 1] - mate_line_point_y_2d))
            dist_3_mate = cp.maximum(cp.abs(mate_pos_jittered[:, :, 1] - mate_line_point_x_2d), cp.abs(-mate_pos_jittered[:, :, 0] - mate_line_point_y_2d))

            # Original minimum distances
            distances_mate_all = cp.minimum(cp.minimum(dist_0_mate, dist_1_mate), cp.minimum(dist_2_mate, dist_3_mate))
            
            # Now use original (non-jittered) mate positions for transformation logic
            x_mate = mate_positions_all[:, :, 0]
            y_mate = mate_positions_all[:, :, 1]
            theta_mate = mate_trees_full[:, :, 2]
            
            # Compute L-infinity distances for all 4 rotational images
            # Image 0: (x, y)
            dist_0_mate = cp.maximum(cp.abs(x_mate - mate_line_point_x_2d), cp.abs(y_mate - mate_line_point_y_2d))
            # Image 1: (-y, x) - 90° rotation
            dist_1_mate = cp.maximum(cp.abs(-y_mate - mate_line_point_x_2d), cp.abs(x_mate - mate_line_point_y_2d))
            # Image 2: (-x, -y) - 180° rotation
            dist_2_mate = cp.maximum(cp.abs(-x_mate - mate_line_point_x_2d), cp.abs(-y_mate - mate_line_point_y_2d))
            # Image 3: (y, -x) - 270° rotation
            dist_3_mate = cp.maximum(cp.abs(y_mate - mate_line_point_x_2d), cp.abs(-x_mate - mate_line_point_y_2d))
            
            # Stack distances and find which image is closest for each tree
            # Shape: (N_moves, N_trees, 4)
            all_distances_mate = cp.stack([dist_0_mate, dist_1_mate, dist_2_mate, dist_3_mate], axis=2)
            # Shape: (N_moves, N_trees) - indices 0-3 indicating which rotation is closest
            closest_rotation_mate = cp.argmin(all_distances_mate, axis=2)   
            
            # Apply the appropriate rotation to each tree based on which image was closest
            # Create masks for each rotation choice
            mask_rot0 = closest_rotation_mate == 0  # (N_moves, N_trees)
            mask_rot1 = closest_rotation_mate == 1
            mask_rot2 = closest_rotation_mate == 2
            mask_rot3 = closest_rotation_mate == 3
            
            # Initialize transformed coordinates (will be modified in-place)
            x_mate_transformed = cp.zeros_like(x_mate)
            y_mate_transformed = cp.zeros_like(y_mate)
            theta_mate_transformed = cp.zeros_like(theta_mate)
            
            # Apply transformations based on which rotation was chosen
            # Rotation 0 (0°): (x, y) -> (x, y), theta -> theta
            x_mate_transformed = cp.where(mask_rot0, x_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot0, y_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot0, theta_mate, theta_mate_transformed)
            
            # Rotation 1 (90°): (x, y) -> (-y, x), theta -> theta + π/2
            x_mate_transformed = cp.where(mask_rot1, -y_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot1, x_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot1, theta_mate + cp.pi/2, theta_mate_transformed)
            
            # Rotation 2 (180°): (x, y) -> (-x, -y), theta -> theta + π
            x_mate_transformed = cp.where(mask_rot2, -x_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot2, -y_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot2, theta_mate + cp.pi, theta_mate_transformed)
            
            # Rotation 3 (270°): (x, y) -> (y, -x), theta -> theta + 3π/2
            x_mate_transformed = cp.where(mask_rot3, y_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot3, -x_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot3, theta_mate + 3*cp.pi/2, theta_mate_transformed)
            
            # Update mate_positions_all and mate_trees_full with transformed coordinates
            mate_positions_all[:, :, 0] = x_mate_transformed
            mate_positions_all[:, :, 1] = y_mate_transformed
            mate_trees_full[:, :, 2] = theta_mate_transformed


        elif self.distance_function == 'square180':
            assert isinstance(mate_sol, kgs.SolutionCollectionSquareSymmetric180)
            # Compute L-infinity distance considering both rotational images (0°, 180°)
            # Each genotype tree at (x, y) appears at 2 phenotype positions:
            #   0°: (x, y), 180°: (-x, -y)
            # We take the minimum distance across both images
            line_point_x_2d = line_point_x[:, cp.newaxis]
            line_point_y_2d = line_point_y[:, cp.newaxis]
            mate_line_point_x_2d = mate_line_point_x[:, cp.newaxis]
            mate_line_point_y_2d = mate_line_point_y[:, cp.newaxis]

            # Apply jitter for distance calculation only
            tree_pos_jittered = tree_positions_all
            mate_pos_jittered = mate_positions_all

            # Compute L-infinity distances for both rotational images using positions directly
            # Image 0: (x, y)
            dist_0 = cp.maximum(cp.abs(tree_pos_jittered[:, :, 0] - line_point_x_2d), cp.abs(tree_pos_jittered[:, :, 1] - line_point_y_2d))
            # Image 1: (-x, -y)
            dist_1 = cp.maximum(cp.abs(-tree_pos_jittered[:, :, 0] - line_point_x_2d), cp.abs(-tree_pos_jittered[:, :, 1] - line_point_y_2d))
           
            # Take minimum distance across both images
            distances_individual_all = cp.minimum(dist_0, dist_1)

            # For mate population trees - compute distances for both rotational images using jittered positions directly
            # Compute L-infinity distances for both rotational images using jittered positions
            dist_0_mate = cp.maximum(cp.abs(mate_pos_jittered[:, :, 0] - mate_line_point_x_2d), cp.abs(mate_pos_jittered[:, :, 1] - mate_line_point_y_2d))
            dist_1_mate = cp.maximum(cp.abs(-mate_pos_jittered[:, :, 0] - mate_line_point_x_2d), cp.abs(-mate_pos_jittered[:, :, 1] - mate_line_point_y_2d))

            # Original minimum distances
            distances_mate_all = cp.minimum(dist_0_mate, dist_1_mate)
                
            x_mate = mate_positions_all[:, :, 0]
            y_mate = mate_positions_all[:, :, 1]
            theta_mate = mate_trees_full[:, :, 2]
            
            # Compute L-infinity distances for both rotational images
            # Image 0: (x, y)
            dist_0_mate = cp.maximum(cp.abs(x_mate - mate_line_point_x_2d), cp.abs(y_mate - mate_line_point_y_2d))
            # Image 1: (-x, -y) - 180° rotation
            dist_1_mate = cp.maximum(cp.abs(-x_mate - mate_line_point_x_2d), cp.abs(-y_mate - mate_line_point_y_2d))
            
            # Stack distances and find which image is closest for each tree
            # Shape: (N_moves, N_trees, 2)
            all_distances_mate = cp.stack([dist_0_mate, dist_1_mate], axis=2)
            # Shape: (N_moves, N_trees) - indices 0-1 indicating which rotation is closest
            closest_rotation_mate = cp.argmin(all_distances_mate, axis=2)            
           
            # Apply the appropriate rotation to each tree based on which image was closest
            # Create masks for each rotation choice
            mask_rot0 = closest_rotation_mate == 0  # (N_moves, N_trees)
            mask_rot1 = closest_rotation_mate == 1
            
            # Initialize transformed coordinates (will be modified in-place)
            x_mate_transformed = cp.zeros_like(x_mate)
            y_mate_transformed = cp.zeros_like(y_mate)
            theta_mate_transformed = cp.zeros_like(theta_mate)
            
            # Apply transformations based on which rotation was chosen
            # Rotation 0 (0°): (x, y) -> (x, y), theta -> theta
            x_mate_transformed = cp.where(mask_rot0, x_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot0, y_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot0, theta_mate, theta_mate_transformed)
            
            # Rotation 1 (180°): (x, y) -> (-x, -y), theta -> theta + π
            x_mate_transformed = cp.where(mask_rot1, -x_mate, x_mate_transformed)
            y_mate_transformed = cp.where(mask_rot1, -y_mate, y_mate_transformed)
            theta_mate_transformed = cp.where(mask_rot1, theta_mate + cp.pi, theta_mate_transformed)
            
            # Update mate_positions_all and mate_trees_full with transformed coordinates
            mate_positions_all[:, :, 0] = x_mate_transformed
            mate_positions_all[:, :, 1] = y_mate_transformed
            mate_trees_full[:, :, 2] = theta_mate_transformed

        else:
            raise Exception(f"Unknown distance function: {self.distance_function}")

        # Mask out invalid trees before ranking
        distances_individual_all = cp.where(valid_ind_mask, distances_individual_all, cp.inf)
        distances_mate_all = cp.where(valid_mate_mask, distances_mate_all, cp.inf)

        # Rank trees by proximity to the stripe for both populations
        sorted_individual_tree_ids = cp.argsort(distances_individual_all, axis=1)
        sorted_mate_tree_ids = cp.argsort(distances_mate_all, axis=1)

        # Build masks describing how many entries each move will actually use
        max_n_trees = int(cp.max(n_trees_to_replace_all))
        tree_idx = cp.arange(max_n_trees)[cp.newaxis, :]
        valid_mask = tree_idx < n_trees_to_replace_all[:, cp.newaxis]

        # Truncate the sorted indices to that shared maximum for batched work
        individual_tree_ids_all = sorted_individual_tree_ids[:, :max_n_trees]
        mate_tree_ids_all = sorted_mate_tree_ids[:, :max_n_trees]

        # Gather the mate trees that will supply the stripe replacement subset
        move_indices = cp.arange(N_moves)[:, cp.newaxis]
        mate_trees_all = mate_trees_full[move_indices, mate_tree_ids_all].copy()

        # Scatter the transformed trees directly into the population tensor
        move_indices_flat, tree_indices_flat = cp.where(valid_mask)
        individual_ids_flat = inds_to_do[move_indices_flat]
        tree_ids_flat = individual_tree_ids_all[move_indices_flat, tree_indices_flat]
        trees_to_write = mate_trees_all[move_indices_flat, tree_indices_flat, :]
        
        # Transform mate trees to proper coordinates (compensate for offset between mate and original points)
        if self.decouple_mate_location:
            # Get the offset between the mate selection center and the target selection center
            # Shape: (N_moves,) for each unique move
            offset_x = line_point_x - mate_line_point_x
            offset_y = line_point_y - mate_line_point_y
            
            # Apply translation to the trees being written
            # Need to map from move_indices_flat to the corresponding offset
            trees_to_write[:, 0] += offset_x[move_indices_flat]
            trees_to_write[:, 1] += offset_y[move_indices_flat]

        new_xyt[individual_ids_flat, tree_ids_flat, :] = trees_to_write

        

    def _apply_orientation_preselection(self, mate_trees_full: cp.ndarray,
                                        rotation_choice_all: cp.ndarray,
                                        do_mirror_all: cp.ndarray):
        """Apply orientation transforms (mirror/rotation) to mate trees."""
        mate_x = mate_trees_full[:, :, 0]
        mate_y = mate_trees_full[:, :, 1]
        mate_theta = mate_trees_full[:, :, 2]

        # Apply mirroring
        mirror_mask = do_mirror_all[:, cp.newaxis]
        mate_y = cp.where(mirror_mask, -mate_y, mate_y)
        mate_theta = cp.where(mirror_mask, cp.pi - mate_theta, mate_theta)

        # Precompute rotation angles
        rot_angles = rotation_choice_all[:, cp.newaxis] * (cp.pi / 2)
        cos_angles = cp.cos(rot_angles)
        sin_angles = cp.sin(rot_angles)

        # Apply rotations
        rotation_mask = rotation_choice_all[:, cp.newaxis] != 0
        rotated_x = mate_x * cos_angles - mate_y * sin_angles
        rotated_y = mate_x * sin_angles + mate_y * cos_angles
        mate_x = cp.where(rotation_mask, rotated_x, mate_x)
        mate_y = cp.where(rotation_mask, rotated_y, mate_y)
        mate_theta = cp.where(rotation_mask, mate_theta + rot_angles, mate_theta)

        # Write transformed coordinates back
        mate_trees_full[:, :, 0] = mate_x
        mate_trees_full[:, :, 1] = mate_y
        mate_trees_full[:, :, 2] = mate_theta
