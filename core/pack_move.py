import numpy as np
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field


# Forward declaration for Population type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pack_ga2 import Population


# ============================================================
# Moves
# ============================================================
@dataclass
class Move(kgs.BaseClass):

    def do_move(self, population:'Population', mate_sol:kgs.SolutionCollection, individual_id:int,
                mate_id:int, generator:cp.random.Generator):
        """
        Single-individual move interface (for testing).

        Parameters
        ----------
        population : Population
            Target population where clone is already in place
        mate_sol : kgs.SolutionCollection
            Source solution collection to read from
        individual_id : int
            Index of individual in population to modify
        mate_id : int
            Index of mate individual in mate_sol to use for crossover
        generator : cp.random.Generator
            Random number generator (GPU-based)
        """
        # Convert to GPU arrays and call vectorized version
        inds_to_do = cp.array([individual_id], dtype=cp.int32)
        inds_mate = cp.array([mate_id], dtype=cp.int32)
        self.do_move_vec(population, inds_to_do, mate_sol, inds_mate, generator)

    def do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                    inds_mate:cp.ndarray, generator:cp.random.Generator):
        """
        Vectorized move interface.

        Parameters
        ----------
        population : Population
            Target population where clones are already in place
        inds_to_do : cp.ndarray
            Indices of individuals in population to modify (shape: N_moves,) - GPU array
        mate_sol : kgs.SolutionCollection
            Source solution collection to read from
        inds_mate : cp.ndarray
            Indices of mate individuals in mate_sol to use for crossover (shape: N_moves,) - GPU array
        generator : cp.random.Generator
            Random number generator (GPU-based)
        """
        self._do_move_vec(population, inds_to_do, mate_sol, inds_mate, generator)

    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """
        Default implementation: loop over individuals and call _do_move for each.
        Clones are assumed to already be in population at indices inds_to_do.
        Subclasses can override this for better performance.
        """
        inds_to_do_cpu = inds_to_do.get()
        inds_mate_cpu = inds_mate.get()
        for ind_to_do, ind_mate in zip(inds_to_do_cpu, inds_mate_cpu):
            self._do_move(population, mate_sol, ind_to_do, ind_mate, generator)    
    
class NoOp(Move):
    def _do_move(self, population, mate_sol, individual_id, mate_id, generator):
        return None

@dataclass
class MoveSelector(Move):
    moves: list = field(init=True, default_factory=list) # each move is [Move, name, weight]
    _probabilities: cp.ndarray = field(init=False, default=None)

    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """
        Vectorized move interface for MoveSelector.

        Loops over individuals, selecting which move type to use for each,
        then groups individuals by move type and calls do_move_vec on each group.
        """
        if self._probabilities is None:
            total_weight = sum([m[2] for m in self.moves])
            self._probabilities = cp.array([m[2]/total_weight for m in self.moves], dtype=kgs.dtype_cp)

        # First, select which move to use for each individual (GPU-based RNG)
        N_moves = int(inds_mate.shape[0])
        # CuPy's generator doesn't have choice, so use cumulative probabilities
        cumulative_probs = cp.cumsum(self._probabilities)
        random_vals = generator.uniform(0, 1, size=N_moves)
        chosen_move_ids_gpu = cp.searchsorted(cumulative_probs, random_vals)

        # Group individuals by chosen move type and execute moves
        for move_id in range(len(self.moves)):
            # Find all individuals that chose this move (on GPU)
            mask = chosen_move_ids_gpu == move_id
            if not cp.any(mask):
                continue

            # Get indices for this move type (using GPU fancy indexing)
            batch_inds_to_do = inds_to_do[mask]
            batch_inds_mate = inds_mate[mask]

            # Call do_move_vec for this move type on the batch
            self.moves[move_id][0].do_move_vec(
                population, batch_inds_to_do, mate_sol, batch_inds_mate, generator
            )

@dataclass
class MoveRandomTree(Move):
    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """Vectorized version: randomly reposition selected trees in multiple individuals."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])

        # Get h parameters (on GPU)
        h_params = new_h[inds_to_do]  # (N_moves, 3) on GPU
        h_sizes = h_params[:, 0]

        # Generate all random values at once (GPU-based RNG)
        trees_to_mutate_gpu = generator.integers(0, N_trees, size=N_moves)
        new_x_gpu = generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 1]
        new_y_gpu = generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 2]
        new_theta_gpu = generator.uniform(-cp.pi, cp.pi, size=N_moves)

        # Apply updates using fancy indexing (fully vectorized on GPU)
        new_xyt[inds_to_do, trees_to_mutate_gpu, 0] = new_x_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 1] = new_y_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 2] = new_theta_gpu  

@dataclass
class JiggleRandomTree(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """Vectorized version: jiggle random trees in multiple individuals."""
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])

        # Generate all random values at once (GPU-based RNG)
        trees_to_mutate_gpu = generator.integers(0, N_trees, size=N_moves)
        offset_x_gpu = generator.uniform(-self.max_xy_move, self.max_xy_move, size=N_moves)
        offset_y_gpu = generator.uniform(-self.max_xy_move, self.max_xy_move, size=N_moves)
        offset_theta_gpu = generator.uniform(-self.max_theta_move, self.max_theta_move, size=N_moves)

        # Apply updates using fancy indexing (fully vectorized on GPU)
        new_xyt[inds_to_do, trees_to_mutate_gpu, 0] += offset_x_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 1] += offset_y_gpu
        new_xyt[inds_to_do, trees_to_mutate_gpu, 2] += offset_theta_gpu   

@dataclass
class JiggleCluster(Move):
    max_xy_move: float = field(init=True, default=0.1)
    max_theta_move: float = field(init=True, default=np.pi)
    min_N_trees: int = field(init=True, default=2)
    max_N_trees: int = field(init=True, default=20)
    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """Fully vectorized version: jiggle variable numbers of trees in clusters."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])

        # Get h parameters (on GPU)
        h_params = new_h[inds_to_do]  # (N_moves, 3) on GPU
        h_sizes = h_params[:, 0]

        # Generate random centers (GPU-based RNG) - shape (N_moves, 1)
        center_x_all = (generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 1])[:, cp.newaxis]
        center_y_all = (generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 2])[:, cp.newaxis]

        # Generate n_trees_to_jiggle for all individuals (GPU-based RNG)
        max_jiggle = min(self.max_N_trees, N_trees)
        min_jiggle = min(self.min_N_trees, N_trees)
        n_trees_to_jiggle_all = generator.integers(min_jiggle, max_jiggle + 1, size=N_moves)  # (N_moves,)

        # Pre-generate all random offsets (GPU-based RNG) - reshape to (N_moves, max_jiggle)
        total_offsets_needed = N_moves * max_jiggle
        all_offset_x = generator.uniform(-self.max_xy_move, self.max_xy_move, size=total_offsets_needed).reshape(N_moves, max_jiggle)
        all_offset_y = generator.uniform(-self.max_xy_move, self.max_xy_move, size=total_offsets_needed).reshape(N_moves, max_jiggle)
        all_offset_theta = generator.uniform(-self.max_theta_move, self.max_theta_move, size=total_offsets_needed).reshape(N_moves, max_jiggle)

        # Get tree positions for all individuals (N_moves, N_trees, 2)
        tree_positions = new_xyt[inds_to_do, :, :2]

        # Compute distances to centers for all individuals at once (N_moves, N_trees)
        dx = tree_positions[:, :, 0] - center_x_all  # (N_moves, N_trees)
        dy = tree_positions[:, :, 1] - center_y_all  # (N_moves, N_trees)
        distances = dx**2 + dy**2  # (N_moves, N_trees)

        # Sort trees by distance for each individual and take the closest max_jiggle trees
        sorted_tree_indices = cp.argsort(distances, axis=1)[:, :max_jiggle]  # (N_moves, max_jiggle)

        # Create mask for which trees to actually jiggle based on n_trees_to_jiggle_all
        # Shape: (N_moves, max_jiggle) - True where tree_idx < n_trees_to_jiggle_all[move_idx]
        tree_indices = cp.arange(max_jiggle)[cp.newaxis, :]  # (1, max_jiggle)
        mask = tree_indices < n_trees_to_jiggle_all[:, cp.newaxis]  # (N_moves, max_jiggle)

        # Apply mask to offsets (zero out offsets for trees we don't want to jiggle)
        all_offset_x = all_offset_x * mask
        all_offset_y = all_offset_y * mask
        all_offset_theta = all_offset_theta * mask

        # Create index arrays for fancy indexing
        individual_indices = inds_to_do[:, cp.newaxis]  # (N_moves, 1)

        # Apply offsets to the closest trees (fully vectorized on GPU)
        new_xyt[individual_indices, sorted_tree_indices, 0] += all_offset_x
        new_xyt[individual_indices, sorted_tree_indices, 1] += all_offset_y
        new_xyt[individual_indices, sorted_tree_indices, 2] += all_offset_theta

@dataclass
class Translate(Move):
    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """Vectorized version: translate all trees in multiple individuals."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_moves = int(inds_to_do.shape[0])

        # Get h parameters (on GPU)
        h_params = new_h[inds_to_do]  # (N_moves, 3) on GPU

        # Generate random offsets (vectorized RNG on GPU)
        h_sizes = h_params[:, 0]
        offset_x_gpu = generator.uniform(-h_sizes / 2, h_sizes / 2)[:, cp.newaxis]  # (N_moves, 1)
        offset_y_gpu = generator.uniform(-h_sizes / 2, h_sizes / 2)[:, cp.newaxis]  # (N_moves, 1)
        h_sizes_gpu = h_params[:, 0:1]  # (N_moves, 1)

        # Apply translation with modulo (fully vectorized on GPU)
        new_xyt[inds_to_do, :, 0] = cp.mod(new_xyt[inds_to_do, :, 0] + offset_x_gpu, h_sizes_gpu) - h_sizes_gpu / 2
        new_xyt[inds_to_do, :, 1] = cp.mod(new_xyt[inds_to_do, :, 1] + offset_y_gpu, h_sizes_gpu) - h_sizes_gpu / 2
    
@dataclass
class Twist(Move):
    # Twist trees around a center. Angle of twist decreases linearly with distance from center
    min_radius: float = field(init=True, default=0.5)
    max_radius: float = field(init=True, default=2.)
    def _do_move_vec(self, population:'Population', inds_to_do:cp.ndarray, mate_sol:kgs.SolutionCollection,
                     inds_mate:cp.ndarray, generator:cp.random.Generator):
        """Vectorized version: twist trees around centers in multiple individuals."""
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_moves = int(inds_to_do.shape[0])

        # Get h parameters (on GPU)
        h_params = new_h[inds_to_do]  # (N_moves, 3) on GPU

        # Generate all random parameters at once (vectorized RNG on GPU)
        h_sizes = h_params[:, 0]
        center_x_gpu = (generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 1])[:, cp.newaxis]  # (N_moves, 1)
        center_y_gpu = (generator.uniform(-h_sizes / 2, h_sizes / 2) + h_params[:, 2])[:, cp.newaxis]  # (N_moves, 1)
        max_twist_angle_gpu = generator.uniform(-cp.pi, cp.pi, size=N_moves)[:, cp.newaxis]  # (N_moves, 1)
        radius_gpu = generator.uniform(self.min_radius, self.max_radius, size=N_moves)[:, cp.newaxis]  # (N_moves, 1)

        # Get tree positions (N_moves, N_trees)
        tree_x = new_xyt[inds_to_do, :, 0]
        tree_y = new_xyt[inds_to_do, :, 1]

        # Compute distances from center (fully vectorized on GPU)
        dx = tree_x - center_x_gpu
        dy = tree_y - center_y_gpu
        distances = cp.sqrt(dx**2 + dy**2)

        # Twist angle decreases linearly with distance
        twist_angles = max_twist_angle_gpu * cp.maximum(0, 1 - distances / radius_gpu)

        # Apply rotation around center point
        cos_angles = cp.cos(twist_angles)
        sin_angles = cp.sin(twist_angles)
        new_x = center_x_gpu + dx * cos_angles - dy * sin_angles
        new_y = center_y_gpu + dx * sin_angles + dy * cos_angles

        # Update all individuals at once (vectorized)
        new_xyt[inds_to_do, :, 0] = new_x
        new_xyt[inds_to_do, :, 1] = new_y
        new_xyt[inds_to_do, :, 2] += twist_angles


@dataclass
class Crossover(Move):
    """Replaces trees near a random point with transformed trees from a mate individual.
    
    Selects n trees closest to a random center point (using L-infinity distance for 
    square selection) and replaces them with the n closest trees from the mate,
    applying a random rotation (0/90/180/270°) and optional mirroring.
    """
    min_N_trees_ratio: float = field(init=True, default=4/np.sqrt(40)) # to be multiplied by sqrt(N_trees)
    max_N_trees_ratio: float = field(init=True, default=0.5) # to be multiplied by N_trees
    simple_mate_location: bool = field(init=True, default=True)

    def _do_move_vec(self, population: 'Population', inds_to_do: cp.ndarray, mate_sol: kgs.SolutionCollection,
                     inds_mate: cp.ndarray, generator: cp.random.Generator):
        """Fully vectorized version: crossover trees from mates into multiple individuals."""
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
        min_trees = min(int(np.round(self.min_N_trees_ratio * np.sqrt(N_trees))), N_trees)
        max_trees = min(int(np.round(self.max_N_trees_ratio * N_trees)), N_trees)
        if max_trees< min_trees:
            max_trees = min_trees
        n_trees_to_replace_all = generator.integers(min_trees, max_trees + 1, size=N_moves)
        rotation_choice_all = generator.integers(0, 4, size=N_moves)
        do_mirror_all = generator.integers(0, 2, size=N_moves) == 1

        # Generate mate offsets
        if self.simple_mate_location:
            # Estimate how far we have to be from the edge for the selected number of trees
            square_size = population.genotype.h[inds_to_do,0]*np.sqrt(n_trees_to_replace_all / N_trees)
            h_sizes -= square_size
            offset_x_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            offset_y_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            center_x_all = offset_x_all + h_params[:, 1]
            center_y_all = offset_y_all + h_params[:, 2]            
            mate_offset_x_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            mate_offset_y_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            mate_center_x_all = mate_offset_x_all + mate_h_params[:, 1]
            mate_center_y_all = mate_offset_y_all + mate_h_params[:, 2]        
        else:            
            offset_x_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            offset_y_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            # CuPy's generator doesn't have choice, use random binary values
            sign_x = generator.integers(0, 2, size=N_moves) * 2 - 1  # 0 or 1 -> -1 or 1
            sign_y = generator.integers(0, 2, size=N_moves) * 2 - 1  # 0 or 1 -> -1 or 1
            center_x_all = offset_x_all + h_params[:, 1]
            center_y_all = offset_y_all + h_params[:, 2]
            mate_h_sizes = mate_h_params[:, 0]            
            mate_offset_x_all = cp.abs(offset_x_all) * (mate_h_sizes / h_sizes) * sign_x
            mate_offset_y_all = cp.abs(offset_y_all) * (mate_h_sizes / h_sizes) * sign_y
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
        mask_expanded = valid_mask[:, :, cp.newaxis]  # (N_moves, max_n_trees, 1)
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
class CrossoverStripe(Move):
    """Stripe-based crossover that selects trees by distance to a random line."""
    min_N_trees_ratio: float = field(init=True, default=4/np.sqrt(40)) # to be multiplied by sqrt(N_trees)
    max_N_trees_ratio: float = field(init=True, default=0.5)

    def _do_move_vec(self, population: 'Population', inds_to_do: cp.ndarray, mate_sol: kgs.SolutionCollection,
                     inds_mate: cp.ndarray, generator: cp.random.Generator):
        # Cache configuration tensors and short-circuit if nothing to do
        new_h = population.genotype.h
        new_xyt = population.genotype.xyt
        N_trees = new_xyt.shape[1]
        N_moves = int(inds_to_do.shape[0])
        if N_moves == 0:
            return

        # Gather boundary parameters for both parents involved in the move
        h_params = new_h[inds_to_do]
        mate_h_params = mate_sol.h[inds_mate]

        # Sample a random point inside each square to anchor the crossover stripe
        h_sizes = h_params[:, 0]
        if isinstance(mate_sol, kgs.SolutionCollectionSquareSymmetric):
            offset_x_all = generator.uniform(-h_sizes / 2, 0.)
            offset_y_all = generator.uniform(-h_sizes / 2, 0.)
        else:
            offset_x_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
            offset_y_all = generator.uniform(-h_sizes / 2, h_sizes / 2)
        line_point_x = offset_x_all + h_params[:, 1]
        line_point_y = offset_y_all + h_params[:, 2]

        # Mirror that same point into mate coordinates after accounting for its square center
        mate_line_point_x = offset_x_all + mate_h_params[:, 1]
        mate_line_point_y = offset_y_all + mate_h_params[:, 2]

        # Draw random line orientations and compute their normals for distance tests
        line_angles = generator.uniform(0, 2 * cp.pi, size=N_moves)
        normal_x = -cp.sin(line_angles)
        normal_y = cp.cos(line_angles)

        # Decide how many trees swap hands and what transforms to apply to mates
        min_trees = min(int(np.round(self.min_N_trees_ratio * np.sqrt(N_trees))), N_trees)
        max_trees = min(int(np.round(self.max_N_trees_ratio * N_trees)), N_trees)
        if max_trees < min_trees:
            max_trees = min_trees
        n_trees_to_replace_all = generator.integers(min_trees, max_trees + 1, size=N_moves)
        rotation_choice_all = generator.integers(0, 4, size=N_moves)
        do_mirror_all = generator.integers(0, 2, size=N_moves) == 1

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

        # Broadcast reference points and line normals for batched projections
        line_point_x_2d = line_point_x[:, cp.newaxis]
        line_point_y_2d = line_point_y[:, cp.newaxis]
        mate_line_point_x_2d = mate_line_point_x[:, cp.newaxis]
        mate_line_point_y_2d = mate_line_point_y[:, cp.newaxis]
        normal_x_2d = normal_x[:, cp.newaxis]
        normal_y_2d = normal_y[:, cp.newaxis]

        # Compute absolute distances to each line for the target population
        distances_individual_all = cp.abs(
            (tree_positions_all[:, :, 0] - line_point_x_2d) * normal_x_2d +
            (tree_positions_all[:, :, 1] - line_point_y_2d) * normal_y_2d
        )

        # Repeat the distance evaluation for the transformed mate coordinates
        distances_mate_all = cp.abs(
            (mate_positions_all[:, :, 0] - mate_line_point_x_2d) * normal_x_2d +
            (mate_positions_all[:, :, 1] - mate_line_point_y_2d) * normal_y_2d
        )

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
        new_xyt[individual_ids_flat, tree_ids_flat, :] = trees_to_write

    def _apply_orientation_preselection(self, mate_trees_full: cp.ndarray,
                                        rotation_choice_all: cp.ndarray,
                                        do_mirror_all: cp.ndarray):
        # Grab mutable references to positions and orientations for the batched mates
        mate_x = mate_trees_full[:, :, 0]
        mate_y = mate_trees_full[:, :, 1]
        mate_theta = mate_trees_full[:, :, 2]

        # Optionally mirror across the x-axis (flip y and adjust heading)
        mirror_mask = do_mirror_all[:, cp.newaxis]
        mate_y = cp.where(mirror_mask, -mate_y, mate_y)
        mate_theta = cp.where(mirror_mask, cp.pi - mate_theta, mate_theta)

        # Precompute rotation angles (0, 90, 180, 270 degrees)
        rot_angles = rotation_choice_all[:, cp.newaxis] * (cp.pi / 2)
        cos_angles = cp.cos(rot_angles)
        sin_angles = cp.sin(rot_angles)

        # Apply rotations only to entries requesting a non-zero turn
        rotation_mask = rotation_choice_all[:, cp.newaxis] != 0
        rotated_x = mate_x * cos_angles - mate_y * sin_angles
        rotated_y = mate_x * sin_angles + mate_y * cos_angles
        mate_x = cp.where(rotation_mask, rotated_x, mate_x)
        mate_y = cp.where(rotation_mask, rotated_y, mate_y)
        mate_theta = cp.where(rotation_mask, mate_theta + rot_angles, mate_theta)

        # Write the transformed coordinates back in-place for downstream selection
        mate_trees_full[:, :, 0] = mate_x
        mate_trees_full[:, :, 1] = mate_y
        mate_trees_full[:, :, 2] = mate_theta
        
