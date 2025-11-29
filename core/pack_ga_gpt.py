import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import kaggle_support as kgs
from dataclasses import dataclass, field, fields
from typeguard import typechecked
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely
import pack_cost

import random
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Tuple, Optional

import deap
import deap.base
import deap.creator
import deap.tools


# ============================================================
# Genome / Phenotype data structures (domain-specific payload)
# ============================================================

@dataclass
class Genome:
    """
    Genotype: can be overlapping / 'illegal'.

    TODO:
    - Put your actual representation here, e.g.:
        * np.ndarray of shape (n_trees, 3) for (x, y, theta)
        * plus any extra info you want moves to see.
    """
    data: Any


@dataclass
class Phenotype:
    """
    Legal (relaxed) configuration; used for scoring/output.

    TODO:
    - Put the post-relaxation representation here (could be same as Genome.data,
      or something richer if helpful).
    """
    data: Any


@dataclass
class RelaxInfo:
    """
    Optional diagnostics from relaxation.

    TODO:
    - Add fields like:
        * success: bool
        * n_iterations: int
        * max_displacement: float
        * etc.
    """
    data: Any = None


def clone_genome(g: Genome) -> Genome:
    """
    Deep-copy a genome.

    TODO:
    - If g.data is a numpy array, use g.data.copy().
    - Ensure all mutable fields are copied.
    """
    raise NotImplementedError


def random_genome() -> Genome:
    """
    Create a random initial genotype (CAN overlap).

    TODO:
    - Sample tree positions/angles randomly.
    - Optionally apply heavy jiggling here.
    """
    raise NotImplementedError


def tiling_genome() -> Genome:
    """
    Create an initial genotype using a tiling-based layout.

    TODO:
    - Implement a tiling or unit-cell based initialization.
    """
    raise NotImplementedError


def relax_to_legal(genome: Genome) -> Tuple[Phenotype, RelaxInfo]:
    """
    Map overlapping genotype to a legal, relaxed phenotype.

    TODO:
    - Implement your 'relaxed configuration' routine:
        * high overlap penalty, low edge pressure,
        * pure minimization (no noise/momentum) if this is pure legalization.
    """
    raise NotImplementedError


def compute_packing_fitness(phenotype: Phenotype) -> float:
    """
    Compute main packing fitness from a phenotype.

    TODO:
    - E.g. fitness = -square_side, or packing density.
    - Higher is better.
    """
    raise NotImplementedError


def distance_genotype_to_phenotype(genome: Genome, phenotype: Phenotype) -> float:
    """
    Optional: measure how far relaxation moved the layout.

    TODO:
    - E.g. average/max displacement/rotation over trees.
    - Can be used as a stability penalty term.
    """
    raise NotImplementedError


# ============================================================
# DEAP: Fitness + Individual (subclassing DEAP where it fits)
# ============================================================

# Fitness: single objective max
deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))

# Base individual type from DEAP's creator
deap.creator.create("PackingIndividualBase", object, fitness=deap.creator.FitnessMax)


class PackingIndividual(deap.creator.PackingIndividualBase):
    """
    Individual with attached Genome, age, and cached fitness.

    This class IS the DEAP individual type: it has .fitness from FitnessMax.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.genome: Optional[Genome] = None
        self.age: int = 0
        self.base_fitness: Optional[float] = None
        self.effective_fitness: Optional[float] = None
        self.last_move: Optional[Tuple[str, Hashable]] = None  # (move_name, param_key)
        # Internal field for move statistics
        self._fitness_before_move: Optional[float] = None


class PackingToolbox(deap.base.Toolbox):
    """
    Small subclass of DEAP's Toolbox, mostly so you can extend/customize later
    without changing external code.
    """
    pass


# ============================================================
# Move system: classes for moves + selector + stats
# ============================================================

@dataclass
class MoveOutcome:
    """Simple struct to pass back info from a move application."""
    move_name: str
    param_key: Hashable
    fitness_before: Optional[float]


class MoveBase:
    """
    Abstract base for a move operator.

    Each move:
    - Knows its name.
    - Can apply itself to an individual (mutating in-place).
    - Tracks its own success statistics keyed by hyperparameter signatures.
    """

    def __init__(self, name: str):
        self.name = name
        # stats[param_key] = {"calls": int, "improved": int, "delta_sum": float}
        self.stats: Dict[Hashable, Dict[str, float]] = {}

    def choose_params(self) -> Hashable:
        """
        Choose hyperparameters for this move for a single application and return
        a hashable 'param_key' describing them.

        TODO:
        - Implement sampling logic for move-specific hyperparameters.
        - For example:
            * strength bucket ("light", "heavy")
            * (step_size, angle_step) tuple
        """
        raise NotImplementedError

    def apply(self, individual: PackingIndividual, param_key: Hashable) -> None:
        """
        Apply the move to the individual's GENOTYPE (allowed to create overlaps).

        TODO:
        - Implement the actual transformation on individual.genome based on param_key.
        """
        raise NotImplementedError

    def record_outcome(self, param_key: Hashable, fitness_before: float, fitness_after: float) -> None:
        """
        Record success statistics for this move under a param_key.
        """
        if fitness_before is None or fitness_after is None:
            return

        d = self.stats.setdefault(param_key, {"calls": 0, "improved": 0, "delta_sum": 0.0})
        d["calls"] += 1
        if fitness_after > fitness_before:
            d["improved"] += 1
            d["delta_sum"] += (fitness_after - fitness_before)

    def summary(self) -> Dict[Hashable, Dict[str, float]]:
        """
        Return current stats for external logging/inspection.
        """
        return self.stats


class JiggleMove(MoveBase):
    """
    Example concrete move: jiggling (non-local exploration).
    """

    def __init__(self):
        super().__init__(name="jiggle")

    def choose_params(self) -> Hashable:
        """
        TODO:
        - Choose a jiggling 'strength' and anything else you care about.
        - Return a hashable key, e.g. ("strength", "light") or just "light".
        """
        # Example placeholder:
        strength = random.choice(["light", "heavy"])
        return strength

    def apply(self, individual: PackingIndividual, param_key: Hashable) -> None:
        """
        TODO:
        - Implement jiggling of individual.genome based on param_key.
        - This is where you do compact+relax with noise/momentum on the GENOTYPE.
        """
        # Example structure:
        # if param_key == "light": ...
        # elif param_key == "heavy": ...
        pass


class SwapTreesMove(MoveBase):
    """
    Example move: swap two trees' positions/orientations.
    """

    def __init__(self):
        super().__init__(name="swap_trees")

    def choose_params(self) -> Hashable:
        """
        TODO:
        - Possibly encode which selection strategy or size bucket you used.
        """
        return None  # e.g. one mode only

    def apply(self, individual: PackingIndividual, param_key: Hashable) -> None:
        """
        TODO:
        - Pick two tree indices and swap them in individual.genome.
        """
        pass


class TranslateLayoutMove(MoveBase):
    """
    Example move: translate entire layout.
    """

    def __init__(self):
        super().__init__(name="translate_layout")

    def choose_params(self) -> Hashable:
        """
        TODO:
        - Sample dx, dy, and bucket them into 'small', 'medium', 'large', etc.
        - Return that bucket label as param_key.
        """
        bucket = "small"  # placeholder
        return bucket

    def apply(self, individual: PackingIndividual, param_key: Hashable) -> None:
        """
        TODO:
        - Apply translation to all tree positions in individual.genome.
        - Magnitude based on param_key.
        """
        pass


class MoveSelector:
    """
    Chooses which MoveBase to apply given a list of moves and (optionally)
    move-level probabilities or adaptive logic.

    This is your central 'mutation controller' per N.
    """

    def __init__(self, moves: List[MoveBase]):
        self.moves = moves
        # TODO: optionally store selection probabilities per move and update them
        # based on stats (e.g. using bandit logic).

    def select_move(self) -> MoveBase:
        """
        Choose a move to apply.

        TODO:
        - Implement selection logic:
            * simple uniform choice,
            * or probability-weighted based on past success,
            * or UCB/bandit-style.
        """
        return random.choice(self.moves)

    def apply_random_move(self, individual: PackingIndividual) -> MoveOutcome:
        """
        Apply a randomly selected move to the individual and return a MoveOutcome
        describing what happened.

        - Does NOT know fitness_after yet; that will be handled by the GA run
          after evaluation.
        """
        move = self.select_move()
        param_key = move.choose_params()

        # Remember fitness before move for later stats (GA will set this)
        fitness_before = individual.base_fitness

        # Apply in-place on genotype
        move.apply(individual, param_key)

        # Record metadata on the individual so we can link outcome to move later
        individual.last_move = (move.name, param_key)
        individual._fitness_before_move = fitness_before

        return MoveOutcome(move_name=move.name, param_key=param_key, fitness_before=fitness_before)

    def record_outcomes_from_population(self, population: List[PackingIndividual]) -> None:
        """
        Once the GA has re-evaluated the population, call this to update stats
        inside each move based on the before/after fitness.

        Assumes:
        - individual.last_move = (move_name, param_key)
        - individual._fitness_before_move set before mutation
        - individual.base_fitness updated after evaluation
        """
        # Build a dict from move_name to MoveBase for quick lookup
        move_dict = {m.name: m for m in self.moves}

        for ind in population:
            if ind.last_move is None or not hasattr(ind, "_fitness_before_move"):
                continue

            move_name, param_key = ind.last_move
            fitness_before = ind._fitness_before_move
            fitness_after = ind.base_fitness

            move_obj = move_dict.get(move_name, None)
            if move_obj is not None:
                move_obj.record_outcome(param_key, fitness_before, fitness_after)

            # Clean up
            del ind._fitness_before_move

    def all_move_stats(self) -> Dict[str, Dict[Hashable, Dict[str, float]]]:
        """
        Aggregate stats for all moves for logging.
        """
        return {m.name: m.summary() for m in self.moves}


# ============================================================
# GA config
# ============================================================

@dataclass
class GAConfig:
    """
    Hyperparameters for a single GA run (for a fixed N).
    """
    pop_size: int = 1024
    n_generations: int = 100
    cx_prob: float = 0.7
    mut_prob: float = 0.3
    elitism: int = 10
    use_batched_eval: bool = True
    frac_tiling_init: float = 0.5


# ============================================================
# GA runner class (per N)
# ============================================================

class GARun:
    """
    Orchestrates the whole GA for a single N:
    - Population management
    - DEAP toolbox
    - Moves & logging
    - Evaluation (batched or per-individual)
    """

    def __init__(self, config: GAConfig, N: int):
        self.config = config
        self.N = N  # number of trees for this run

        # Toolbox as a subclass of DEAP Toolbox
        self.toolbox = PackingToolbox()

        # Moves and selector
        self.moves: List[MoveBase] = self._create_moves_for_N(N)
        self.move_selector = MoveSelector(self.moves)

        # Register DEAP-like operators into the toolbox
        self._setup_toolbox()

        # Population
        self.population: List[PackingIndividual] = []

    def _create_moves_for_N(self, N: int) -> List[MoveBase]:
        """
        Create the list of move objects for this N.

        TODO:
        - Add moves conditionally on N if needed.
        - Include jiggle, swap, translate, rotate-section, etc.
        """
        return [
            JiggleMove(),
            SwapTreesMove(),
            TranslateLayoutMove(),
            # TODO: add more MoveBase subclasses here
        ]

    # ---------- Toolbox registration and population init ----------

    def _init_individual_random(self) -> PackingIndividual:
        ind = PackingIndividual()
        ind.genome = random_genome()
        return ind

    def _init_individual_tiling(self) -> PackingIndividual:
        ind = PackingIndividual()
        ind.genome = tiling_genome()
        return ind

    def _setup_toolbox(self) -> None:
        """
        Register individual init, selection, crossover, and mutation hooks
        in the toolbox.
        """
        self.toolbox.register("individual_random", self._init_individual_random)
        self.toolbox.register("individual_tiling", self._init_individual_tiling)

        # DEAP selection; can be replaced later
        self.toolbox.register("select", deap.tools.selTournament, tournsize=3)

        # Crossover and mutation will delegate to class methods
        self.toolbox.register("mate", self._crossover_operator)
        self.toolbox.register("mutate", self._mutation_operator)

        # Evaluation will be handled by our own methods, not toolbox.evaluate,
        # but you can also register it if you like:
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _init_population(self) -> None:
        """
        Initialize population with mix of random + tiling-based individuals.
        """
        pop_size = self.config.pop_size
        frac_tiling = self.config.frac_tiling_init

        n_tiling = int(pop_size * frac_tiling)
        self.population = []
        for _ in range(n_tiling):
            self.population.append(self.toolbox.individual_tiling())
        for _ in range(pop_size - n_tiling):
            self.population.append(self.toolbox.individual_random())

    # ---------- Variation operators (using MoveSelector for mutation) ----------

    def _crossover_operator(
        self,
        ind1: PackingIndividual,
        ind2: PackingIndividual,
    ) -> Tuple[PackingIndividual, PackingIndividual]:
        """
        Crossover: recombine genotypes of two individuals.

        TODO:
        - Implement 'swap sections between parents' here:
            * choose spatial region(s) or index subset(s),
            * exchange corresponding trees between ind1.genome and ind2.genome.
        - You can record last_move for children if you want to track crossover success.
        """
        # Placeholder: no-op crossover
        return ind1, ind2

    def _mutation_operator(self, individual: PackingIndividual) -> Tuple[PackingIndividual]:
        """
        Single mutation entry point for DEAP, delegating to MoveSelector.
        """
        self.move_selector.apply_random_move(individual)
        return (individual,)

    # ---------- Evaluation (batched or per-individual) ----------

    def _evaluate_individual(self, ind: PackingIndividual) -> Tuple[float]:
        """
        Fallback per-individual evaluation; used if batched eval is disabled.
        """
        genome_copy = clone_genome(ind.genome)
        phenotype, relax_info = relax_to_legal(genome_copy)

        # TODO: define relaxation_failed using relax_info
        relaxation_failed = False
        if relaxation_failed:
            base_fitness = -1e9
        else:
            base_fitness = compute_packing_fitness(phenotype)

        # Optional stability penalty
        use_stability_penalty = False
        if use_stability_penalty:
            dist = distance_genotype_to_phenotype(ind.genome, phenotype)
            stability_weight = 0.0  # TODO
            base_fitness -= stability_weight * dist

        ind.base_fitness = base_fitness
        return (base_fitness,)

    def _batched_relax_and_score(self, inds: List[PackingIndividual]) -> None:
        """
        Batched evaluation: this is where you exploit massive parallelism.

        TODO:
        - Pack all genomes into big array(s) suitable for GPU/vectorized relaxation.
        - Run your relaxation (and any evaluation-time jiggling) on ALL of them.
        - Compute packing fitness for each phenotype.
        - Optionally, compute genotype->phenotype distance for stability penalty.
        - Store result in ind.base_fitness for each individual.
        """
        raise NotImplementedError

    # ---------- Diversity / age bonuses and effective fitness ----------

    def _compute_diversity_bonus(self) -> Dict[PackingIndividual, float]:
        """
        Compute diversity bonus per individual.

        TODO:
        - Define a distance over genomes (or phenotypes).
        - Implement sharing/clustering / kNN-based bonuses.
        """
        return {ind: 0.0 for ind in self.population}

    def _compute_youth_bonus(self) -> Dict[PackingIndividual, float]:
        """
        Compute youth bonus per individual based on age.

        TODO:
        - E.g. bonus = max(0, max_age - age) * weight.
        """
        return {ind: 0.0 for ind in self.population}

    def _update_effective_fitness(self) -> None:
        """
        Combine base fitness, diversity, and youth into .effective_fitness and .fitness.
        """
        div_bonus = self._compute_diversity_bonus()
        youth_bonus = self._compute_youth_bonus()

        for ind in self.population:
            base = ind.base_fitness
            eff = base + div_bonus[ind] + youth_bonus[ind]
            ind.effective_fitness = eff
            ind.fitness.values = (eff,)

    # ---------- Main run loop ----------

    def run(self) -> None:
        """
        Main GA loop for this N.
        """
        cfg = self.config

        # Initialize pop and evaluate
        self._init_population()

        if cfg.use_batched_eval:
            self._batched_relax_and_score(self.population)
        else:
            for ind in self.population:
                self.toolbox.evaluate(ind)

        self._update_effective_fitness()

        for gen in range(cfg.n_generations):
            print(f"[N={self.N}] Generation {gen}")

            # Elitism
            n_elite = cfg.elitism
            elites = deap.tools.selBest(self.population, n_elite)

            # Select parents for offspring
            parents = self.toolbox.select(self.population, cfg.pop_size - n_elite)
            offspring: List[PackingIndividual] = [deap.tools.clone(ind) for ind in parents]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cfg.cx_prob:
                    c1, c2 = offspring[i - 1], offspring[i]
                    self.toolbox.mate(c1, c2)
                    # Optionally tag crossover in last_move

            # Mutation via MoveSelector
            for ind in offspring:
                if random.random() < cfg.mut_prob:
                    # store fitness before move (for move stats)
                    ind._fitness_before_move = ind.base_fitness
                    self.toolbox.mutate(ind)

            # Age individuals
            for ind in elites + offspring:
                ind.age += 1

            # Evaluate offspring
            if cfg.use_batched_eval:
                self._batched_relax_and_score(offspring)
            else:
                for ind in offspring:
                    self.toolbox.evaluate(ind)

            # Record per-move success stats now that base_fitness is known
            self.move_selector.record_outcomes_from_population(offspring)

            # New population = elites + offspring
            self.population = elites + offspring

            # Update effective fitness (with diversity, youth, etc.)
            self._update_effective_fitness()

            # TODO:
            # - Log best/mean fitness
            # - Log self.move_selector.all_move_stats()
            # - Possibly adapt move probabilities based on stats
            # - Check for convergence / early stopping

        # After run, best individual is:
        best = deap.tools.selBest(self.population, 1)[0]
        print(f"[N={self.N}] Best base_fitness:", best.base_fitness)
        # TODO:
        # - Optionally store best.genome, relax and output final phenotype, etc.
