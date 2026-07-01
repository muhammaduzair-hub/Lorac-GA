"""GA genetic operators: selection, crossover, mutation."""

from __future__ import annotations
from typing import Sequence

from src.ga.chromosome import Chromosome


def tournament_selection(
    population: Sequence[Chromosome],
    tournament_size: int = 3,
    seed: int = 42,
) -> Chromosome:
    """Select one parent via tournament selection.

    Args:
        population: Current population of evaluated chromosomes.
        tournament_size: Number of candidates drawn per tournament.
        seed: Random seed for reproducibility.

    Returns:
        Winning chromosome with highest fitness in the tournament.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def single_point_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    p_c: float = 0.5,
    seed: int = 42,
) -> tuple[Chromosome, Chromosome]:
    """Produce two offspring via single-point crossover on K encoding.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome.
        p_c: Crossover probability.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of two child chromosomes.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def mutate(
    chromosome: Chromosome,
    K_min: int,
    K_max: int,
    p_m: float = 0.2,
    seed: int = 42,
) -> Chromosome:
    """Mutate chromosome K value with probability p_m.

    Args:
        chromosome: Chromosome to mutate.
        K_min: Minimum allowed K value.
        K_max: Maximum allowed K value.
        p_m: Mutation probability.
        seed: Random seed for reproducibility.

    Returns:
        Mutated chromosome (may be unchanged if not triggered).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
