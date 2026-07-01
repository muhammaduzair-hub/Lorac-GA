"""Main GA search loop for optimal K selection."""

from __future__ import annotations
from typing import Callable


def ga_search(
    fitness_fn: Callable[[int], float],
    K_min: int,
    K_max: int,
    population_size: int = 20,
    generations: int = 30,
    p_c: float = 0.5,
    p_m: float = 0.2,
    seed: int = 42,
) -> int:
    """Run GA to find K* that maximises fitness under bandwidth budget.

    Args:
        fitness_fn: Callable mapping K -> fitness score.
        K_min: Minimum candidate K value.
        K_max: Maximum candidate K value (total available clients).
        population_size: Chromosomes per generation (P).
        generations: Number of GA generations (G).
        p_c: Crossover probability.
        p_m: Mutation probability.
        seed: Random seed for reproducibility.

    Returns:
        Optimal K* found by the GA.

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError
