"""GA genetic operators: selection, crossover, mutation (Solat & Lee, Sec. 4.3)."""

from __future__ import annotations

import random
from typing import Sequence

from src.ga.chromosome import Chromosome


def tournament_selection(
    population: Sequence[Chromosome],
    rng: random.Random,
    tournament_size: int = 3,
) -> Chromosome:
    """Select one parent via tournament selection.

    Ties go to the smaller K (cheaper in communication).

    Args:
        population: Current population of evaluated chromosomes.
        rng: Random generator (owned by the caller so runs are reproducible).
        tournament_size: Number of candidates drawn per tournament.

    Returns:
        Winning chromosome with highest fitness in the tournament.

    Raises:
        ValueError: If the population is empty or a chromosome is unevaluated.
    """
    if not population:
        raise ValueError("population must not be empty.")
    contenders = rng.sample(list(population), min(tournament_size, len(population)))
    if any(c.fitness is None for c in contenders):
        raise ValueError("tournament_selection needs evaluated chromosomes.")
    return max(contenders, key=lambda c: (c.fitness, -c.K))


def crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rng: random.Random,
    p_c: float = 0.5,
) -> Chromosome:
    """Produce one offspring from two parents.

    With probability ``p_c`` the child takes K from ``parent_a`` or ``parent_b``
    with equal probability (paper Sec. 4.3); otherwise it copies ``parent_a``.
    On a single integer gene this is the only meaningful crossover.

    Args:
        parent_a: First parent chromosome.
        parent_b: Second parent chromosome.
        rng: Random generator.
        p_c: Crossover probability.

    Returns:
        New (unevaluated) child chromosome.
    """
    if rng.random() < p_c:
        return Chromosome(K=rng.choice((parent_a.K, parent_b.K)))
    return Chromosome(K=parent_a.K)


def mutate(
    chromosome: Chromosome,
    K_min: int,
    K_max: int,
    rng: random.Random,
    p_m: float = 0.2,
) -> Chromosome:
    """Replace K with a uniform random value in ``[K_min, K_max]`` w.p. ``p_m``.

    Args:
        chromosome: Chromosome to mutate.
        K_min: Minimum allowed K value.
        K_max: Maximum allowed K value.
        rng: Random generator.
        p_m: Mutation probability.

    Returns:
        New (unevaluated) chromosome; K unchanged if mutation is not triggered.
    """
    if rng.random() < p_m:
        return Chromosome(K=rng.randint(K_min, K_max))
    return Chromosome(K=chromosome.K)
