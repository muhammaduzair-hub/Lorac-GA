"""Search loops for the optimal client count K: LoRaC-GA plus two baselines.

``ga_search`` follows Algorithm 1 of Solat & Lee (2025). ``random_search`` and
``greedy_search`` are the comparison curves of their Fig. 3; the paper does not
define them precisely, so the definitions here are ours (see each docstring).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from src.ga.chromosome import Chromosome
from src.ga.operators import crossover, mutate, tournament_selection


@dataclass
class SearchResult:
    """Outcome of a K search.

    Attributes:
        K_star: Best client count found.
        fitness: Fitness of ``K_star``.
        history: Best-so-far fitness after each iteration (generation).
    """

    K_star: int
    fitness: float
    history: list[float] = field(default_factory=list)


def _best(candidates: list[Chromosome]) -> Chromosome:
    return max(candidates, key=lambda c: (c.fitness, -c.K))


def ga_search(
    fitness_fn: Callable[[int], float],
    K_min: int,
    K_max: int,
    population_size: int = 20,
    generations: int = 30,
    p_c: float = 0.5,
    p_m: float = 0.2,
    elitism: float = 0.1,
    tournament_size: int = 3,
    seed: int = 42,
) -> SearchResult:
    """Run LoRaC-GA to find K* that maximises fitness under a bandwidth budget.

    Args:
        fitness_fn: Callable mapping K -> fitness score.
        K_min: Minimum candidate K value.
        K_max: Maximum candidate K value (budget-feasible upper bound).
        population_size: Chromosomes per generation (P).
        generations: Number of GA generations (G).
        p_c: Crossover probability.
        p_m: Mutation probability.
        elitism: Fraction of top individuals carried over unchanged (rho).
        tournament_size: Candidates per tournament.
        seed: Random seed for reproducibility.

    Returns:
        ``SearchResult`` with K*, its fitness and the per-generation best.

    Raises:
        ValueError: If ``K_min > K_max`` or the population is empty.
    """
    if K_min > K_max:
        raise ValueError(f"K_min={K_min} > K_max={K_max}: no feasible K.")
    if population_size < 1:
        raise ValueError("population_size must be >= 1.")

    rng = random.Random(seed)
    n_elite = max(1, round(elitism * population_size))
    population = [
        Chromosome(K=rng.randint(K_min, K_max)) for _ in range(population_size)
    ]
    history: list[float] = []

    for _ in range(generations):
        for c in population:
            c.fitness = fitness_fn(c.K)
        population.sort(key=lambda c: (c.fitness, -c.K), reverse=True)
        history.append(population[0].fitness)

        offspring = []
        for _ in range(population_size - n_elite):
            a = tournament_selection(population, rng, tournament_size)
            b = tournament_selection(population, rng, tournament_size)
            child = crossover(a, b, rng, p_c)
            offspring.append(mutate(child, K_min, K_max, rng, p_m))
        population = population[:n_elite] + offspring

    for c in population:
        c.fitness = fitness_fn(c.K)
    best = _best(population)
    return SearchResult(best.K, best.fitness, history)


def random_search(
    fitness_fn: Callable[[int], float],
    K_min: int,
    K_max: int,
    iterations: int = 30,
    seed: int = 42,
) -> SearchResult:
    """Baseline: evaluate one uniform random K per iteration, keep the best.

    Args:
        fitness_fn: Callable mapping K -> fitness score.
        K_min: Minimum candidate K value.
        K_max: Maximum candidate K value.
        iterations: Number of random evaluations.
        seed: Random seed for reproducibility.

    Returns:
        ``SearchResult`` with the best K seen and its best-so-far history.
    """
    rng = random.Random(seed)
    best = Chromosome(K=rng.randint(K_min, K_max))
    best.fitness = fitness_fn(best.K)
    history = [best.fitness]
    for _ in range(iterations - 1):
        c = Chromosome(K=rng.randint(K_min, K_max))
        c.fitness = fitness_fn(c.K)
        best = _best([best, c])
        history.append(best.fitness)
    return SearchResult(best.K, best.fitness, history)


def greedy_search(
    fitness_fn: Callable[[int], float],
    K_min: int,
    K_max: int,
    iterations: int = 30,
    seed: int = 42,
) -> SearchResult:
    """Baseline: hill-climb from a random start, moving to the better of K-1/K+1.

    Stops improving at the first local optimum (history stays flat).

    Args:
        fitness_fn: Callable mapping K -> fitness score.
        K_min: Minimum candidate K value.
        K_max: Maximum candidate K value.
        iterations: Number of climbing steps.
        seed: Random seed for the starting point.

    Returns:
        ``SearchResult`` with the K reached and its fitness history.
    """
    rng = random.Random(seed)
    K = rng.randint(K_min, K_max)
    best_f = fitness_fn(K)
    history = [best_f]
    for _ in range(iterations - 1):
        neighbours = [k for k in (K - 1, K + 1) if K_min <= k <= K_max]
        scored = [(fitness_fn(k), -k, k) for k in neighbours]
        if scored:
            f, _, k = max(scored)
            if f > best_f:
                K, best_f = k, f
        history.append(best_f)
    return SearchResult(K, best_f, history)
