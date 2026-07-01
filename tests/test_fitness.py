"""Unit tests for GA fitness functions and Chromosome dataclass."""

import pytest

from src.ga.fitness import comm_cost, efficiency, fitness
from src.ga.chromosome import Chromosome


class TestCommCost:
    def test_basic(self):
        assert comm_cost(K=5, R=10, S=0.1) == 5.0

    def test_returns_float(self):
        assert isinstance(comm_cost(K=1, R=1, S=1), float)


class TestEfficiency:
    def test_basic(self):
        # 100 / (10 * 5 * 0.1) = 100 / 5.0 = 20.0
        assert efficiency(K=5, B=100, R=10, S=0.1) == 20.0

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            efficiency(K=0, B=100, R=10, S=0.1)


class TestFitness:
    def test_budget_loose(self):
        # efficiency=20.0 > A_K=0.9, so fitness = A_K
        result = fitness(K=5, A_K=0.9, B=100, R=10, S=0.1)
        assert result == pytest.approx(0.9)

    def test_budget_tight(self):
        # efficiency = 1/(10*5*0.1) = 0.2 < A_K=0.9, so fitness = efficiency
        result = fitness(K=5, A_K=0.9, B=1, R=10, S=0.1)
        assert result == pytest.approx(0.2)


class TestChromosome:
    def test_default_fitness_is_none(self):
        c = Chromosome(K=10)
        assert c.fitness is None

    def test_k_stored(self):
        c = Chromosome(K=42)
        assert c.K == 42

    def test_fitness_assignable(self):
        c = Chromosome(K=5, fitness=0.75)
        assert c.fitness == pytest.approx(0.75)
