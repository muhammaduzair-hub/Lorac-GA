"""Unit tests for the K-only GA: operators, search loops and budget handling."""

import random

import pytest

from src.ga.chromosome import Chromosome
from src.ga.fitness import make_fitness, max_feasible_K
from src.ga.operators import crossover, mutate, tournament_selection
from src.ga.search import ga_search, greedy_search, random_search

R, S = 10, 0.0833  # paper's rounds and per-client payload


def peaked(K: int) -> float:
    """Synthetic A(K): single peak at K=12."""
    return 0.9 - 0.001 * (K - 12) ** 2


class TestOperators:
    def test_tournament_full_pool_picks_fittest(self):
        pop = [Chromosome(K=k, fitness=f) for k, f in [(1, 0.1), (2, 0.9), (3, 0.5)]]
        assert tournament_selection(pop, random.Random(0), tournament_size=3).K == 2

    def test_tournament_tie_prefers_smaller_K(self):
        pop = [Chromosome(K=9, fitness=0.5), Chromosome(K=3, fitness=0.5)]
        assert tournament_selection(pop, random.Random(0), tournament_size=2).K == 3

    def test_tournament_unevaluated_raises(self):
        with pytest.raises(ValueError):
            tournament_selection([Chromosome(K=1)], random.Random(0))

    def test_crossover_child_is_a_parent_gene(self):
        a, b = Chromosome(K=3), Chromosome(K=7)
        kids = {crossover(a, b, random.Random(i), p_c=1.0).K for i in range(50)}
        assert kids == {3, 7}

    def test_crossover_disabled_copies_parent_a(self):
        child = crossover(Chromosome(K=3), Chromosome(K=7), random.Random(0), p_c=0.0)
        assert child.K == 3

    def test_mutate_stays_in_bounds_and_changes(self):
        rng = random.Random(0)
        ks = {mutate(Chromosome(K=50), 1, 10, rng, p_m=1.0).K for _ in range(200)}
        assert ks <= set(range(1, 11)) and len(ks) > 1

    def test_mutate_disabled_keeps_K(self):
        assert mutate(Chromosome(K=5), 1, 10, random.Random(0), p_m=0.0).K == 5


class TestBudget:
    def test_max_feasible_K(self):
        assert max_feasible_K(B=10, R=R, S=S, K_max=100) == 12
        assert max_feasible_K(B=1, R=R, S=S, K_max=100) == 1
        assert max_feasible_K(B=0.5, R=R, S=S, K_max=100) == 0
        assert max_feasible_K(B=4000, R=R, S=S, K_max=100) == 100

    def test_make_fitness_interpolates_and_clamps(self):
        f = make_fitness({1: 0.5, 11: 0.9}, B=1e9, R=R, S=S)
        assert f(6) == pytest.approx(0.7)
        assert f(50) == pytest.approx(0.9)  # held constant beyond profile

    def test_binding_budget_caps_fitness(self):
        # B/C(2) = 1 / (10*2*0.0833) ~= 0.6 < A = 0.9
        f = make_fitness({1: 0.9, 100: 0.9}, B=1.0, R=R, S=S)
        assert f(1) == pytest.approx(0.9)
        assert f(2) == pytest.approx(1.0 / (R * 2 * S))

    def test_paper_setup_budget_never_binds_for_B_ge_100(self):
        # M0 observation: with S=0.0833 and A<=1, f(K)=A(K) for every K<=100.
        f = make_fitness({1: 0.9, 100: 0.9}, B=100, R=R, S=S)
        assert all(f(K) == pytest.approx(0.9) for K in range(1, 101))


class TestSearch:
    def test_ga_finds_known_optimum(self):
        res = ga_search(peaked, 1, 100, seed=42)
        assert res.K_star == 12
        assert res.fitness == pytest.approx(0.9)

    def test_ga_history_monotone_with_elitism(self):
        res = ga_search(peaked, 1, 100, generations=30, seed=1)
        assert len(res.history) == 30
        assert res.history == sorted(res.history)

    def test_ga_respects_feasible_upper_bound(self):
        res = ga_search(peaked, 1, 5, seed=42)
        assert res.K_star == 5

    def test_ga_is_reproducible(self):
        assert ga_search(peaked, 1, 100, seed=7) == ga_search(peaked, 1, 100, seed=7)

    def test_ga_no_feasible_K_raises(self):
        with pytest.raises(ValueError):
            ga_search(peaked, 1, 0)

    def test_random_and_greedy_return_valid_histories(self):
        for search in (random_search, greedy_search):
            res = search(peaked, 1, 100, iterations=30, seed=42)
            assert 1 <= res.K_star <= 100
            assert len(res.history) == 30
            assert res.history == sorted(res.history)

    def test_greedy_reaches_peak_on_unimodal_start(self):
        res = greedy_search(peaked, 1, 100, iterations=200, seed=3)
        assert res.K_star == 12
