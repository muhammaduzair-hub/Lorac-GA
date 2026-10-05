"""GA fitness functions for bandwidth-aware client selection."""

from __future__ import annotations

from bisect import bisect_left
from typing import Callable, Mapping


def comm_cost(K: int, R: int, S: float) -> float:
    """Compute total communication cost C(K).

    Args:
        K: Number of clients selected per round.
        R: Total number of communication rounds.
        S: LoRA adapter size per client in MB.

    Returns:
        Total communication cost R * K * S in MB.
    """
    return float(R * K * S)


def efficiency(K: int, B: float, R: int, S: float) -> float:
    """Compute bandwidth efficiency B / C(K).

    Args:
        K: Number of clients selected per round.
        B: Bandwidth budget in MB.
        R: Total number of communication rounds.
        S: LoRA adapter size per client in MB.

    Returns:
        Bandwidth efficiency B / (R * K * S).

    Raises:
        ValueError: If K is zero.
    """
    if K == 0:
        raise ValueError("K must be > 0; got K=0 causes division by zero in comm_cost.")
    return B / comm_cost(K, R, S)


def fitness(K: int, A_K: float, B: float, R: int, S: float) -> float:
    """Compute GA fitness f(K) = min(A(K), B / C(K)).

    Args:
        K: Number of clients selected per round.
        A_K: Empirical accuracy with K clients (profiled offline).
        B: Bandwidth budget in MB.
        R: Total number of communication rounds.
        S: LoRA adapter size per client in MB.

    Returns:
        Fitness score as minimum of accuracy and bandwidth efficiency.

    Raises:
        ValueError: If K is zero.
    """
    return min(A_K, efficiency(K, B, R, S))


def max_feasible_K(B: float, R: int, S: float, K_max: int) -> int:
    """Largest K satisfying the budget constraint C(K) = R*K*S <= B.

    Args:
        B: Bandwidth budget in MB.
        R: Total number of communication rounds.
        S: LoRA adapter size per client in MB.
        K_max: Total available clients.

    Returns:
        ``min(K_max, floor(B / (R * S)))``; 0 means even K=1 is infeasible.
    """
    return min(K_max, int(B // (R * S)))


def make_fitness(
    profile: Mapping[int, float], B: float, R: int, S: float
) -> Callable[[int], float]:
    """Build f(K) = min(A(K), B / C(K)) from a profiled accuracy table.

    A(K) is linearly interpolated between profiled K values and held constant
    outside the profiled range.

    Args:
        profile: Profiled accuracy A(K) keyed by client count.
        B: Bandwidth budget in MB.
        R: Total number of communication rounds.
        S: LoRA adapter size per client in MB.

    Returns:
        Callable mapping K to its fitness score.

    Raises:
        ValueError: If the profile is empty.
    """
    if not profile:
        raise ValueError("profile must contain at least one (K, accuracy) point.")
    ks = sorted(profile)
    accs = [profile[k] for k in ks]

    def f(K: int) -> float:
        i = bisect_left(ks, K)
        if i == 0:
            a = accs[0]
        elif i == len(ks):
            a = accs[-1]
        else:
            k0, k1 = ks[i - 1], ks[i]
            a = accs[i - 1] + (accs[i] - accs[i - 1]) * (K - k0) / (k1 - k0)
        return fitness(K, a, B, R, S)

    return f
