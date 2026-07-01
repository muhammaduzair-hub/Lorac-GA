"""GA fitness functions for bandwidth-aware client selection."""

from __future__ import annotations


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
