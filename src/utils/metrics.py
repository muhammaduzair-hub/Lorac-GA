"""Accuracy and communication-cost metrics for the federated simulation."""

from __future__ import annotations

from typing import Sequence

import torch


def accuracy(preds: Sequence[int] | torch.Tensor,
             labels: Sequence[int] | torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.

    Returns:
        Fraction of correct predictions in [0, 1].

    Raises:
        ValueError: If the inputs are empty or of different lengths.
    """
    preds = torch.as_tensor(preds)
    labels = torch.as_tensor(labels)
    if preds.numel() == 0:
        raise ValueError("Cannot compute accuracy on an empty prediction set.")
    if preds.shape[0] != labels.shape[0]:
        raise ValueError(
            f"preds and labels differ in length: {preds.shape[0]} vs {labels.shape[0]}"
        )
    return float((preds == labels).sum().item() / preds.shape[0])


def round_comm_mb(K: int, S: float, bidirectional: bool = True) -> float:
    """Communication volume of one federated round.

    Args:
        K: Clients selected this round.
        S: Adapter payload per client in MB.
        bidirectional: Count server->client download as well as the upload.
            The GA cost model `C = R * K * r * s0` is uplink-only; the logged
            wall-clock traffic counts both directions.

    Returns:
        Megabytes transferred in the round.

    Raises:
        ValueError: If K or S is negative.
    """
    if K < 0 or S < 0:
        raise ValueError(f"K and S must be non-negative; got K={K}, S={S}")
    return float((2 if bidirectional else 1) * K * S)


def cumulative_comm_mb(rounds: int, K: int, S: float,
                       bidirectional: bool = True) -> float:
    """Total communication volume after a number of identical rounds.

    Args:
        rounds: Number of completed rounds.
        K: Clients selected per round.
        S: Adapter payload per client in MB.
        bidirectional: See :func:`round_comm_mb`.

    Returns:
        Cumulative megabytes transferred.
    """
    return rounds * round_comm_mb(K, S, bidirectional)
