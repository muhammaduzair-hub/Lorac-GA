"""Non-IID client partitioning via a label-wise Dirichlet distribution.

For every class ``c`` the samples of that class are split across clients with
proportions drawn from ``Dir(alpha)``. Small ``alpha`` (0.3 in this thesis)
produces highly skewed, label-imbalanced clients; large ``alpha`` approaches an
IID split.

Empty-client handling: a skewed draw can leave a client with fewer than
``min_samples`` indices, which would crash local training (empty DataLoader).
Such clients are topped up by moving randomly chosen indices from the currently
largest client, repeatedly, until the floor is met. This preserves the total
sample count and keeps partitions disjoint, at the cost of slightly reducing
the skew of the biggest clients.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def dirichlet_split(
    labels: Sequence[int] | np.ndarray,
    num_clients: int = 100,
    alpha: float = 0.3,
    seed: int = 42,
    min_samples: int = 1,
) -> list[list[int]]:
    """Partition dataset indices across clients with a Dirichlet prior.

    Args:
        labels: Label of every sample in the training split.
        num_clients: Number of simulated clients.
        alpha: Dirichlet concentration; lower is more non-IID.
        seed: RNG seed for reproducibility.
        min_samples: Minimum indices each client must end up with.

    Returns:
        List of ``num_clients`` index lists, disjoint and covering every sample.

    Raises:
        ValueError: If the dataset cannot supply ``min_samples`` to every client
            or if arguments are non-positive.
    """
    labels = np.asarray(labels)
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0; got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got {alpha}")
    if len(labels) < num_clients * min_samples:
        raise ValueError(
            f"Cannot give {min_samples} sample(s) to each of {num_clients} clients "
            f"from {len(labels)} samples."
        )

    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(num_clients)]

    for cls in np.unique(labels):
        cls_indices = np.flatnonzero(labels == cls)
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Cumulative cut points; the last chunk absorbs the rounding remainder.
        cuts = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(cls_indices, cuts)):
            parts[client_id].extend(int(i) for i in chunk)

    _enforce_min_samples(parts, min_samples, rng)
    for part in parts:
        part.sort()

    logger.info(
        "Dirichlet split: clients=%d, alpha=%.3f, sizes min=%d max=%d",
        num_clients,
        alpha,
        min(len(p) for p in parts),
        max(len(p) for p in parts),
    )
    return parts


def _enforce_min_samples(
    parts: list[list[int]], min_samples: int, rng: np.random.Generator
) -> None:
    """Top up undersized clients from the largest client, in place."""
    if min_samples <= 0:
        return
    for client_id, part in enumerate(parts):
        while len(part) < min_samples:
            donor = max(range(len(parts)), key=lambda i: len(parts[i]))
            if len(parts[donor]) <= min_samples:
                raise ValueError(
                    "No donor client can spare samples; lower min_samples or "
                    "num_clients."
                )
            moved = parts[donor].pop(int(rng.integers(len(parts[donor]))))
            part.append(moved)


def summarize_split(
    parts: list[list[int]],
    labels: Sequence[int] | np.ndarray,
    num_classes: int | None = None,
) -> dict:
    """Describe a partition for reporting and thesis plots.

    Args:
        parts: Output of :func:`dirichlet_split`.
        labels: Label of every sample in the training split.
        num_classes: Class count; inferred from ``labels`` when omitted.

    Returns:
        Dict with ``num_clients``, ``total_samples``, ``client_sizes``,
        ``min_client_size``, ``max_client_size``, ``mean_client_size`` and a
        ``label_histogram`` of shape ``(num_clients, num_classes)``.
    """
    labels = np.asarray(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1

    sizes = [len(p) for p in parts]
    histogram = [
        np.bincount(labels[p], minlength=num_classes).tolist() if p else [0] * num_classes
        for p in parts
    ]
    return {
        "num_clients": len(parts),
        "num_classes": num_classes,
        "total_samples": int(sum(sizes)),
        "client_sizes": sizes,
        "min_client_size": min(sizes),
        "max_client_size": max(sizes),
        "mean_client_size": float(np.mean(sizes)),
        "label_histogram": histogram,
    }
