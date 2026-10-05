"""EMNIST-Balanced and MNIST loading for the M0 base-paper replication.

Both loaders return plain ``TensorDataset`` splits of normalized ``(N, 1, 28, 28)``
float images and ``(N,)`` long labels, so ``Subset`` + ``DataLoader`` work for
federated clients and ``get_labels`` feeds the Dirichlet partitioner.

* ``load_emnist`` — the federated task (Balanced: 47 classes, 112,800 train /
  18,800 test; Solat & Lee 2025, Sec. 5.1).
* ``load_mnist``  — centralized pretraining data for the frozen backbone.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

MEAN, STD = 0.1307, 0.3081  # standard MNIST statistics, reused for EMNIST


def to_dataset(images: torch.Tensor, targets: torch.Tensor,
               transpose: bool = False) -> TensorDataset:
    """Convert raw uint8 images and labels to a normalized ``TensorDataset``.

    Args:
        images: ``(N, 28, 28)`` uint8 tensor.
        targets: ``(N,)`` integer labels.
        transpose: Swap height and width. torchvision's EMNIST images are stored
            transposed (rotated/flipped) relative to MNIST; this makes them upright.

    Returns:
        ``TensorDataset`` of ``(N, 1, 28, 28)`` float images and long labels.
    """
    if transpose:
        images = images.transpose(1, 2)
    x = (images.float().div(255.0) - MEAN) / STD
    return TensorDataset(x.unsqueeze(1).contiguous(), targets.long())


def load_emnist(root: str = "data", split: str = "balanced") -> dict[str, TensorDataset]:
    """Load EMNIST (downloads on first use).

    Args:
        root: Download / cache directory.
        split: EMNIST split name; the paper uses ``"balanced"``.

    Returns:
        Dict with ``train`` and ``eval`` (the official test set) datasets.
    """
    from torchvision.datasets import EMNIST

    out = {}
    for key, train in (("train", True), ("eval", False)):
        ds = EMNIST(root, split=split, train=train, download=True)
        out[key] = to_dataset(ds.data, ds.targets, transpose=True)
    logger.info("EMNIST-%s loaded: train=%d, eval=%d", split,
                len(out["train"]), len(out["eval"]))
    return out


def load_mnist(root: str = "data") -> dict[str, TensorDataset]:
    """Load MNIST digits for centralized backbone pretraining.

    Args:
        root: Download / cache directory.

    Returns:
        Dict with ``train`` and ``eval`` datasets.
    """
    from torchvision.datasets import MNIST

    out = {}
    for key, train in (("train", True), ("eval", False)):
        ds = MNIST(root, train=train, download=True)
        out[key] = to_dataset(ds.data, ds.targets)
    logger.info("MNIST loaded: train=%d, eval=%d", len(out["train"]), len(out["eval"]))
    return out


def get_labels(split: TensorDataset) -> np.ndarray:
    """Return the label array of a split (input to ``dirichlet_split``).

    Args:
        split: Dataset produced by ``to_dataset``.

    Returns:
        1-D numpy array of integer labels.
    """
    return split.tensors[1].numpy()
