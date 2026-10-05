"""Offline tests for EMNIST preprocessing (no download)."""

import pytest

torch = pytest.importorskip("torch")

from src.data.emnist_loader import MEAN, STD, get_labels, to_dataset  # noqa: E402
from src.fl.dirichlet import dirichlet_split  # noqa: E402


def _raw(n=20):
    g = torch.Generator().manual_seed(0)
    return (torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8, generator=g),
            torch.arange(n) % 5)


def test_shape_dtype_and_range():
    ds = to_dataset(*_raw())
    x, y = ds.tensors
    assert x.shape == (20, 1, 28, 28) and x.dtype == torch.float32
    assert y.dtype == torch.long
    assert x.min() >= (0 - MEAN) / STD - 1e-5 and x.max() <= (1 - MEAN) / STD + 1e-5


def test_transpose_swaps_axes():
    imgs, y = _raw()
    plain, flipped = to_dataset(imgs, y), to_dataset(imgs, y, transpose=True)
    assert torch.equal(flipped.tensors[0], plain.tensors[0].transpose(2, 3))


def test_get_labels_feeds_dirichlet_split():
    ds = to_dataset(*_raw(200))
    labels = get_labels(ds)
    parts = dirichlet_split(labels, num_clients=10, alpha=0.3, seed=42)
    assert sorted(i for p in parts for i in p) == list(range(200))
