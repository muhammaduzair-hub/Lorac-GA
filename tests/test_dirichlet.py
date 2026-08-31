"""Unit tests for the non-IID Dirichlet partitioner."""

import numpy as np
import pytest

from src.fl.dirichlet import dirichlet_split, summarize_split


@pytest.fixture
def labels() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=2000)


class TestDirichletSplit:
    def test_returns_one_index_list_per_client(self, labels):
        parts = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        assert len(parts) == 20

    def test_every_sample_assigned_exactly_once(self, labels):
        parts = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        flat = [i for p in parts for i in p]
        assert sorted(flat) == list(range(len(labels)))

    def test_partitions_do_not_overlap(self, labels):
        parts = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        seen: set[int] = set()
        for p in parts:
            assert not seen & set(p)
            seen |= set(p)

    def test_same_seed_reproduces_partition(self, labels):
        a = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        b = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        assert a == b

    def test_different_seed_changes_partition(self, labels):
        a = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=42)
        b = dirichlet_split(labels, num_clients=20, alpha=0.3, seed=7)
        assert a != b

    def test_no_client_is_left_empty(self, labels):
        # alpha=0.1 with 100 clients on 2 classes leaves empty clients unless
        # the redistribution step kicks in.
        parts = dirichlet_split(labels, num_clients=100, alpha=0.1, seed=42)
        assert all(len(p) >= 1 for p in parts)

    def test_min_samples_per_client_is_enforced(self, labels):
        parts = dirichlet_split(
            labels, num_clients=100, alpha=0.1, seed=42, min_samples=5
        )
        assert min(len(p) for p in parts) >= 5

    def test_small_alpha_is_more_non_iid_than_large_alpha(self, labels):
        def mean_majority_share(parts):
            shares = []
            for p in parts:
                counts = np.bincount(labels[p], minlength=2)
                shares.append(counts.max() / counts.sum())
            return float(np.mean(shares))

        skewed = dirichlet_split(labels, num_clients=20, alpha=0.1, seed=42)
        uniform = dirichlet_split(labels, num_clients=20, alpha=100.0, seed=42)
        assert mean_majority_share(skewed) > mean_majority_share(uniform)

    def test_raises_when_clients_exceed_samples(self):
        with pytest.raises(ValueError):
            dirichlet_split(np.array([0, 1, 0]), num_clients=10, alpha=0.3, seed=42)


class TestSummarizeSplit:
    def test_reports_per_client_sizes(self, labels):
        parts = dirichlet_split(labels, num_clients=10, alpha=0.3, seed=42)
        summary = summarize_split(parts, labels)
        assert summary["client_sizes"] == [len(p) for p in parts]

    def test_label_histogram_has_client_by_class_shape(self, labels):
        parts = dirichlet_split(labels, num_clients=10, alpha=0.3, seed=42)
        summary = summarize_split(parts, labels)
        assert np.asarray(summary["label_histogram"]).shape == (10, 2)

    def test_histogram_rows_sum_to_client_sizes(self, labels):
        parts = dirichlet_split(labels, num_clients=10, alpha=0.3, seed=42)
        summary = summarize_split(parts, labels)
        rows = np.asarray(summary["label_histogram"]).sum(axis=1).tolist()
        assert rows == summary["client_sizes"]

    def test_reports_totals_and_extremes(self, labels):
        parts = dirichlet_split(labels, num_clients=10, alpha=0.3, seed=42)
        summary = summarize_split(parts, labels)
        assert summary["num_clients"] == 10
        assert summary["total_samples"] == len(labels)
        assert summary["min_client_size"] == min(summary["client_sizes"])
        assert summary["max_client_size"] == max(summary["client_sizes"])
