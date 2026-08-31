"""Unit tests for accuracy and communication-cost metrics."""

import pytest
import torch

from src.utils.metrics import accuracy, cumulative_comm_mb, round_comm_mb


class TestAccuracy:
    def test_all_correct(self):
        assert accuracy(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 1])) == 1.0

    def test_all_wrong(self):
        assert accuracy(torch.tensor([1, 0]), torch.tensor([0, 1])) == 0.0

    def test_partial(self):
        assert accuracy(torch.tensor([0, 1, 0, 0]), torch.tensor([0, 1, 1, 1])) == 0.5

    def test_accepts_python_lists(self):
        assert accuracy([0, 1], [0, 1]) == 1.0

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            accuracy(torch.tensor([0, 1]), torch.tensor([0]))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            accuracy(torch.tensor([]), torch.tensor([]))


class TestRoundCommMb:
    def test_counts_download_and_upload_for_every_client(self):
        # K=10 clients, S=1.5 MB each, down + up = 2 * 10 * 1.5
        assert round_comm_mb(K=10, S=1.5) == 30.0

    def test_uplink_only_when_bidirectional_is_false(self):
        assert round_comm_mb(K=10, S=1.5, bidirectional=False) == 15.0

    def test_rejects_negative_client_count(self):
        with pytest.raises(ValueError):
            round_comm_mb(K=-1, S=1.5)


class TestCumulativeCommMb:
    def test_accumulates_over_rounds(self):
        assert cumulative_comm_mb(rounds=3, K=10, S=1.5) == 90.0

    def test_matches_repeated_round_cost(self):
        assert cumulative_comm_mb(rounds=4, K=5, S=0.6) == pytest.approx(
            4 * round_comm_mb(K=5, S=0.6)
        )
