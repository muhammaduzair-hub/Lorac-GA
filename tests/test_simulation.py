"""Unit tests for the FedAvg simulation core."""

import pytest
import torch
from omegaconf import OmegaConf

from src.fl import simulation
from src.fl.simulation import fedavg_aggregate, run_federated, run_round, select_clients
from src.models.lora_wrap import adapter_size_mb, get_adapter_state
from tests.conftest import make_tiny_dataset


class TestFedavgAggregate:
    def test_equal_weights_give_the_plain_mean(self):
        states = [{"w": torch.tensor([0.0, 2.0])}, {"w": torch.tensor([2.0, 4.0])}]
        out = fedavg_aggregate(states, [10, 10])
        assert torch.allclose(out["w"], torch.tensor([1.0, 3.0]))

    def test_weights_by_client_sample_count(self):
        states = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([10.0])}]
        # (0*30 + 10*10) / 40 = 2.5
        out = fedavg_aggregate(states, [30, 10])
        assert torch.allclose(out["w"], torch.tensor([2.5]))

    def test_single_client_returns_its_own_state(self):
        states = [{"w": torch.tensor([3.0, -1.0])}]
        assert torch.allclose(fedavg_aggregate(states, [7])["w"],
                              torch.tensor([3.0, -1.0]))

    def test_does_not_mutate_the_input_states(self):
        original = torch.tensor([1.0, 1.0])
        states = [{"w": original}, {"w": torch.tensor([3.0, 3.0])}]
        fedavg_aggregate(states, [1, 1])
        assert torch.allclose(original, torch.tensor([1.0, 1.0]))

    def test_rejects_empty_state_list(self):
        with pytest.raises(ValueError):
            fedavg_aggregate([], [])

    def test_rejects_weight_count_mismatch(self):
        with pytest.raises(ValueError):
            fedavg_aggregate([{"w": torch.zeros(1)}], [1, 2])

    def test_rejects_zero_total_weight(self):
        with pytest.raises(ValueError):
            fedavg_aggregate([{"w": torch.zeros(1)}], [0])

    def test_rejects_states_with_different_keys(self):
        with pytest.raises(KeyError):
            fedavg_aggregate([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}], [1, 1])


class TestSelectClients:
    def test_returns_k_distinct_client_ids(self):
        ids = select_clients(num_clients=100, K=10, seed=42, rnd=0)
        assert len(ids) == 10
        assert len(set(ids)) == 10
        assert all(0 <= i < 100 for i in ids)

    def test_same_seed_and_round_reproduce_the_selection(self):
        a = select_clients(num_clients=100, K=10, seed=42, rnd=3)
        b = select_clients(num_clients=100, K=10, seed=42, rnd=3)
        assert a == b

    def test_different_rounds_select_different_clients(self):
        a = select_clients(num_clients=100, K=10, seed=42, rnd=0)
        b = select_clients(num_clients=100, K=10, seed=42, rnd=1)
        assert a != b

    def test_rejects_k_larger_than_client_pool(self):
        with pytest.raises(ValueError):
            select_clients(num_clients=5, K=10, seed=42, rnd=0)

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            select_clients(num_clients=5, K=0, seed=42, rnd=0)


class TestRunRound:
    def test_returns_a_state_with_the_global_parameter_names(self, tiny_model,
                                                             tiny_dataset):
        global_state = get_adapter_state(tiny_model)
        partition = [list(range(0, 16)), list(range(16, 32))]
        new_state = run_round(
            tiny_model, tiny_dataset, partition, [0, 1], global_state,
            lr=1e-3, local_epochs=1, batch_size=8, device="cpu",
        )
        assert set(new_state) == set(global_state)

    def test_aggregation_is_weighted_by_client_shard_size(self, tiny_model,
                                                          tiny_dataset, monkeypatch):
        seen_weights: list[int] = []

        def fake_local_train(model, loader, lr, local_epochs, device):
            return {"w": torch.tensor([float(len(loader.dataset))])}

        def spy_aggregate(states, weights):
            seen_weights.extend(weights)
            return states[0]

        monkeypatch.setattr(simulation, "local_train", fake_local_train)
        monkeypatch.setattr(simulation, "fedavg_aggregate", spy_aggregate)

        partition = [list(range(0, 24)), list(range(24, 32))]
        run_round(
            tiny_model, tiny_dataset, partition, [0, 1], get_adapter_state(tiny_model),
            lr=1e-3, local_epochs=1, batch_size=8, device="cpu",
        )
        assert seen_weights == [24, 8]

    def test_every_client_starts_from_the_same_global_state(self, tiny_model,
                                                            tiny_dataset, monkeypatch):
        starting_states = []

        def fake_local_train(model, loader, lr, local_epochs, device):
            starting_states.append(get_adapter_state(model))
            return get_adapter_state(model)

        monkeypatch.setattr(simulation, "local_train", fake_local_train)
        global_state = get_adapter_state(tiny_model)
        partition = [list(range(0, 16)), list(range(16, 32))]
        run_round(
            tiny_model, tiny_dataset, partition, [0, 1], global_state,
            lr=1e-3, local_epochs=1, batch_size=8, device="cpu",
        )
        key = next(iter(global_state))
        assert torch.allclose(starting_states[0][key], starting_states[1][key])

    def test_skips_clients_whose_shard_is_empty(self, tiny_model, tiny_dataset,
                                                monkeypatch):
        trained: list[int] = []

        def fake_local_train(model, loader, lr, local_epochs, device):
            trained.append(len(loader.dataset))
            return {"w": torch.zeros(1)}

        monkeypatch.setattr(simulation, "local_train", fake_local_train)
        monkeypatch.setattr(simulation, "fedavg_aggregate", lambda s, w: s[0])
        partition = [[], list(range(0, 8))]
        run_round(
            tiny_model, tiny_dataset, partition, [0, 1], get_adapter_state(tiny_model),
            lr=1e-3, local_epochs=1, batch_size=8, device="cpu",
        )
        assert trained == [8]


@pytest.fixture
def smoke_cfg(tmp_path):
    return OmegaConf.create(
        {
            "model_name": "tiny-test",
            "num_labels": 2,
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "max_len": 8,
            "num_clients": 4,
            "alpha_dirichlet": 0.3,
            "min_samples_per_client": 2,
            "max_samples_per_client": None,
            "K": 2,
            "R": 2,
            "local_epochs": 1,
            "batch_size": 8,
            "eval_batch_size": 8,
            "lr": 1e-3,
            "seed": 42,
            "device": "cpu",
            "output_dir": str(tmp_path / "run"),
        }
    )


@pytest.fixture
def smoke_datasets():
    return {"train": make_tiny_dataset(n=40, seed=1),
            "eval": make_tiny_dataset(n=16, seed=2)}


class TestRunFederated:
    def test_logs_one_record_per_round(self, smoke_cfg, smoke_datasets, tiny_model):
        result = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        assert [h["round"] for h in result["history"]] == [1, 2]

    def test_each_record_carries_accuracy_and_communication(self, smoke_cfg,
                                                            smoke_datasets, tiny_model):
        result = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        record = result["history"][0]
        assert {"round", "test_acc", "comm_mb", "comm_mb_cumulative"} <= set(record)
        assert 0.0 <= record["test_acc"] <= 1.0

    def test_communication_accumulates_as_two_times_k_times_s(self, smoke_cfg,
                                                              smoke_datasets,
                                                              tiny_model):
        result = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        per_round = 2 * smoke_cfg.K * adapter_size_mb(tiny_model)
        assert result["history"][0]["comm_mb_cumulative"] == pytest.approx(per_round)
        assert result["history"][1]["comm_mb_cumulative"] == pytest.approx(2 * per_round)

    def test_reports_the_measured_adapter_size(self, smoke_cfg, smoke_datasets,
                                               tiny_model):
        result = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        assert result["adapter_size_mb"] == pytest.approx(adapter_size_mb(tiny_model))

    def test_checkpoints_every_round(self, smoke_cfg, smoke_datasets, tiny_model):
        from pathlib import Path

        run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        out = Path(smoke_cfg.output_dir)
        assert (out / "checkpoint_round1.pt").exists()
        assert (out / "checkpoint_round2.pt").exists()

    def test_resumes_instead_of_restarting(self, smoke_cfg, smoke_datasets, tiny_model):
        first = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        smoke_cfg.R = 4
        second = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        assert [h["round"] for h in second["history"]] == [1, 2, 3, 4]
        assert second["history"][0]["test_acc"] == first["history"][0]["test_acc"]

    def test_split_summary_is_reported_for_the_thesis(self, smoke_cfg, smoke_datasets,
                                                      tiny_model):
        result = run_federated(smoke_cfg, model=tiny_model, datasets=smoke_datasets)
        summary = result["split_summary"]
        assert summary["num_clients"] == 4
        assert sum(summary["client_sizes"]) == 40
