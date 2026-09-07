"""Unit tests for the A(K, r) surface profiler.

The heavy federated runner is injected via ``run_fn``, so these tests exercise
ordering, resume-skip, aggregation and the JSON schema fully offline (no torch,
no data download).
"""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.fl.profiler import (
    aggregate_surface,
    load_surface,
    order_cells,
    profile_AKr,
)


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "model_name": "tiny-test",
            "num_labels": 2,
            "r": 8,
            "lora_alpha": 16,
            "K": 10,
            "R": 10,
            "seed": 42,
            "output_dir": str(tmp_path / "m3"),
        }
    )


def make_fake_run(calls=None):
    """A deterministic stand-in for run_federated that records call order.

    Accuracy is a smooth function of (K, r, seed); adapter size is 0.1*r so
    s0 = S/r is a constant 0.1, which the schema test checks.
    """
    def fake_run(cell, model=None, datasets=None):
        K, r, seed = int(cell.K), int(cell.r), int(cell.seed)
        if calls is not None:
            calls.append((K, r, seed))
        S = 0.1 * r
        acc = 0.5 + 0.002 * K + 0.01 * r + 0.001 * (seed - 42)
        comm = 2 * K * S * cell.R
        return {
            "adapter_size_mb": S,
            "final_acc": acc,
            "history": [{"round": cell.R, "comm_mb_cumulative": comm}],
        }
    return fake_run


class TestOrderCells:
    def test_full_grid_when_no_priority(self):
        cells = order_cells([2, 5], [4, 8])
        assert cells == [(2, 4), (2, 8), (5, 4), (5, 8)]

    def test_priority_cells_run_first(self):
        cells = order_cells([2, 5], [4, 8], priority_cells=[[5, 8], [2, 8]])
        assert cells[:2] == [(5, 8), (2, 8)]
        assert set(cells) == {(2, 4), (2, 8), (5, 4), (5, 8)}
        assert len(cells) == 4

    def test_priority_entries_outside_grid_are_ignored(self):
        cells = order_cells([2], [8], priority_cells=[[99, 99], [2, 8]])
        assert cells == [(2, 8)]

    def test_cell_subset_restricts_the_grid(self):
        cells = order_cells([2, 5, 10], [4, 8], cell_subset=[[5, 8], [10, 4]])
        assert set(cells) == {(5, 8), (10, 4)}


class TestAggregateSurface:
    def test_averages_accuracy_over_seeds(self):
        records = [
            {"K": 5, "r": 8, "seed": 42, "acc": 0.80, "comm_mb": 1.0, "s0": 0.1},
            {"K": 5, "r": 8, "seed": 43, "acc": 0.90, "comm_mb": 1.0, "s0": 0.1},
        ]
        surface = aggregate_surface(records)
        assert len(surface) == 1
        cell = surface[0]
        assert cell["acc_mean"] == pytest.approx(0.85)
        assert cell["acc_std"] == pytest.approx(0.05)
        assert cell["n_seeds"] == 2
        assert cell["seeds"] == [42, 43]

    def test_single_seed_has_zero_std(self):
        records = [{"K": 5, "r": 8, "seed": 42, "acc": 0.8, "comm_mb": 1.0, "s0": 0.1}]
        assert aggregate_surface(records)[0]["acc_std"] == 0.0

    def test_sorted_by_k_then_r(self):
        records = [
            {"K": 10, "r": 4, "seed": 42, "acc": 0.7, "comm_mb": 1.0, "s0": 0.1},
            {"K": 5, "r": 8, "seed": 42, "acc": 0.8, "comm_mb": 1.0, "s0": 0.1},
            {"K": 5, "r": 4, "seed": 42, "acc": 0.75, "comm_mb": 1.0, "s0": 0.1},
        ]
        keys = [(c["K"], c["r"]) for c in aggregate_surface(records)]
        assert keys == [(5, 4), (5, 8), (10, 4)]


class TestProfileAKr:
    def test_runs_every_cell_and_seed(self, cfg):
        calls = []
        out = profile_AKr(
            cfg, K_values=[2, 5], r_values=[4, 8], seeds=[42],
            run_fn=make_fake_run(calls),
        )
        assert len(calls) == 4
        assert len(out["records"]) == 4
        assert len(out["surface"]) == 4

    def test_json_schema_includes_required_fields(self, cfg):
        profile_AKr(cfg, K_values=[5], r_values=[8], seeds=[42],
                    run_fn=make_fake_run())
        path = Path(cfg.output_dir) / "A_Kr_surface.json"
        document = json.loads(path.read_text())
        record = document["records"][0]
        for field in ("K", "r", "seed", "acc", "comm_mb", "s0"):
            assert field in record
        assert record["s0"] == pytest.approx(0.1)  # S/r = (0.1*r)/r

    def test_priority_cells_are_profiled_first(self, cfg):
        calls = []
        profile_AKr(
            cfg, K_values=[2, 5], r_values=[4, 8], seeds=[42],
            priority_cells=[[5, 8]], run_fn=make_fake_run(calls),
        )
        assert calls[0] == (5, 8, 42)

    def test_resume_skips_already_profiled_triples(self, cfg):
        first_calls = []
        profile_AKr(cfg, K_values=[2, 5], r_values=[8], seeds=[42],
                    run_fn=make_fake_run(first_calls))
        assert len(first_calls) == 2

        second_calls = []
        out = profile_AKr(cfg, K_values=[2, 5], r_values=[8], seeds=[42, 43],
                          run_fn=make_fake_run(second_calls))
        # Only the new seed 43 for each cell should actually run.
        assert set(second_calls) == {(2, 8, 43), (5, 8, 43)}
        # No duplicate records: 2 cells x 2 seeds.
        assert len(out["records"]) == 4

    def test_aggregation_over_two_seeds_after_resume(self, cfg):
        profile_AKr(cfg, K_values=[5], r_values=[8], seeds=[42],
                    run_fn=make_fake_run())
        out = profile_AKr(cfg, K_values=[5], r_values=[8], seeds=[43],
                          run_fn=make_fake_run())
        cell = out["surface"][0]
        assert cell["n_seeds"] == 2
        assert cell["seeds"] == [42, 43]

    def test_cell_subset_limits_the_run(self, cfg):
        calls = []
        profile_AKr(
            cfg, K_values=[2, 5, 10], r_values=[4, 8], seeds=[42],
            cell_subset=[[5, 8]], run_fn=make_fake_run(calls),
        )
        assert calls == [(5, 8, 42)]

    def test_each_cell_gets_an_isolated_output_dir(self, cfg):
        seen_dirs = []

        def spy(cell, model=None, datasets=None):
            seen_dirs.append(cell.output_dir)
            return {"adapter_size_mb": 0.1 * cell.r, "final_acc": 0.8,
                    "history": [{"comm_mb_cumulative": 1.0}]}

        profile_AKr(cfg, K_values=[2, 5], r_values=[8], seeds=[42], run_fn=spy)
        assert len(set(seen_dirs)) == 2
        assert all("cells" in d for d in seen_dirs)

    def test_comm_mb_taken_from_history_cumulative(self, cfg):
        out = profile_AKr(cfg, K_values=[5], r_values=[8], seeds=[42],
                          run_fn=make_fake_run())
        # comm = 2*K*S*R = 2*5*(0.1*8)*10 = 80.0
        assert out["records"][0]["comm_mb"] == pytest.approx(80.0)


class TestLoadSurface:
    def test_missing_file_returns_empty_list(self, tmp_path):
        assert load_surface(tmp_path / "nope.json") == []

    def test_reads_back_records_written_by_profiler(self, cfg):
        profile_AKr(cfg, K_values=[5], r_values=[8], seeds=[42],
                    run_fn=make_fake_run())
        path = Path(cfg.output_dir) / "A_Kr_surface.json"
        records = load_surface(path)
        assert len(records) == 1
        assert records[0]["K"] == 5
