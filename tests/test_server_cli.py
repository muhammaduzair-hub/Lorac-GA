"""Unit tests for the M2 config loading / CLI override layer."""

import pytest

from src.fl.server import build_config, parse_args


class TestBuildConfig:
    def test_loads_the_shipped_m2_config(self):
        cfg = build_config("configs/m2_baseline.yaml")
        assert cfg.seed == 42
        assert cfg.K == 10
        assert cfg.R == 10
        assert cfg.r == 8
        assert cfg.num_clients == 100
        assert cfg.alpha_dirichlet == 0.3

    def test_cli_overrides_win_over_the_file(self):
        cfg = build_config("configs/m2_baseline.yaml", ["K=3", "R=2"])
        assert cfg.K == 3
        assert cfg.R == 2

    def test_override_keeps_the_declared_type(self):
        cfg = build_config("configs/m2_baseline.yaml", ["lr=1e-3"])
        assert isinstance(cfg.lr, float)

    def test_rejects_unknown_override_keys(self):
        with pytest.raises(KeyError):
            build_config("configs/m2_baseline.yaml", ["nonexistent_key=1"])

    def test_rejects_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            build_config("configs/does_not_exist.yaml")


class TestParseArgs:
    def test_defaults_to_the_m2_config(self):
        args = parse_args([])
        assert args.config == "configs/m2_baseline.yaml"
        assert args.overrides == []

    def test_collects_key_value_overrides(self):
        args = parse_args(["--config", "c.yaml", "K=5", "R=2"])
        assert args.config == "c.yaml"
        assert args.overrides == ["K=5", "R=2"]


class TestResultsPayload:
    def test_results_json_records_config_env_and_history(self, tmp_path, monkeypatch):
        import json

        from src.fl import server

        monkeypatch.setattr(
            server, "run_federated",
            lambda cfg: {"history": [{"round": 1, "test_acc": 0.8}],
                         "adapter_size_mb": 2.9583, "final_acc": 0.8},
        )
        out = tmp_path / "run"
        server.main(["--config", "configs/m2_baseline.yaml", f"output_dir={out}"])

        payload = json.loads((out / "results.json").read_text())
        assert payload["config"]["K"] == 10
        assert payload["env"]["torch"]           # captured, not empty
        assert payload["final_acc"] == 0.8
