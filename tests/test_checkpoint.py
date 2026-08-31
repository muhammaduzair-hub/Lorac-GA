"""Unit tests for round-level checkpointing and resume."""

import json

import torch

from src.utils.checkpoint import load_latest_checkpoint, save_checkpoint


def _state() -> dict[str, torch.Tensor]:
    return {"lora_A": torch.ones(2, 2), "lora_B": torch.zeros(2)}


class TestSaveCheckpoint:
    def test_writes_adapter_and_history_files(self, tmp_path):
        save_checkpoint(tmp_path, rnd=1, adapter_state=_state(), history=[{"round": 1}])
        assert (tmp_path / "checkpoint_round1.pt").exists()
        assert (tmp_path / "history.json").exists()

    def test_history_json_is_human_readable(self, tmp_path):
        save_checkpoint(
            tmp_path, rnd=2, adapter_state=_state(), history=[{"round": 2, "acc": 0.9}]
        )
        loaded = json.loads((tmp_path / "history.json").read_text())
        assert loaded == [{"round": 2, "acc": 0.9}]

    def test_creates_output_dir_when_missing(self, tmp_path):
        target = tmp_path / "nested" / "run"
        save_checkpoint(target, rnd=1, adapter_state=_state(), history=[])
        assert (target / "checkpoint_round1.pt").exists()


class TestLoadLatestCheckpoint:
    def test_returns_none_for_empty_dir(self, tmp_path):
        assert load_latest_checkpoint(tmp_path) is None

    def test_returns_none_for_missing_dir(self, tmp_path):
        assert load_latest_checkpoint(tmp_path / "does_not_exist") is None

    def test_round_trips_adapter_state(self, tmp_path):
        save_checkpoint(tmp_path, rnd=1, adapter_state=_state(), history=[{"round": 1}])
        ckpt = load_latest_checkpoint(tmp_path)
        assert torch.equal(ckpt["adapter_state"]["lora_A"], torch.ones(2, 2))

    def test_picks_highest_round_not_alphabetical_order(self, tmp_path):
        # "checkpoint_round10" sorts before "checkpoint_round9" as a string.
        for rnd in (9, 10):
            save_checkpoint(
                tmp_path, rnd=rnd, adapter_state=_state(), history=[{"round": rnd}]
            )
        assert load_latest_checkpoint(tmp_path)["round"] == 10

    def test_restores_history_for_resume(self, tmp_path):
        history = [{"round": 1, "acc": 0.7}, {"round": 2, "acc": 0.8}]
        save_checkpoint(tmp_path, rnd=2, adapter_state=_state(), history=history)
        assert load_latest_checkpoint(tmp_path)["history"] == history
