"""Unit tests for the SST-2 GLUE loader.

No network access: `datasets.load_dataset` and the tokenizer are patched with
tiny local fakes so the real transformation code path still runs.
"""

import pytest

from datasets import Dataset, DatasetDict

from src.data import glue_loader


class _FakeTokenizer:
    """Minimal stand-in for a HuggingFace fast tokenizer."""

    def __call__(self, texts, truncation=True, padding="max_length", max_length=8):
        return {
            "input_ids": [[1] * max_length for _ in texts],
            "attention_mask": [[1] * max_length for _ in texts],
        }


def _fake_sst2() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "sentence": ["good movie", "bad movie", "ok movie", "great film"],
                    "label": [1, 0, 1, 1],
                    "idx": [0, 1, 2, 3],
                }
            ),
            "validation": Dataset.from_dict(
                {"sentence": ["nice", "awful"], "label": [1, 0], "idx": [0, 1]}
            ),
            "test": Dataset.from_dict(
                {"sentence": ["unlabeled"], "label": [-1], "idx": [0]}
            ),
        }
    )


@pytest.fixture
def patched_loader(monkeypatch):
    monkeypatch.setattr(glue_loader, "load_dataset", lambda *a, **kw: _fake_sst2())
    monkeypatch.setattr(
        glue_loader.AutoTokenizer, "from_pretrained", lambda *a, **kw: _FakeTokenizer()
    )


class TestLoadSst2:
    def test_returns_train_and_eval_splits(self, patched_loader):
        ds = glue_loader.load_sst2(max_len=8)
        assert set(ds.keys()) == {"train", "eval"}

    def test_eval_split_comes_from_validation_not_unlabeled_test(self, patched_loader):
        # GLUE SST-2 `test` labels are all -1, so it is unusable for accuracy.
        ds = glue_loader.load_sst2(max_len=8)
        assert len(ds["eval"]) == 2
        assert -1 not in ds["eval"]["labels"].tolist()

    def test_columns_are_model_ready(self, patched_loader):
        ds = glue_loader.load_sst2(max_len=8)
        assert set(ds["train"].column_names) == {"input_ids", "attention_mask", "labels"}

    def test_sequence_length_matches_max_len(self, patched_loader):
        ds = glue_loader.load_sst2(max_len=8)
        assert ds["train"][0]["input_ids"].shape[0] == 8


class TestGetLabels:
    def test_returns_label_array_for_split(self, patched_loader):
        ds = glue_loader.load_sst2(max_len=8)
        labels = glue_loader.get_labels(ds["train"])
        assert labels.tolist() == [1, 0, 1, 1]


class TestSubsampleIndices:
    def test_caps_at_max_samples(self):
        out = glue_loader.subsample_indices(list(range(50)), max_samples=10, seed=42)
        assert len(out) == 10

    def test_returns_all_when_max_samples_is_none(self):
        idx = list(range(7))
        assert glue_loader.subsample_indices(idx, max_samples=None, seed=42) == idx

    def test_returns_all_when_fewer_than_max(self):
        idx = list(range(3))
        assert glue_loader.subsample_indices(idx, max_samples=10, seed=42) == idx

    def test_is_a_subset_without_duplicates(self):
        out = glue_loader.subsample_indices(list(range(50)), max_samples=10, seed=42)
        assert len(set(out)) == len(out)
        assert set(out).issubset(set(range(50)))

    def test_same_seed_same_subset(self):
        a = glue_loader.subsample_indices(list(range(50)), max_samples=10, seed=42)
        b = glue_loader.subsample_indices(list(range(50)), max_samples=10, seed=42)
        assert a == b

    def test_different_seed_different_subset(self):
        a = glue_loader.subsample_indices(list(range(200)), max_samples=10, seed=42)
        b = glue_loader.subsample_indices(list(range(200)), max_samples=10, seed=7)
        assert a != b
