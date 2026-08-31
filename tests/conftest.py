"""Shared fixtures: a tiny CPU-only LoRA model and a tiny tokenized dataset.

Nothing here downloads weights or data, so the whole suite runs offline.
"""

import pytest
import torch
from datasets import Dataset
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.models.lora_wrap import apply_lora

SEQ_LEN = 8
VOCAB = 100


def make_tiny_base(num_labels: int = 2) -> DistilBertForSequenceClassification:
    """Build a randomly initialized, very small DistilBERT classifier."""
    config = DistilBertConfig(
        vocab_size=VOCAB,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(config)


def make_tiny_dataset(n: int = 32, seed: int = 0) -> Dataset:
    """Build a tokenized dataset with the same columns as the SST-2 loader."""
    generator = torch.Generator().manual_seed(seed)
    data = {
        "input_ids": torch.randint(0, VOCAB, (n, SEQ_LEN), generator=generator).tolist(),
        "attention_mask": torch.ones(n, SEQ_LEN, dtype=torch.long).tolist(),
        "labels": torch.randint(0, 2, (n,), generator=generator).tolist(),
    }
    ds = Dataset.from_dict(data)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return apply_lora(make_tiny_base(), r=4, alpha=8, dropout=0.0)


@pytest.fixture
def tiny_dataset():
    return make_tiny_dataset()


@pytest.fixture
def dataset_factory():
    """Hand tests the dataset builder itself, for custom sizes and seeds.

    Exposed as a fixture rather than imported across test modules: `tests/` is
    not a package, and `from tests.conftest import ...` only resolves under some
    pytest path-resolution modes (it fails on the Kaggle image).
    """
    return make_tiny_dataset
