"""GLUE SST-2 loading and tokenization for federated LoRA experiments.

The loader returns a `DatasetDict` with two splits:

* ``train`` — the SST-2 training split, later partitioned across clients.
* ``eval``  — the SST-2 *validation* split. The official ``test`` split ships
  with label ``-1`` (labels are held out on the GLUE server), so it cannot be
  used to measure accuracy; validation is the standard stand-in.
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_COLUMNS = ["input_ids", "attention_mask", "labels"]


def load_sst2(
    tokenizer_name: str = "distilbert-base-uncased",
    max_len: int = 128,
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load and tokenize GLUE SST-2.

    Args:
        tokenizer_name: HuggingFace tokenizer / model identifier.
        max_len: Maximum sequence length; sequences are padded and truncated
            to exactly this length so batches need no dynamic collator.
        cache_dir: Optional HuggingFace cache directory (useful on Kaggle where
            the default cache is not persisted between sessions).

    Returns:
        DatasetDict with ``train`` and ``eval`` splits, torch-formatted and
        holding only ``input_ids``, ``attention_mask`` and ``labels``.
    """
    raw = load_dataset("glue", "sst2", cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    splits = DatasetDict({"train": raw["train"], "eval": raw["validation"]})
    splits = splits.map(_tokenize, batched=True)
    splits = splits.rename_column("label", "labels")
    splits = splits.remove_columns(
        [c for c in splits["train"].column_names if c not in MODEL_COLUMNS]
    )
    splits.set_format(type="torch", columns=MODEL_COLUMNS)

    logger.info(
        "SST-2 loaded: train=%d, eval=%d, max_len=%d",
        len(splits["train"]),
        len(splits["eval"]),
        max_len,
    )
    return splits


def get_labels(split: Dataset) -> np.ndarray:
    """Extract the integer label array of a split.

    Args:
        split: A tokenized SST-2 split holding a ``labels`` column.

    Returns:
        1-D numpy array of labels, used by the Dirichlet partitioner.
    """
    return np.asarray(split["labels"])


def subsample_indices(
    indices: Sequence[int],
    max_samples: int | None,
    seed: int = 42,
) -> list[int]:
    """Cap a client's sample count for cheap runs.

    Args:
        indices: Dataset indices owned by one client.
        max_samples: Maximum samples to keep; ``None`` keeps everything.
        seed: RNG seed, so every run picks the same subset.

    Returns:
        Sorted list of at most ``max_samples`` indices, without duplicates.
    """
    indices = list(indices)
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = random.Random(seed)
    return sorted(rng.sample(indices, max_samples))
