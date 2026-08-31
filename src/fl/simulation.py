"""Lightweight FedAvg simulation for LoRA fine-tuning (Route B).

Implemented directly on HuggingFace transformers + PEFT, informed by the
FederatedScope-LLM architecture (which stays a read-only reference under
``third_party/``). Clients are simulated sequentially on one device: a single
model instance is reused and only adapter states are swapped, so memory does
not grow with K.

Per round: select K clients -> each trains locally from the current global
adapter state -> states are averaged weighted by shard size -> the global model
is evaluated and checkpointed.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping, Sequence

import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.glue_loader import get_labels, load_sst2, subsample_indices
from src.fl.client import evaluate, local_train
from src.fl.dirichlet import dirichlet_split, summarize_split
from src.models.lora_wrap import (
    adapter_size_mb,
    build_lora_model,
    get_adapter_state,
    set_adapter_state,
)
from src.utils.checkpoint import load_latest_checkpoint, save_checkpoint
from src.utils.metrics import round_comm_mb

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducible runs."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device, falling back to CPU so debugging works anywhere."""
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def fedavg_aggregate(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Average client adapter states, weighted by local sample count.

    Args:
        states: One adapter state per participating client.
        weights: Number of local samples behind each state.

    Returns:
        The aggregated global adapter state (fresh CPU tensors).

    Raises:
        ValueError: If the inputs are empty, mismatched, or sum to zero weight.
        KeyError: If the client states do not share the same parameter names.
    """
    if not states:
        raise ValueError("fedavg_aggregate needs at least one client state.")
    if len(states) != len(weights):
        raise ValueError(
            f"Got {len(states)} states but {len(weights)} weights."
        )
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total client weight must be > 0.")

    keys = set(states[0])
    for state in states[1:]:
        if set(state) != keys:
            raise KeyError("Client adapter states have mismatched parameter names.")

    aggregated: dict[str, torch.Tensor] = {}
    for key in states[0]:
        stacked = torch.stack(
            [state[key].float() * (w / total) for state, w in zip(states, weights)]
        )
        aggregated[key] = stacked.sum(dim=0)
    return aggregated


def select_clients(num_clients: int, K: int, seed: int, rnd: int) -> list[int]:
    """Sample K distinct client ids for one round.

    The RNG is derived from ``seed`` and the round index, so a run is
    reproducible and resumable without replaying earlier rounds.

    Args:
        num_clients: Size of the client pool.
        K: Clients to select this round.
        seed: Experiment seed.
        rnd: Zero-based round index.

    Returns:
        Sorted list of K distinct client ids.

    Raises:
        ValueError: If K is not in ``[1, num_clients]``.
    """
    if K <= 0:
        raise ValueError(f"K must be > 0; got {K}")
    if K > num_clients:
        raise ValueError(f"Cannot select K={K} from {num_clients} clients.")
    rng = random.Random(seed + rnd)
    return sorted(rng.sample(range(num_clients), K))


def run_round(
    model: nn.Module,
    train_split: Dataset,
    partition: Sequence[Sequence[int]],
    selected_ids: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    lr: float,
    local_epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Run one federated round and return the new global adapter state.

    Args:
        model: Shared LoRA-wrapped model instance.
        train_split: Tokenized training split.
        partition: Client id -> dataset indices.
        selected_ids: Clients participating this round.
        global_state: Current global adapter state.
        lr: AdamW learning rate for local training.
        local_epochs: Local epochs per client.
        batch_size: Local batch size.
        device: Torch device string.

    Returns:
        Aggregated adapter state after this round.

    Raises:
        ValueError: If every selected client has an empty shard.
    """
    client_states: list[dict[str, torch.Tensor]] = []
    client_weights: list[int] = []

    for client_id in selected_ids:
        indices = list(partition[client_id])
        if not indices:
            logger.warning("Client %d has no samples this round; skipping.", client_id)
            continue
        # Every client starts from the same global state.
        set_adapter_state(model, global_state)
        loader = DataLoader(
            Subset(train_split, indices), batch_size=batch_size, shuffle=True
        )
        client_states.append(
            local_train(model, loader, lr=lr, local_epochs=local_epochs, device=device)
        )
        client_weights.append(len(indices))

    if not client_states:
        raise ValueError("No selected client had any samples this round.")

    return fedavg_aggregate(client_states, client_weights)


def build_partition(cfg, train_split: Dataset) -> list[list[int]]:
    """Build the non-IID client partition described by the config."""
    partition = dirichlet_split(
        get_labels(train_split),
        num_clients=cfg.num_clients,
        alpha=cfg.alpha_dirichlet,
        seed=cfg.seed,
        min_samples=cfg.get("min_samples_per_client", 1),
    )
    max_per_client = cfg.get("max_samples_per_client", None)
    if max_per_client:
        partition = [
            subsample_indices(part, max_per_client, seed=cfg.seed + i)
            for i, part in enumerate(partition)
        ]
    return partition


def run_federated(cfg, model: nn.Module | None = None,
                  datasets: Mapping[str, Dataset] | None = None) -> dict[str, Any]:
    """Run the full FedAvg baseline with a fixed client count K.

    Args:
        cfg: OmegaConf config (see ``configs/m2_baseline.yaml``).
        model: Pre-built LoRA model; built from ``cfg`` when omitted.
        datasets: Mapping with ``train`` and ``eval`` splits; SST-2 is loaded
            when omitted. Both hooks exist so unit tests stay offline.

    Returns:
        Dict with ``history`` (per-round records), ``adapter_size_mb`` (S),
        ``split_summary`` and ``final_acc``.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.get("device", "auto"))

    if datasets is None:
        datasets = load_sst2(cfg.model_name, max_len=cfg.max_len)
    if model is None:
        model = build_lora_model(
            cfg.model_name,
            num_labels=cfg.num_labels,
            r=cfg.r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )

    train_split, eval_split = datasets["train"], datasets["eval"]
    partition = build_partition(cfg, train_split)
    split_summary = summarize_split(partition, get_labels(train_split))
    eval_loader = DataLoader(
        eval_split, batch_size=cfg.get("eval_batch_size", cfg.batch_size)
    )

    S = adapter_size_mb(model)
    per_round_mb = round_comm_mb(cfg.K, S)

    checkpoint = load_latest_checkpoint(cfg.output_dir)
    if checkpoint:
        global_state = checkpoint["adapter_state"]
        history = checkpoint["history"]
        start_round = checkpoint["round"]
        set_adapter_state(model, global_state)
        logger.info("Resumed at round %d", start_round)
    else:
        global_state = get_adapter_state(model)
        history = []
        start_round = 0

    logger.info(
        "FedAvg: clients=%d, K=%d, R=%d, S=%.4f MB, device=%s",
        cfg.num_clients, cfg.K, cfg.R, S, device,
    )

    for rnd in range(start_round, cfg.R):
        selected = select_clients(cfg.num_clients, cfg.K, cfg.seed, rnd)
        global_state = run_round(
            model, train_split, partition, selected, global_state,
            lr=cfg.lr, local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size, device=device,
        )
        set_adapter_state(model, global_state)
        acc = evaluate(model, eval_loader, device=device)

        history.append(
            {
                "round": rnd + 1,
                "test_acc": acc,
                "K": cfg.K,
                "selected_clients": selected,
                "comm_mb": per_round_mb,
                "comm_mb_cumulative": per_round_mb * (rnd + 1),
            }
        )
        logger.info("Round %d/%d: acc=%.4f, comm=%.2f MB cumulative",
                    rnd + 1, cfg.R, acc, per_round_mb * (rnd + 1))
        save_checkpoint(cfg.output_dir, rnd + 1, global_state, history)

    return {
        "history": history,
        "adapter_size_mb": S,
        "split_summary": split_summary,
        "final_acc": history[-1]["test_acc"] if history else None,
    }
