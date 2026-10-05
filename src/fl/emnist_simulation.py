"""FedAvg / FedProx simulation on EMNIST-Balanced for the M0 replication.

Same structure as ``src.fl.simulation.run_federated`` (one shared model, only
adapter state is swapped between clients) but for image batches. Communication is
logged *uplink-only* so cumulative cost equals the paper's ``C(K) = R * K * S``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.emnist_loader import get_labels, load_emnist
from src.fl.dirichlet import dirichlet_split, summarize_split
from src.fl.emnist_client import evaluate, local_train
from src.fl.simulation import fedavg_aggregate, resolve_device, select_clients, set_seed
from src.models.emnist_lora import build_lora_vit
from src.models.lora_wrap import adapter_size_mb, get_adapter_state, set_adapter_state
from src.utils.checkpoint import load_latest_checkpoint, save_checkpoint
from src.utils.metrics import round_comm_mb

logger = logging.getLogger(__name__)


def tail_mean(history: Sequence[Mapping[str, Any]], n: int) -> float:
    """Mean test accuracy over the last ``n`` rounds.

    Non-IID rounds swing with the chosen clients, so the tail mean is a steadier
    estimate of A(K) than the single final round (see ``src/fl/profiler.py``).

    Args:
        history: Per-round records with a ``test_acc`` field.
        n: Number of trailing rounds to average (clipped to the history length).

    Returns:
        Mean accuracy in [0, 1].

    Raises:
        ValueError: If the history is empty or ``n`` < 1.
    """
    if not history or n < 1:
        raise ValueError("tail_mean needs a non-empty history and n >= 1.")
    tail = history[-n:]
    return sum(h["test_acc"] for h in tail) / len(tail)


def build_model(cfg) -> nn.Module:
    """Build the frozen-backbone LoRA ViT described by the config."""
    return build_lora_vit(
        cfg.num_classes, cfg.r, cfg.lora_alpha,
        backbone_path=cfg.get("backbone_path"),
        targets=tuple(cfg.lora_targets), **cfg.vit,
    )


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
    prox_mu: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Run one federated round and return the new global adapter state.

    Args:
        model: Shared LoRA model instance.
        train_split: Training dataset of ``(image, label)`` samples.
        partition: Client id -> dataset indices.
        selected_ids: Clients participating this round.
        global_state: Current global adapter state.
        lr: Local AdamW learning rate.
        local_epochs: Local epochs per client.
        batch_size: Local batch size.
        device: Torch device string.
        prox_mu: FedProx coefficient (0 = FedAvg).

    Returns:
        Aggregated adapter state, weighted by client shard size.

    Raises:
        ValueError: If every selected client has an empty shard.
    """
    states, weights = [], []
    for client_id in selected_ids:
        indices = list(partition[client_id])
        if not indices:
            logger.warning("Client %d has no samples; skipping.", client_id)
            continue
        set_adapter_state(model, global_state)
        loader = DataLoader(Subset(train_split, indices), batch_size=batch_size,
                            shuffle=True)
        states.append(local_train(model, loader, lr=lr, local_epochs=local_epochs,
                                  device=device, prox_mu=prox_mu))
        weights.append(len(indices))
    if not states:
        raise ValueError("No selected client had any samples this round.")
    return fedavg_aggregate(states, weights)


def run_federated_emnist(cfg, model: nn.Module | None = None,
                         datasets: Mapping[str, Dataset] | None = None) -> dict[str, Any]:
    """Run federated training with a fixed client count ``cfg.K``.

    Args:
        cfg: OmegaConf config (see ``configs/m0_emnist.yaml``); ``cfg.prox_mu > 0``
            switches local training to FedProx.
        model: Pre-built model; built from ``cfg`` when omitted.
        datasets: ``train``/``eval`` datasets; EMNIST is loaded when omitted
            (the hook keeps unit tests offline).

    Returns:
        Dict with ``history`` (per-round records), ``adapter_size_mb`` (S),
        ``split_summary``, ``final_acc`` and ``tail_acc`` (mean of the last
        ``cfg.tail_rounds`` rounds).
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.get("device", "auto"))
    if datasets is None:
        datasets = load_emnist(cfg.data_root)
    if model is None:
        model = build_model(cfg)

    train_split, eval_split = datasets["train"], datasets["eval"]
    labels = get_labels(train_split)
    partition = dirichlet_split(labels, num_clients=cfg.num_clients,
                                alpha=cfg.alpha_dirichlet, seed=cfg.seed,
                                min_samples=cfg.get("min_samples_per_client", 1))
    split_summary = summarize_split(partition, labels)
    eval_loader = DataLoader(eval_split, batch_size=cfg.get("eval_batch_size", 512))

    S = adapter_size_mb(model)
    per_round_mb = round_comm_mb(cfg.K, S, bidirectional=False)  # paper: R*K*S
    prox_mu = cfg.get("prox_mu", 0.0)

    checkpoint = load_latest_checkpoint(cfg.output_dir)
    if checkpoint:
        global_state, history = checkpoint["adapter_state"], checkpoint["history"]
        start_round = checkpoint["round"]
        set_adapter_state(model, global_state)
        logger.info("Resumed at round %d", start_round)
    else:
        global_state, history, start_round = get_adapter_state(model), [], 0

    logger.info("EMNIST FL: clients=%d, K=%d, R=%d, S=%.4f MB, prox_mu=%g, device=%s",
                cfg.num_clients, cfg.K, cfg.R, S, prox_mu, device)

    for rnd in range(start_round, cfg.R):
        selected = select_clients(cfg.num_clients, cfg.K, cfg.seed, rnd)
        global_state = run_round(
            model, train_split, partition, selected, global_state,
            lr=cfg.lr, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size,
            device=device, prox_mu=prox_mu,
        )
        set_adapter_state(model, global_state)
        acc = evaluate(model, eval_loader, device=device)
        history.append({
            "round": rnd + 1, "test_acc": acc, "K": cfg.K,
            "selected_clients": selected, "comm_mb": per_round_mb,
            "comm_mb_cumulative": per_round_mb * (rnd + 1),
        })
        logger.info("Round %d/%d: acc=%.4f, comm=%.2f MB cumulative",
                    rnd + 1, cfg.R, acc, per_round_mb * (rnd + 1))
        save_checkpoint(cfg.output_dir, rnd + 1, global_state, history)

    return {
        "history": history,
        "adapter_size_mb": S,
        "split_summary": split_summary,
        "final_acc": history[-1]["test_acc"] if history else None,
        "tail_acc": tail_mean(history, cfg.get("tail_rounds", 3)) if history else None,
    }
