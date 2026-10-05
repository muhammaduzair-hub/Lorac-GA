"""Local training and evaluation for the EMNIST (image) experiments.

Mirrors ``src/fl/client.py`` (HF/token batches) for ``(x, y)`` tensor batches and
adds the optional FedProx proximal term used by the paper's baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.models.lora_wrap import get_adapter_state
from src.utils.metrics import accuracy


def local_train(
    model: nn.Module,
    loader: DataLoader,
    lr: float = 1e-3,
    local_epochs: int = 1,
    device: str = "cpu",
    prox_mu: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Train the trainable parameters of one client on its local shard.

    Args:
        model: Model already holding the current global state.
        loader: DataLoader yielding ``(images, labels)``.
        lr: AdamW learning rate.
        local_epochs: Local passes over the client's data.
        device: Torch device string.
        prox_mu: FedProx coefficient mu; 0 gives plain FedAvg local training.

    Returns:
        The client's trainable-parameter state after training, as CPU tensors.

    Raises:
        ValueError: If the loader is empty.
    """
    if len(loader) == 0:
        raise ValueError("Client loader is empty; every client needs >= 1 batch.")

    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    ref = [p.detach().clone() for p in params] if prox_mu > 0 else None
    optimizer = torch.optim.AdamW(params, lr=lr)

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if ref is not None:
                loss = loss + prox_mu / 2 * sum(
                    ((p - r) ** 2).sum() for p, r in zip(params, ref)
                )
            loss.backward()
            optimizer.step()

    return get_adapter_state(model)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Evaluate the model on a held-out split.

    Args:
        model: Model holding the state to evaluate.
        loader: DataLoader yielding ``(images, labels)``.
        device: Torch device string.

    Returns:
        Accuracy in [0, 1].
    """
    model.to(device).eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=-1).cpu())
        targets.append(y)
    return accuracy(torch.cat(preds), torch.cat(targets))
