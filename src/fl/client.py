"""Client-side local LoRA training and evaluation.

One model instance is shared across all simulated clients: before a client
trains, the server's adapter state is loaded into it, and afterwards only the
adapter tensors are read back out. Nothing else travels, and no per-client model
copy is ever materialised (that would blow up GPU memory at K=100).
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.lora_wrap import get_adapter_state
from src.utils.metrics import accuracy

logger = logging.getLogger(__name__)


def local_train(
    model: nn.Module,
    loader: DataLoader,
    lr: float = 2e-4,
    local_epochs: int = 1,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Train the adapters of one client on its local shard.

    Args:
        model: LoRA-wrapped model already holding the current global state.
        loader: DataLoader over this client's samples.
        lr: AdamW learning rate.
        local_epochs: Local passes over the client's data.
        device: Torch device string.

    Returns:
        The client's adapter state after training, as CPU tensors.

    Raises:
        ValueError: If the client has no samples (an empty DataLoader).
    """
    if len(loader) == 0:
        raise ValueError("Client loader is empty; every client needs >= 1 batch.")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    for _ in range(local_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

    return get_adapter_state(model)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Evaluate the global model on the held-out split.

    Args:
        model: LoRA-wrapped model holding the state to evaluate.
        loader: DataLoader over the evaluation split.
        device: Torch device string.

    Returns:
        Accuracy in [0, 1].
    """
    model.to(device)
    model.eval()

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        labels = batch["labels"]
        inputs = {
            k: v.to(device) for k, v in batch.items() if k in ("input_ids",
                                                               "attention_mask")
        }
        logits = model(**inputs).logits
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels.cpu())

    return accuracy(torch.cat(preds), torch.cat(targets))
