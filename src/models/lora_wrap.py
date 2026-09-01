"""LoRA adapter wrapping for sequence-classification backbones.

Only the LoRA matrices (and the freshly initialized classification head) are
trainable; the backbone stays frozen. In the federated simulation exactly these
tensors travel between server and clients, so their byte size is the
per-client payload ``S`` used by the cost model ``C = R * K * r * s0``.

``r`` is a real argument everywhere, so M3 can sweep the rank dimension with
this same wrapper.
"""

from __future__ import annotations

import importlib
import logging
from typing import Mapping, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

#: Attention projections targeted for DistilBERT / RoBERTa style backbones.
DEFAULT_TARGET_MODULES: dict[str, list[str]] = {
    "distilbert": ["q_lin", "v_lin"],
    "roberta": ["query", "value"],
    "bert": ["query", "value"],
}


def _default_targets(model: nn.Module) -> list[str]:
    """Pick attention target modules from the backbone's model type."""
    model_type = getattr(model.config, "model_type", "")
    if model_type not in DEFAULT_TARGET_MODULES:
        raise ValueError(
            f"No default LoRA targets for model_type={model_type!r}; "
            f"pass target_modules explicitly."
        )
    return DEFAULT_TARGET_MODULES[model_type]


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
):
    """Attach LoRA adapters to an existing classification backbone.

    Args:
        model: A HuggingFace ``*ForSequenceClassification`` model.
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout applied inside the LoRA branch.
        target_modules: Module name suffixes to adapt; inferred from the
            backbone type when omitted.

    Returns:
        A ``peft.PeftModel`` whose only trainable tensors are the LoRA matrices
        and the classification head.

    Raises:
        ValueError: If ``r`` is not positive or the backbone type is unknown.
    """
    if r <= 0:
        raise ValueError(f"LoRA rank r must be > 0; got {r}")

    disable_incompatible_torchao_dispatch()

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules) if target_modules else _default_targets(model),
        bias="none",
    )
    return get_peft_model(model, config)


def build_lora_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] | None = None,
):
    """Load a pretrained backbone and wrap it with LoRA adapters.

    Args:
        model_name: HuggingFace model identifier.
        num_labels: Number of classification labels (SST-2 = 2).
        r: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout applied inside the LoRA branch.
        target_modules: Module name suffixes to adapt; inferred when omitted.

    Returns:
        A ``peft.PeftModel`` ready for federated local training.
    """
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model = apply_lora(base, r=r, alpha=alpha, dropout=dropout,
                       target_modules=target_modules)
    report = report_trainable(model)
    logger.info(
        "LoRA model %s (r=%d): trainable %d / %d params (%.3f%%), S=%.4f MB",
        model_name,
        r,
        report["trainable"],
        report["total"],
        report["pct"],
        adapter_size_mb(model),
    )
    return model


def report_trainable(model: nn.Module) -> dict[str, float]:
    """Count trainable versus total parameters.

    Args:
        model: Any torch module, typically a LoRA-wrapped model.

    Returns:
        Dict with ``total``, ``trainable`` and ``pct`` (trainable percentage).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "pct": 100.0 * trainable / total if total else 0.0,
    }


def get_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot the tensors that travel between client and server.

    Args:
        model: A LoRA-wrapped model.

    Returns:
        Mapping from parameter name to a detached CPU clone. Cloning matters:
        the returned state must not change when local training continues.
    """
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def set_adapter_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load an adapter state produced by :func:`get_adapter_state`.

    Args:
        model: A LoRA-wrapped model.
        state: Mapping from parameter name to tensor.

    Raises:
        KeyError: If ``state`` holds a name the model does not own.
    """
    params = dict(model.named_parameters())
    unknown = set(state) - set(params)
    if unknown:
        raise KeyError(f"Unknown adapter parameters: {sorted(unknown)}")
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(params[name].device))


def adapter_size_mb(model: nn.Module) -> float:
    """Measure the per-client uplink payload ``S`` in megabytes.

    Args:
        model: A LoRA-wrapped model.

    Returns:
        Size in MB (10^6 bytes) of all trainable tensors, i.e. what one client
        uploads per round.
    """
    total_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    return total_bytes / 1e6


_TORCHAO_DISPATCH_MODULE = "peft.tuners.lora.torchao"


def disable_incompatible_torchao_dispatch() -> bool:
    """Stop peft's torchao LoRA dispatcher from crashing on an old torchao.

    peft >= 0.19 ships a dispatcher for torchao-quantized layers whose
    availability check raises ``ImportError`` when torchao is installed but
    older than 0.16 — which is exactly the Kaggle image (torchao 0.10). The
    dispatcher runs for every adapted module, so ``get_peft_model`` dies before
    any of our code executes.

    Our backbones are never torchao-quantized, so the dispatcher's own
    "not available -> skip me" branch is the correct outcome. This replaces the
    raising check with one that reports False, and leaves a healthy install
    alone.

    Returns:
        True if the check was replaced, False if nothing needed doing.

    Raises:
        ImportError: If the check fails for a reason unrelated to the torchao
            version, which would be a real problem worth surfacing.
    """
    try:
        module = importlib.import_module(_TORCHAO_DISPATCH_MODULE)
    except ImportError:
        return False  # peft too old to have the dispatcher; nothing to disarm.

    checker = getattr(module, "is_torchao_available", None)
    if checker is None or module is None:
        return False

    try:
        checker()
    except ImportError as exc:
        if "torchao" not in str(exc):
            raise
        module.is_torchao_available = lambda: False
        logger.warning(
            "Disabled peft's torchao LoRA dispatcher (%s). Our models are not "
            "torchao-quantized, so this is safe.", exc,
        )
        return True
    return False
