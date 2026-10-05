"""Tiny ViT with LoRA for the M0 EMNIST replication (Solat & Lee 2025).

The paper only says "compact transformer-based classifier fine-tuned using
LoRA"; the architecture below is *our* assumption. The backbone is pretrained
centrally on MNIST, then frozen. Federated clients train only the LoRA matrices
on the attention ``q``/``v`` projections plus a fresh classification head, and
exactly those tensors form the per-client payload ``S``.

Adapter-state helpers (``get_adapter_state`` etc.) are reused from
``src.models.lora_wrap`` — they act on any module's ``requires_grad`` params.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.lora_wrap import adapter_size_mb

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank update ``(alpha/r) * B A x``.

    ``B`` starts at zero, so a freshly wrapped layer is exactly the base layer.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class Block(nn.Module):
    """Pre-norm transformer block with separate q/k/v/proj linears."""

    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.proj = (nn.Linear(dim, dim) for _ in range(4))
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        B, N, D = h.shape
        q, k, v = (f(h).view(B, N, self.heads, D // self.heads).transpose(1, 2)
                   for f in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    """Small ViT for 28x28 grayscale images (CLS-token classification)."""

    def __init__(self, num_classes: int, dim: int = 128, depth: int = 4,
                 heads: int = 4, mlp_dim: int = 256, patch: int = 4,
                 img_size: int = 28) -> None:
        super().__init__()
        n_tokens = (img_size // patch) ** 2 + 1
        self.patch_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


def save_backbone(model: TinyViT, path: str) -> None:
    """Save every weight except the classification head.

    Args:
        model: A (pretrained) plain ``TinyViT``.
        path: Output ``.pt`` file.
    """
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}
    torch.save(state, path)


def build_lora_vit(
    num_classes: int,
    r: int,
    alpha: float,
    backbone_path: str | None = None,
    targets: Sequence[str] = ("q", "v"),
    **vit_kwargs,
) -> TinyViT:
    """Build the federated model: frozen backbone + LoRA on ``targets`` + new head.

    Args:
        num_classes: Output classes (47 for EMNIST-Balanced).
        r: LoRA rank.
        alpha: LoRA scaling numerator (scaling = alpha / r).
        backbone_path: Pretrained backbone from :func:`save_backbone`; random
            init when ``None`` (only sensible for tests).
        targets: Attention projections to wrap with LoRA.
        **vit_kwargs: Forwarded to :class:`TinyViT` (must match the backbone).

    Returns:
        Model whose only trainable params are the LoRA matrices and the head.

    Raises:
        RuntimeError: If the backbone file does not match the architecture.
    """
    model = TinyViT(num_classes, **vit_kwargs)
    if backbone_path is not None:
        missing, unexpected = model.load_state_dict(
            torch.load(backbone_path, map_location="cpu"), strict=False
        )
        if unexpected or any(not k.startswith("head.") for k in missing):
            raise RuntimeError(f"Backbone mismatch: missing={missing}, "
                               f"unexpected={unexpected}")
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.blocks:
        for name in targets:
            setattr(blk, name, LoRALinear(getattr(blk, name), r, alpha))
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def closest_rank(target_mb: float, num_classes: int, alpha_ratio: float = 2.0,
                 r_max: int = 32, **kwargs) -> tuple[int, float]:
    """Pick the LoRA rank whose payload ``S`` is closest to ``target_mb``.

    Args:
        target_mb: Desired per-client payload (paper: 0.0833 MB).
        num_classes: Output classes.
        alpha_ratio: ``alpha = alpha_ratio * r``.
        r_max: Largest rank to try.
        **kwargs: Forwarded to :func:`build_lora_vit`.

    Returns:
        ``(rank, payload_mb)``.
    """
    sizes = {r: adapter_size_mb(build_lora_vit(num_classes, r, alpha_ratio * r, **kwargs))
             for r in range(1, r_max + 1)}
    best = min(sizes, key=lambda r: abs(sizes[r] - target_mb))
    return best, sizes[best]
