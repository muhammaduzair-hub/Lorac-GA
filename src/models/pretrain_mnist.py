"""Centrally pretrain the ViT backbone on MNIST, then save it (head excluded).

Usage (Kaggle / local):  python -m src.models.pretrain_mnist --config configs/m0_emnist.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.emnist_loader import load_mnist
from src.fl.emnist_client import evaluate, local_train
from src.fl.simulation import resolve_device, set_seed
from src.models.emnist_lora import TinyViT, save_backbone

logger = logging.getLogger(__name__)


def pretrain(cfg) -> float:
    """Pretrain on MNIST and save the backbone to ``cfg.backbone_path``.

    Args:
        cfg: OmegaConf config (see ``configs/m0_emnist.yaml``).

    Returns:
        MNIST test accuracy of the pretrained model.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.get("device", "auto"))
    data = load_mnist(cfg.data_root)
    model = TinyViT(num_classes=10, **cfg.vit)

    pre = cfg.pretrain
    loader = DataLoader(data["train"], batch_size=pre.batch_size, shuffle=True)
    for ep in range(pre.epochs):
        local_train(model, loader, lr=pre.lr, local_epochs=1, device=device)
        acc = evaluate(model, DataLoader(data["eval"], batch_size=512), device=device)
        logger.info("pretrain epoch %d/%d: MNIST acc=%.4f", ep + 1, pre.epochs, acc)

    os.makedirs(os.path.dirname(cfg.backbone_path) or ".", exist_ok=True)
    save_backbone(model, cfg.backbone_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "pretrain.json"), "w") as f:
        json.dump({"mnist_acc": acc, "epochs": pre.epochs, "seed": cfg.seed}, f)
    return acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m0_emnist.yaml")
    pretrain(OmegaConf.load(ap.parse_args().config))
