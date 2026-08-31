"""Per-round checkpointing so interrupted Kaggle runs can resume.

Each round writes ``checkpoint_round<N>.pt`` (adapter state + round number) and
overwrites ``history.json`` (plain JSON, small enough to commit to git).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

_ROUND_RE = re.compile(r"checkpoint_round(\d+)\.pt$")


def save_checkpoint(
    output_dir: str | Path,
    rnd: int,
    adapter_state: Mapping[str, torch.Tensor],
    history: list[dict[str, Any]],
) -> Path:
    """Persist one round's adapter state and the run history.

    Args:
        output_dir: Directory for this run's artifacts; created if missing.
        rnd: Round number just completed (1-based).
        adapter_state: Global adapter state after aggregation.
        history: Per-round metric records accumulated so far.

    Returns:
        Path of the written checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"checkpoint_round{rnd}.pt"
    torch.save({"round": rnd, "adapter_state": dict(adapter_state)}, ckpt_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    logger.info("Checkpoint saved: %s", ckpt_path)
    return ckpt_path


def load_latest_checkpoint(output_dir: str | Path) -> dict[str, Any] | None:
    """Load the highest-numbered checkpoint in a run directory.

    Args:
        output_dir: Directory previously passed to :func:`save_checkpoint`.

    Returns:
        Dict with ``round``, ``adapter_state`` and ``history``, or ``None`` when
        the directory holds no checkpoint (fresh run).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    rounds = [
        (int(m.group(1)), p)
        for p in output_dir.glob("checkpoint_round*.pt")
        if (m := _ROUND_RE.search(p.name))
    ]
    if not rounds:
        return None

    latest_round, latest_path = max(rounds, key=lambda item: item[0])
    payload = torch.load(latest_path, map_location="cpu")

    history_path = output_dir / "history.json"
    payload["history"] = (
        json.loads(history_path.read_text()) if history_path.exists() else []
    )
    logger.info("Resuming from %s (round %d)", latest_path, latest_round)
    return payload
