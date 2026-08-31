"""CLI entrypoint for the M2 federated baseline.

Usage:
    python -m src.fl.server --config configs/m2_baseline.yaml K=10 R=10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from src.fl.simulation import run_federated

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when omitted.

    Returns:
        Namespace with ``config`` (path) and ``overrides`` (``key=value`` list).
    """
    parser = argparse.ArgumentParser(description="Run the M2 FedAvg LoRA baseline.")
    parser.add_argument("--config", default="configs/m2_baseline.yaml",
                        help="Path to the YAML experiment config.")
    parser.add_argument("overrides", nargs="*", default=[],
                        help="Dotlist overrides, e.g. K=5 R=2.")
    return parser.parse_args(argv)


def build_config(config_path: str | Path,
                 overrides: Sequence[str] | None = None) -> DictConfig:
    """Load a YAML config and apply ``key=value`` CLI overrides.

    Args:
        config_path: Path to the YAML config.
        overrides: Dotlist overrides applied on top of the file.

    Returns:
        The merged config.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If an override names a key the config does not define.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if overrides:
        extra = OmegaConf.from_dotlist(list(overrides))
        unknown = set(extra.keys()) - set(cfg.keys())
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        cfg = OmegaConf.merge(cfg, extra)
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline and write ``results.json`` into the run directory."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    cfg = build_config(args.config, args.overrides)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    result = run_federated(cfg)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": OmegaConf.to_container(cfg, resolve=True), **result}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    logger.info("Final accuracy: %.4f", result["final_acc"])


if __name__ == "__main__":
    main()
