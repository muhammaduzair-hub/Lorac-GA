"""Capture the runtime environment of an experiment.

Kaggle rebuilds its image regularly and resolves its own package versions, so
pinning exact versions there is not workable (forcing them downgrades numpy and
breaks the image). Instead every run records what it actually ran with, and that
record is stored next to the results.
"""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata

TRACKED_PACKAGES = ("torch", "numpy", "transformers", "peft", "datasets")


def _version_of(package: str) -> str:
    """Return an installed package's version string.

    Args:
        package: Distribution name.

    Returns:
        Version string.

    Raises:
        importlib.metadata.PackageNotFoundError: If the package is absent.
    """
    return metadata.version(package)


def _git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_env() -> dict[str, str]:
    """Describe the current run environment.

    Returns:
        Dict of version strings plus ``cuda_available``, ``device_name``,
        ``platform`` and ``git_commit``. A package that is not installed is
        reported as ``"not installed"`` rather than raising, so logging an
        experiment never breaks the experiment.
    """
    env: dict[str, str] = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            env[package] = _version_of(package)
        except Exception:
            env[package] = "not installed"

    cuda_available, device_name = "False", "cpu"
    try:
        import torch

        cuda_available = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    env["cuda_available"] = cuda_available
    env["device_name"] = device_name
    env["platform"] = platform.platform()
    env["git_commit"] = _git_commit()
    return env
