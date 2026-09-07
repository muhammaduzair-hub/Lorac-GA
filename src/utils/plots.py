"""Publication-quality figures for the M3 accuracy surface A(K, r).

Two figures drive the research questions:

* :func:`plot_AK_curve` — accuracy vs client count K at a fixed rank (default
  r=8). This is the RQ1 figure: does A(K) rise then saturate on a real LLM?
* :func:`plot_AKr_heatmap` — the full K x r grid coloured by accuracy. This is
  the RQ2 setup figure: how does rank trade off against client count?

Both accept either a loaded surface dict (``{"records", "surface"}``), a bare
list of per-triple records, or a path to a surface JSON. Rendering uses plain
matplotlib (no seaborn) to keep the dependency set minimal — matplotlib is the
only plotting library the project pins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: works on Kaggle and in CI without a display

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
import matplotlib.ticker as mticker  # noqa: E402

from src.fl.profiler import aggregate_surface

Surface = Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path


def _as_surface_cells(surface: Surface) -> list[dict[str, Any]]:
    """Normalize any accepted input into a list of aggregated (K, r) cells."""
    if isinstance(surface, (str, Path)):
        payload = json.loads(Path(surface).read_text())
    else:
        payload = surface

    if isinstance(payload, Mapping):
        if "surface" in payload:
            return list(payload["surface"])
        if "records" in payload:
            return aggregate_surface(payload["records"])
        raise ValueError("Surface dict must contain 'surface' or 'records'.")

    # A bare list: could be aggregated cells or per-triple records.
    cells = list(payload)
    if cells and "acc_mean" not in cells[0]:
        return aggregate_surface(cells)
    return cells


def plot_AK_curve(
    surface: Surface,
    out_pdf: str | Path,
    r_fixed: int = 8,
    dpi: int = 300,
) -> Path:
    """Plot accuracy vs K at a fixed rank — the RQ1 saturation curve.

    Args:
        surface: Surface dict, record list, or path to a surface JSON.
        out_pdf: Output figure path (extension picks the format).
        r_fixed: Rank whose row is plotted (default 8).
        dpi: Raster resolution for non-vector formats.

    Returns:
        The path the figure was written to.

    Raises:
        ValueError: If no profiled cell has ``r == r_fixed``.
    """
    cells = [c for c in _as_surface_cells(surface) if int(c["r"]) == r_fixed]
    if not cells:
        raise ValueError(f"No profiled cells at r={r_fixed}.")
    cells.sort(key=lambda c: c["K"])

    Ks = [c["K"] for c in cells]
    means = [c["acc_mean"] for c in cells]
    stds = [c.get("acc_std", 0.0) for c in cells]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        Ks, means, yerr=stds, marker="o", capsize=4,
        linewidth=1.8, markersize=6, color="#1f77b4",
    )
    ax.set_xlabel("Number of clients per round  $K$")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"A(K) at fixed LoRA rank $r={r_fixed}$")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_AKr_heatmap(
    surface: Surface,
    out_pdf: str | Path,
    dpi: int = 300,
) -> Path:
    """Plot the K x r accuracy grid as an annotated heatmap — the RQ2 figure.

    Args:
        surface: Surface dict, record list, or path to a surface JSON.
        out_pdf: Output figure path.
        dpi: Raster resolution for non-vector formats.

    Returns:
        The path the figure was written to.

    Raises:
        ValueError: If the surface has no cells.
    """
    cells = _as_surface_cells(surface)
    if not cells:
        raise ValueError("Surface is empty; nothing to plot.")

    Ks = sorted({int(c["K"]) for c in cells})
    rs = sorted({int(c["r"]) for c in cells})
    lookup = {(int(c["K"]), int(c["r"])): c["acc_mean"] for c in cells}

    # rows = rank r (y), cols = client count K (x); NaN for un-profiled cells.
    grid = [[lookup.get((K, r), float("nan")) for K in Ks] for r in rs]

    fig, ax = plt.subplots(figsize=(1.3 * len(Ks) + 1.5, 1.0 * len(rs) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")

    ax.set_xticks(range(len(Ks)), labels=Ks)
    ax.set_yticks(range(len(rs)), labels=rs)
    ax.set_xlabel("Number of clients per round  $K$")
    ax.set_ylabel("LoRA rank  $r$")
    ax.set_title("Accuracy surface  $A(K, r)$")

    for i, r in enumerate(rs):
        for j, K in enumerate(Ks):
            value = lookup.get((K, r))
            if value is None:
                continue
            ax.text(
                j, i, f"{value:.3f}", ha="center", va="center",
                color="white" if value < 0.65 else "black", fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test accuracy")
    fig.tight_layout()

    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
