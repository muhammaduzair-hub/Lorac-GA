"""Two-variable accuracy-surface profiler A(K, r) for M3.

Runs short FedAvg simulations over a grid of client counts ``K`` and LoRA
ranks ``r`` and records the final test accuracy of each cell. The resulting
surface is the empirical input to the joint-optimization GA in M4, whose cost
model is ``C = R * K * r * s0`` with ``s0`` the per-unit-rank uplink payload.

Design notes:

* **Isolation.** Each (K, r, seed) cell trains in its own ``output_dir``
  (``<base>/cells/K{K}_r{r}_s{seed}``). :func:`run_federated` resumes from any
  checkpoint it finds in ``output_dir``; without a per-cell directory a later
  cell would resume a *different* cell's adapter state. Isolation also makes the
  incremental JSON safe to interrupt: a crashed cell leaves the surface JSON
  intact and simply reruns.
* **Incremental + resumable.** The surface JSON is rewritten after every cell,
  and cells already present (matched on the (K, r, seed) triple) are skipped, so
  a run whose Kaggle quota runs out mid-grid loses nothing and resumes cleanly.
* **Priority ordering.** ``priority_cells`` runs the RQ1 curve (r=8) and RQ2
  frontier first, so partial quota still yields the figures that matter.
* **Testability.** The heavy :func:`run_federated` is injected via ``run_fn``
  and imported lazily only when needed, so the orchestration logic (ordering,
  resume, aggregation, schema) is unit-tested offline without torch.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

#: Fields every per-triple record in the surface JSON carries.
RECORD_FIELDS = ("K", "r", "seed", "acc", "comm_mb", "s0", "adapter_size_mb")


def _surface_path(cfg, surface_path: str | Path | None) -> Path:
    """Resolve where the surface JSON lives (defaults under ``cfg.output_dir``)."""
    if surface_path is not None:
        return Path(surface_path)
    return Path(cfg.output_dir) / "A_Kr_surface.json"


def load_surface(surface_path: str | Path) -> list[dict[str, Any]]:
    """Load the per-triple records from a surface JSON file.

    Args:
        surface_path: Path to a surface JSON written by :func:`profile_AKr`.

    Returns:
        The list of per-triple records, or an empty list when the file is
        missing (a fresh profiling run).
    """
    path = Path(surface_path)
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    # Accept either the full {"records", "surface"} document or a bare list.
    if isinstance(payload, dict):
        return list(payload.get("records", []))
    return list(payload)


def aggregate_surface(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-triple records into one entry per (K, r) cell.

    Args:
        records: Per-triple records (each with ``K``, ``r``, ``seed``, ``acc``,
            ``comm_mb``, ``s0``).

    Returns:
        One dict per (K, r) with mean/std accuracy across seeds, sorted by
        (K, r). ``acc_std`` is 0.0 for a single seed.
    """
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[(int(rec["K"]), int(rec["r"]))].append(rec)

    surface: list[dict[str, Any]] = []
    for (K, r), recs in sorted(groups.items()):
        accs = [float(x["acc"]) for x in recs]
        surface.append(
            {
                "K": K,
                "r": r,
                "acc_mean": statistics.fmean(accs),
                "acc_std": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "n_seeds": len(recs),
                "seeds": sorted(int(x["seed"]) for x in recs),
                "comm_mb": statistics.fmean(float(x["comm_mb"]) for x in recs),
                "s0": statistics.fmean(float(x["s0"]) for x in recs),
            }
        )
    return surface


def _save_surface(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write the ``{records, surface}`` document atomically-ish and small."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "records": list(records),
        "surface": aggregate_surface(records),
    }
    path.write_text(json.dumps(document, indent=2))


def order_cells(
    K_values: Sequence[int],
    r_values: Sequence[int],
    priority_cells: Sequence[Sequence[int]] | None = None,
    cell_subset: Sequence[Sequence[int]] | None = None,
) -> list[tuple[int, int]]:
    """Compute the (K, r) run order for one profiling call.

    Args:
        K_values: Client counts to sweep.
        r_values: LoRA ranks to sweep.
        priority_cells: Ordered (K, r) pairs to run first (the RQ1 curve and
            RQ2 frontier); ignored entries not in the grid.
        cell_subset: When given, restrict the grid to exactly these (K, r)
            pairs — used to split the surface across weeks/quota windows.

    Returns:
        The (K, r) cells in the order they should run, priority first.
    """
    grid = [(int(K), int(r)) for K in K_values for r in r_values]
    if cell_subset is not None:
        subset = {(int(K), int(r)) for K, r in cell_subset}
        grid = [cell for cell in grid if cell in subset]

    grid_set = set(grid)
    if not priority_cells:
        return grid

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for K, r in priority_cells:
        cell = (int(K), int(r))
        if cell in grid_set and cell not in seen:
            ordered.append(cell)
            seen.add(cell)
    ordered.extend(cell for cell in grid if cell not in seen)
    return ordered


def _cell_cfg(cfg, K: int, r: int, seed: int, base_dir: Path):
    """Clone ``cfg`` with this cell's K, r, seed and an isolated output_dir."""
    cell = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cell.K = K
    cell.r = r
    cell.seed = seed
    cell.output_dir = str(base_dir / "cells" / f"K{K}_r{r}_s{seed}")
    return cell


def profile_AKr(
    cfg,
    K_values: Sequence[int],
    r_values: Sequence[int],
    seeds: Sequence[int],
    priority_cells: Sequence[Sequence[int]] | None = None,
    cell_subset: Sequence[Sequence[int]] | None = None,
    datasets: Mapping[str, Any] | None = None,
    surface_path: str | Path | None = None,
    run_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Profile the accuracy surface A(K, r) over a grid of cells.

    For each (K, r, seed) triple a short federated run is executed at that
    client count and LoRA rank; the final test accuracy, total communication
    and measured per-unit-rank payload ``s0 = S / r`` are recorded. The surface
    JSON is updated after every cell and existing triples are skipped, so the
    run is crash-safe and resumable.

    Args:
        cfg: Base OmegaConf config (see ``configs/m3_profile.yaml``). ``K``,
            ``r`` and ``seed`` are overridden per cell; ``output_dir`` is the
            base under which per-cell directories and the surface JSON live.
        K_values: Client counts to sweep.
        r_values: LoRA ranks to sweep.
        seeds: Seeds to run for every cell in this call. A second seed for
            frontier cells is added by calling again with the frontier as
            ``cell_subset`` — resume-skip merges the result.
        priority_cells: Ordered (K, r) pairs to run first.
        cell_subset: Restrict this call to exactly these (K, r) pairs.
        datasets: Pre-loaded ``{"train", "eval"}`` splits, forwarded to
            ``run_fn`` so the data is tokenized once for the whole grid.
        surface_path: Override for the surface JSON path.
        run_fn: Federated runner; defaults to
            :func:`src.fl.simulation.run_federated` (imported lazily).

    Returns:
        Dict with ``records`` (per-triple) and ``surface`` (aggregated per
        (K, r)), matching the on-disk JSON.
    """
    if run_fn is None:
        from src.fl.simulation import run_federated as run_fn  # lazy: needs torch

    base_dir = Path(cfg.output_dir)
    path = _surface_path(cfg, surface_path)

    records = load_surface(path)
    done = {(int(r["K"]), int(r["r"]), int(r["seed"])) for r in records}

    cells = order_cells(K_values, r_values, priority_cells, cell_subset)
    total = len(cells) * len(seeds)
    logger.info(
        "M3 profiling: %d cells x %d seed(s) = %d runs (%d already done)",
        len(cells), len(seeds), total, len(done),
    )

    for K, r in cells:
        for seed in seeds:
            triple = (int(K), int(r), int(seed))
            if triple in done:
                logger.info("skip K=%d r=%d seed=%d (already profiled)", K, r, seed)
                continue

            cell = _cell_cfg(cfg, K, r, seed, base_dir)
            result = run_fn(cell, model=None, datasets=datasets)

            S = float(result["adapter_size_mb"])
            acc = result["final_acc"]
            history = result.get("history") or []
            comm_mb = float(history[-1]["comm_mb_cumulative"]) if history else 0.0

            record = {
                "K": int(K),
                "r": int(r),
                "seed": int(seed),
                "acc": float(acc) if acc is not None else None,
                "comm_mb": comm_mb,
                "s0": S / r,
                "adapter_size_mb": S,
            }
            records.append(record)
            done.add(triple)
            _save_surface(path, records)  # incremental: crash = nothing lost
            logger.info(
                "cell K=%d r=%d seed=%d -> acc=%.4f, comm=%.1f MB, s0=%.4f",
                K, r, seed, record["acc"] if record["acc"] is not None else float("nan"),
                comm_mb, record["s0"],
            )

    return {"records": records, "surface": aggregate_surface(records)}
