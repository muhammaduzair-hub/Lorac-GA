"""Two-variable accuracy-surface profiler A(K, r) for M3.

Runs short FedAvg simulations over a grid of client counts ``K`` and LoRA
ranks ``r`` and records the test accuracy of each cell. The resulting surface
is the empirical input to the joint-optimization GA in M4.

Design notes:

* **Tail-mean accuracy.** A cell's accuracy is the mean over its last
  ``TAIL_ROUNDS`` rounds, not the single final round. Federated accuracy keeps
  swinging after it converges because each round draws a different set of
  non-IID clients: a measured R=30 probe (K=10, r=8) settled at a mean of
  0.7743 over its converged rounds but with a std of 0.0649, and its round 30
  happened to land at 0.6376. Recording that one round would have written a
  value 0.14 below the truth, in a direction that changes cell to cell, and the
  effect of ``K`` would sit underneath that noise. Averaging the tail costs no
  extra compute and cut the standard error to 0.0177 on the same data.
* **Cost model caveat.** M4's ``C = R * K * r * s0`` treats ``s0 = S / r`` as a
  rank-independent constant. It is not: ``S`` also carries the classification
  head, which does not depend on ``r``, so ``s0`` drifts ~5.7x across r=2..16.
  Use the per-cell ``adapter_size_mb`` recorded here rather than rebuilding
  ``S`` from ``r * s0``.

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

from src.utils.env_info import collect_env

logger = logging.getLogger(__name__)

#: Rounds averaged into a cell's reported accuracy (see the module docstring).
TAIL_ROUNDS = 10

#: Fields every per-triple record in the surface JSON carries.
RECORD_FIELDS = ("K", "r", "seed", "R", "acc", "acc_final", "n_tail_rounds",
                 "comm_mb", "s0", "adapter_size_mb")


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
    """Write the ``{env, records, surface}`` document.

    The ``env`` block carries the package versions, CUDA device and git commit
    the surface was produced with. Kaggle resolves its own package versions and
    rebuilds its image regularly, so pinning them is not workable; recording
    what actually ran is what makes the surface reproducible later. It is
    rewritten with every cell, so a surface finished across several sessions
    reports the environment of the session that wrote the last cell.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "env": collect_env(),
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


def _reject_mismatched_horizon(records, cfg, path: Path) -> None:
    """Refuse to extend a surface whose cells were run at a different ``R``.

    Resume-skip matches on (K, r, seed) alone, so without this check a surface
    started at one round count and finished at another would silently mix the
    two: cells already present keep their old accuracy while new cells get the
    new horizon, and nothing in the file says which is which. Accuracy and
    ``comm_mb`` both move with ``R``, so the resulting surface -- and the GA
    trained on it -- would be comparing cells that were never comparable.

    Args:
        records: Records loaded from an existing surface JSON.
        cfg: The config this call is about to profile with.
        path: Where the surface JSON lives, quoted in the error.

    Raises:
        ValueError: If any existing record was produced at a different ``R``.
    """
    current = int(cfg.R)
    other = sorted({int(rec["R"]) for rec in records if "R" in rec} - {current})
    if not other:
        return
    raise ValueError(
        f"{path} holds cells profiled at R={other} but this run uses R={current}. "
        f"Accuracy and comm_mb both depend on R, so the cells are not comparable "
        f"and resume-skip would mix them silently. Move or delete that file to "
        f"start a fresh surface at R={current}, or set R back to {other[0]}."
    )


def tail_mean_acc(
    history: Sequence[Mapping[str, Any]],
    tail_rounds: int = TAIL_ROUNDS,
) -> tuple[float | None, int]:
    """Average the accuracy of a run's last rounds.

    Args:
        history: Per-round records from :func:`src.fl.simulation.run_federated`,
            each holding ``test_acc``.
        tail_rounds: How many trailing rounds to average. A run shorter than
            this averages everything it has.

    Returns:
        ``(mean_accuracy, rounds_averaged)``, or ``(None, 0)`` when no round
        carries a ``test_acc`` — which is the case for injected test doubles
        that only report ``final_acc``.
    """
    tail = [
        float(h["test_acc"])
        for h in list(history)[-tail_rounds:]
        if h.get("test_acc") is not None
    ]
    if not tail:
        return None, 0
    return statistics.fmean(tail), len(tail)


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
    client count and LoRA rank; its accuracy (the mean over the last
    ``TAIL_ROUNDS`` rounds -- see the module docstring for why the single final
    round is too noisy), total communication and measured per-unit-rank payload
    ``s0 = S / r`` are recorded. The single final round is kept alongside as
    ``acc_final`` so the smoothing stays auditable. The surface JSON is updated
    after every cell and existing triples are skipped, so the run is crash-safe
    and resumable.

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
    _reject_mismatched_horizon(records, cfg, path)
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
            history = result.get("history") or []
            comm_mb = float(history[-1]["comm_mb_cumulative"]) if history else 0.0

            acc_final = result["final_acc"]
            acc, n_tail = tail_mean_acc(history)
            if acc is None:  # history without test_acc: fall back to the run's own
                acc, n_tail = acc_final, 1

            record = {
                "K": int(K),
                "r": int(r),
                "seed": int(seed),
                "R": int(cfg.R),
                "acc": float(acc) if acc is not None else None,
                "acc_final": float(acc_final) if acc_final is not None else None,
                "n_tail_rounds": int(n_tail),
                "comm_mb": comm_mb,
                "s0": S / r,
                "adapter_size_mb": S,
            }
            records.append(record)
            done.add(triple)
            _save_surface(path, records)  # incremental: crash = nothing lost
            logger.info(
                "cell K=%d r=%d seed=%d -> acc=%.4f (mean of last %d rounds; "
                "final round %.4f), comm=%.1f MB, s0=%.4f",
                K, r, seed,
                record["acc"] if record["acc"] is not None else float("nan"),
                n_tail,
                record["acc_final"] if record["acc_final"] is not None else float("nan"),
                comm_mb, record["s0"],
            )

    return {"records": records, "surface": aggregate_surface(records)}
