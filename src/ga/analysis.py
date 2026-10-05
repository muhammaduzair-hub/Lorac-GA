"""M0 analysis: run the GA on a profiled A(K) and compare with Solat & Lee (2025).

Produces the Table 3 analogue (K*, accuracy, cost per budget B), a GA-vs-exhaustive
check, convergence curves for GA / greedy / random (Fig. 3 analogue), saturation
and communication-saving statistics, and a markdown comparison with the paper.

Usage:
    python -m src.ga.analysis --surface results/m0_emnist/A_Kr_surface.json \
        --r 7 --out results/m0_emnist/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

from src.ga.fitness import interpolate_accuracy, make_fitness, max_feasible_K
from src.ga.search import SearchResult, ga_search, greedy_search, random_search

logger = logging.getLogger(__name__)

#: Table 3 of Solat & Lee (2025): B -> (K*, accuracy %, comm cost MB).
PAPER_TABLE3: dict[int, tuple[int, float, float]] = {
    10: (2, 85.1, 1.67), 50: (4, 91.7, 3.33), 100: (5, 93.1, 4.17),
    400: (5, 93.1, 4.17), 1000: (12, 96.6, 10.00), 4000: (10, 96.6, 8.33),
}


def load_profile(surface_json: str | Path, r: int) -> tuple[dict[int, float], float]:
    """Read A(K) at one rank from a profiler surface JSON.

    Args:
        surface_json: Path written by ``src.fl.profiler.profile_AKr``.
        r: LoRA rank whose cells form the K-only profile.

    Returns:
        ``(profile, S)`` where ``profile`` maps K to mean accuracy and ``S`` is
        the measured per-client payload in MB.

    Raises:
        ValueError: If the file has no cells at ``r``.
    """
    doc = json.loads(Path(surface_json).read_text())
    profile = {int(c["K"]): float(c["acc_mean"]) for c in doc["surface"]
               if int(c["r"]) == r}
    sizes = [float(x["adapter_size_mb"]) for x in doc["records"] if int(x["r"]) == r]
    if not profile or not sizes:
        raise ValueError(f"No profiled cells at r={r} in {surface_json}.")
    return profile, sizes[0]


def saturation_K(profile: Mapping[int, float], frac: float = 0.99) -> int:
    """Smallest K whose interpolated accuracy reaches ``frac`` of the maximum.

    Args:
        profile: Profiled accuracy keyed by K.
        frac: Fraction of the peak accuracy counted as saturated.

    Returns:
        The saturation point on the integer grid ``1..max(profile)``.
    """
    A = interpolate_accuracy(profile)
    ks = range(1, max(profile) + 1)
    peak = max(A(k) for k in ks)
    return next(k for k in ks if A(k) >= frac * peak)


def budget_row(profile: Mapping[int, float], B: float, R: int, S: float, K_max: int,
               ga_kwargs: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    """Run the GA for one budget and collect the table-3 style statistics.

    Args:
        profile: Profiled accuracy keyed by K.
        B: Bandwidth budget in MB.
        R: Communication rounds.
        S: Per-client payload in MB.
        K_max: Total clients.
        ga_kwargs: GA settings (population_size, generations, p_c, p_m, ...).
        seeds: GA seeds; the first gives the reported K*, all feed the success rate.

    Returns:
        Row dict; ``K_star`` is ``None`` when even K=1 exceeds the budget.
    """
    cap = max_feasible_K(B, R, S, K_max)
    row: dict[str, Any] = {"B": B, "K_cap": cap, "K_star": None}
    if cap < 1:
        return row

    A = interpolate_accuracy(profile)
    f = make_fitness(profile, B, R, S)
    best = max(range(1, cap + 1), key=lambda k: (f(k), -k))  # exhaustive optimum
    runs = [ga_search(f, 1, cap, seed=s, **ga_kwargs) for s in seeds]
    res = runs[0]
    row.update(
        K_star=res.K_star, acc=A(res.K_star), cost=R * res.K_star * S,
        fitness=res.fitness, K_exhaustive=best,
        ga_matches_exhaustive=res.K_star == best,
        ga_success_rate=sum(r.K_star == best for r in runs) / len(runs),
        min_term_inactive=f(res.K_star) == A(res.K_star),
        K_spend_all=cap, acc_spend_all=A(cap), cost_spend_all=R * cap * S,
        saving_pct=100 * (1 - res.K_star / cap),
        acc_gap=A(res.K_star) - A(cap),
    )
    return row


def convergence_curves(fitness_fn, K_max: int, seeds: Sequence[int], iterations: int,
                       ga_kwargs: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    """Best-so-far fitness per iteration for GA, greedy and random (Fig. 3 analogue).

    Args:
        fitness_fn: Callable K -> fitness.
        K_max: Largest feasible K.
        seeds: One run per seed and method.
        iterations: Iterations per run (GA uses this as its generation count).
        ga_kwargs: GA settings; ``generations`` is overridden by ``iterations``.

    Returns:
        ``{method: [history per seed]}``.
    """
    kw = {**ga_kwargs, "generations": iterations}
    out: dict[str, list[list[float]]] = {"GA": [], "greedy": [], "random": []}
    for s in seeds:
        out["GA"].append(ga_search(fitness_fn, 1, K_max, seed=s, **kw).history)
        out["greedy"].append(greedy_search(fitness_fn, 1, K_max, iterations, s).history)
        out["random"].append(random_search(fitness_fn, 1, K_max, iterations, s).history)
    return out


def comparison_markdown(rows: Sequence[Mapping[str, Any]], sat_K: int, peak: float,
                        R: int, S: float) -> str:
    """Render the paper-vs-ours markdown report.

    Args:
        rows: Output of :func:`budget_row` per budget.
        sat_K: Saturation K of our profile.
        peak: Peak profiled accuracy (fraction).
        R: Communication rounds.
        S: Per-client payload in MB.

    Returns:
        Markdown text with the comparison table and computed observations.
    """
    lines = [
        "# M0 — LoRaC-GA on EMNIST: paper vs ours", "",
        f"R={R}, S={S:.4f} MB (paper: 0.0833). Accuracy = mean of last rounds of "
        "a single seed. Paper numbers from Table 3.", "",
        "| B (MB) | K* ours | K* paper | acc ours % | acc paper % | cost ours MB "
        "| cost paper MB | GA = exhaustive | saving vs spend-all % | acc gap pts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pk, pa, pc = PAPER_TABLE3.get(int(r["B"]), ("-", "-", "-"))
        if r["K_star"] is None:
            lines.append(f"| {r['B']:g} | infeasible | {pk} | - | {pa} | - | {pc} | - | - | - |")
            continue
        lines.append(
            f"| {r['B']:g} | {r['K_star']} | {pk} | {100 * r['acc']:.1f} | {pa} "
            f"| {r['cost']:.2f} | {pc} | {'yes' if r['ga_matches_exhaustive'] else 'NO'} "
            f"({100 * r['ga_success_rate']:.0f}% of seeds) | {r['saving_pct']:.0f} "
            f"| {100 * r['acc_gap']:+.1f} |")
    feasible = [r for r in rows if r["K_star"] is not None]
    lines += ["", "## Observations (computed)", "",
              f"- Our A(K) peaks at {100 * peak:.1f}% (paper: ~96.6%) and reaches 99% "
              f"of that peak at K={sat_K}.",
              f"- GA found the exhaustive optimum for {sum(r['ga_matches_exhaustive'] for r in feasible)}"
              f"/{len(feasible)} budgets.",
              f"- min(A, B/C) equals A(K*) at every budget: "
              f"{all(r['min_term_inactive'] for r in feasible)}. On the feasible set "
              "C<=B means B/C>=1>=A, so the budget acts only through the cap K<=B/(R*S); "
              "the min term never binds (paper red flag #1).",
              "- Paper Table 3 is inconsistent with its own formula (K*=2 at B=10 although "
              "K=5 is feasible and more accurate; K* not monotone in B); see "
              "docs/M0_EMNIST_Validation.md."]
    return "\n".join(lines) + "\n"


def run_analysis(surface_json: str | Path, r: int, cfg, out_dir: str | Path,
                 S: float | None = None, fig3_budget: float = 100.0,
                 n_seeds: int = 10) -> dict[str, Any]:
    """Run the full M0 analysis and write ``analysis.json`` and ``comparison.md``.

    Args:
        surface_json: Profiler output.
        r: LoRA rank of the K-only profile.
        cfg: GA config (see ``configs/ga_search.yaml``).
        out_dir: Output directory.
        S: Payload override in MB; the measured value in the profile when ``None``.
        fig3_budget: Budget used for the convergence curves.
        n_seeds: Seeds for the success rate and convergence curves.

    Returns:
        The dict written to ``analysis.json``.
    """
    profile, S_meas = load_profile(surface_json, r)
    S = S_meas if S is None else S
    ga_kwargs = dict(population_size=cfg.population_size, generations=cfg.generations,
                     p_c=cfg.p_c, p_m=cfg.p_m, elitism=cfg.elitism,
                     tournament_size=cfg.tournament_size)
    seeds = [cfg.seed + i for i in range(n_seeds)]
    rows = [budget_row(profile, B, cfg.R, S, cfg.K_max, ga_kwargs, seeds)
            for B in cfg.budgets_mb]

    cap = max_feasible_K(fig3_budget, cfg.R, S, cfg.K_max)
    curves = convergence_curves(make_fitness(profile, fig3_budget, cfg.R, S), max(cap, 1),
                                seeds, cfg.generations, ga_kwargs)
    sat, peak = saturation_K(profile), max(profile.values())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"profile": {str(k): v for k, v in sorted(profile.items())}, "S": S,
              "R": cfg.R, "rows": rows, "saturation_K": sat, "peak_acc": peak,
              "fig3_budget": fig3_budget, "convergence": curves}
    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    (out / "comparison.md").write_text(comparison_markdown(rows, sat, peak, cfg.R, S))

    from src.utils.plots import plot_convergence
    plot_convergence(curves, out / "convergence.png",
                     title=f"Fitness vs iteration, B={fig3_budget:g} MB")
    plot_convergence(curves, out / "convergence.pdf",
                     title=f"Fitness vs iteration, B={fig3_budget:g} MB")
    logger.info("Analysis written to %s", out)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="results/m0_emnist/A_Kr_surface.json")
    ap.add_argument("--config", default="configs/ga_search.yaml")
    ap.add_argument("--r", type=int, default=7)
    ap.add_argument("--S", type=float, default=None, help="payload MB override")
    ap.add_argument("--fig3-budget", type=float, default=100.0)
    ap.add_argument("--out", default="results/m0_emnist/analysis")
    a = ap.parse_args()
    run_analysis(a.surface, a.r, OmegaConf.load(a.config), a.out, a.S, a.fig3_budget)
