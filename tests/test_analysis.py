"""Tests for the M0 analysis (synthetic profile; no torch needed)."""

import json

import pytest
from omegaconf import OmegaConf

from src.ga.analysis import (PAPER_TABLE3, budget_row, comparison_markdown,
                             convergence_curves, load_profile, run_analysis,
                             saturation_K)
from src.ga.fitness import interpolate_accuracy, make_fitness

R, S = 10, 0.0833
GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100]


def peaked_profile(peak_K=12):
    """A(K) rises to a peak at K=12 then declines slightly (non-monotone)."""
    return {k: 0.96 - 0.0008 * (k - peak_K) ** 2 / 4 if k < peak_K else
            0.96 - 0.0001 * (k - peak_K) for k in GRID}


GA_KW = dict(population_size=20, generations=30, p_c=0.5, p_m=0.2,
             elitism=0.1, tournament_size=3)


def write_surface(path, profile, r=7, S=0.0816):
    doc = {"surface": [{"K": k, "r": r, "acc_mean": a, "acc_std": 0.0, "n_seeds": 1,
                        "seeds": [42], "comm_mb": 0, "s0": S / r}
                       for k, a in profile.items()],
           "records": [{"K": k, "r": r, "seed": 42, "R": 10, "acc": a,
                        "adapter_size_mb": S} for k, a in profile.items()]}
    path.write_text(json.dumps(doc))


def ga_cfg():
    return OmegaConf.create(dict(seed=42, K_max=100, R=R, budgets_mb=[10, 50, 100, 1000],
                                 **GA_KW))


def test_interpolate_accuracy():
    A = interpolate_accuracy({1: 0.5, 3: 0.9})
    assert A(2) == pytest.approx(0.7) and A(0) == 0.5 and A(10) == 0.9


def test_load_profile_roundtrip_and_missing_rank(tmp_path):
    p = tmp_path / "s.json"
    write_surface(p, {1: 0.5, 5: 0.8})
    prof, S_meas = load_profile(p, 7)
    assert prof == {1: 0.5, 5: 0.8} and S_meas == 0.0816
    with pytest.raises(ValueError):
        load_profile(p, 3)


def test_saturation_K():
    assert saturation_K({1: 0.5, 10: 0.9, 100: 0.9}) <= 10
    assert saturation_K({1: 0.1, 100: 0.9}) > 90


def test_budget_caps_K_star_and_matches_exhaustive():
    prof = peaked_profile()
    tight = budget_row(prof, 5, R, S, 100, GA_KW, [42, 43, 44])   # cap = 6
    assert tight["K_cap"] == 6 and tight["K_star"] == 6
    loose = budget_row(prof, 1000, R, S, 100, GA_KW, [42, 43, 44])
    assert loose["K_star"] == 12 == loose["K_exhaustive"]
    assert loose["ga_matches_exhaustive"] and loose["ga_success_rate"] == 1.0
    assert loose["saving_pct"] > 80 and loose["acc_gap"] >= 0


def test_min_term_never_binds_on_feasible_set():
    prof = peaked_profile()
    for B in (10, 50, 100, 1000):
        assert budget_row(prof, B, R, S, 100, GA_KW, [42])["min_term_inactive"]


def test_infeasible_budget_has_no_K_star():
    row = budget_row(peaked_profile(), 0.5, R, S, 100, GA_KW, [42])
    assert row["K_star"] is None and row["K_cap"] == 0


def test_convergence_curves_shape():
    f = make_fitness(peaked_profile(), 1000, R, S)
    c = convergence_curves(f, 100, [1, 2], 15, GA_KW)
    assert set(c) == {"GA", "greedy", "random"}
    assert all(len(h) == 15 for runs in c.values() for h in runs) and len(c["GA"]) == 2


def test_comparison_markdown_has_paper_columns_and_claims():
    rows = [budget_row(peaked_profile(), B, R, S, 100, GA_KW, [42]) for B in (10, 1000)]
    md = comparison_markdown(rows, saturation_K(peaked_profile()), 0.96, R, S)
    assert "K* paper" in md and "min(A, B/C)" in md
    assert str(PAPER_TABLE3[10][0]) in md


def test_run_analysis_writes_outputs(tmp_path):
    p = tmp_path / "s.json"
    write_surface(p, peaked_profile())
    res = run_analysis(p, 7, ga_cfg(), tmp_path / "out", n_seeds=3)
    for name in ("analysis.json", "comparison.md", "convergence.png", "convergence.pdf"):
        assert (tmp_path / "out" / name).exists()
    assert [r["B"] for r in res["rows"]] == [10, 50, 100, 1000]
