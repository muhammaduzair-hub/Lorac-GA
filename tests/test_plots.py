"""Unit tests for the M3 surface figures.

Rendering is verified structurally (file written, non-empty) against a small
fake surface, plus the r-selection error path.
"""

import pytest

from src.utils.plots import plot_AK_curve, plot_AKr_heatmap


@pytest.fixture
def fake_surface():
    """A tiny surface dict spanning two K values and two ranks, incl. r=8."""
    records = [
        {"K": 5, "r": 8, "seed": 42, "acc": 0.80, "comm_mb": 1.0, "s0": 0.1},
        {"K": 5, "r": 8, "seed": 43, "acc": 0.82, "comm_mb": 1.0, "s0": 0.1},
        {"K": 10, "r": 8, "seed": 42, "acc": 0.85, "comm_mb": 2.0, "s0": 0.1},
        {"K": 5, "r": 4, "seed": 42, "acc": 0.70, "comm_mb": 0.5, "s0": 0.1},
        {"K": 10, "r": 4, "seed": 42, "acc": 0.74, "comm_mb": 1.0, "s0": 0.1},
    ]
    return {"records": records}


class TestPlotAKCurve:
    def test_writes_a_nonempty_figure(self, fake_surface, tmp_path):
        out = plot_AK_curve(fake_surface, tmp_path / "ak.pdf", r_fixed=8)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_accepts_a_json_path(self, fake_surface, tmp_path):
        import json
        path = tmp_path / "surface.json"
        path.write_text(json.dumps(fake_surface))
        out = plot_AK_curve(path, tmp_path / "ak.png", r_fixed=8)
        assert out.exists()

    def test_raises_when_no_cell_at_that_rank(self, fake_surface, tmp_path):
        with pytest.raises(ValueError):
            plot_AK_curve(fake_surface, tmp_path / "ak.pdf", r_fixed=16)


class TestPlotAKrHeatmap:
    def test_writes_a_nonempty_figure(self, fake_surface, tmp_path):
        out = plot_AKr_heatmap(fake_surface, tmp_path / "heat.pdf")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_missing_cells_without_crashing(self, tmp_path):
        # Sparse surface: (10, 8) is absent -> rendered as a blank cell.
        surface = {"records": [
            {"K": 5, "r": 8, "seed": 42, "acc": 0.8, "comm_mb": 1.0, "s0": 0.1},
            {"K": 5, "r": 4, "seed": 42, "acc": 0.7, "comm_mb": 1.0, "s0": 0.1},
            {"K": 10, "r": 4, "seed": 42, "acc": 0.72, "comm_mb": 1.0, "s0": 0.1},
        ]}
        out = plot_AKr_heatmap(surface, tmp_path / "heat.png")
        assert out.exists()

    def test_raises_on_empty_surface(self, tmp_path):
        with pytest.raises(ValueError):
            plot_AKr_heatmap({"records": []}, tmp_path / "heat.pdf")
