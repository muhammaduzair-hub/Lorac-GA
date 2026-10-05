"""Offline tests for the EMNIST federated loop (synthetic data, tiny model)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

from src.fl.emnist_simulation import (build_model, run_federated_emnist,  # noqa: E402
                                      tail_mean)

CLASSES = 5


def make_datasets(n_train=400, n_eval=64):
    g = torch.Generator().manual_seed(0)

    def ds(n):
        y = torch.arange(n) % CLASSES
        x = torch.randn(n, 1, 28, 28, generator=g) + y.view(-1, 1, 1, 1).float()
        return TensorDataset(x, y)
    return {"train": ds(n_train), "eval": ds(n_eval)}


def make_cfg(tmp_path, **kw):
    cfg = dict(
        data_root="data", num_classes=CLASSES, num_clients=10, alpha_dirichlet=0.5,
        min_samples_per_client=1, vit=dict(dim=16, depth=2, heads=2, mlp_dim=32),
        backbone_path=None, lora_targets=["q", "v"], r=2, lora_alpha=4, K=3, R=3,
        tail_rounds=2, prox_mu=0.0, local_epochs=1, batch_size=16, eval_batch_size=64,
        lr=1e-2, seed=42, device="cpu", output_dir=str(tmp_path / "run"),
    )
    cfg.update(kw)
    return OmegaConf.create(cfg)


def test_tail_mean():
    hist = [{"test_acc": a} for a in (0.1, 0.5, 0.7)]
    assert tail_mean(hist, 2) == pytest.approx(0.6)
    assert tail_mean(hist, 10) == pytest.approx(1.3 / 3)
    with pytest.raises(ValueError):
        tail_mean([], 3)


def test_run_records_paper_cost_and_history(tmp_path):
    cfg = make_cfg(tmp_path)
    out = run_federated_emnist(cfg, datasets=make_datasets())
    S = out["adapter_size_mb"]
    assert len(out["history"]) == cfg.R
    assert out["history"][-1]["comm_mb_cumulative"] == pytest.approx(cfg.R * cfg.K * S)
    assert out["final_acc"] == out["history"][-1]["test_acc"]
    assert 0.0 <= out["tail_acc"] <= 1.0


def test_is_deterministic(tmp_path):
    a = run_federated_emnist(make_cfg(tmp_path / "a"), datasets=make_datasets())
    b = run_federated_emnist(make_cfg(tmp_path / "b"), datasets=make_datasets())
    assert [h["test_acc"] for h in a["history"]] == [h["test_acc"] for h in b["history"]]


def test_resume_keeps_earlier_rounds(tmp_path):
    ds = make_datasets()
    run_federated_emnist(make_cfg(tmp_path, R=2), datasets=ds)
    out = run_federated_emnist(make_cfg(tmp_path, R=4), datasets=ds)
    assert [h["round"] for h in out["history"]] == [1, 2, 3, 4]


def test_fedprox_runs_and_backbone_stays_frozen(tmp_path):
    cfg = make_cfg(tmp_path, prox_mu=0.01)
    model = build_model(cfg)
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    out = run_federated_emnist(cfg, model=model, datasets=make_datasets())
    assert len(out["history"]) == cfg.R
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters()
               if n in before)


def test_learns_synthetic_task(tmp_path):
    cfg = make_cfg(tmp_path, R=6, K=5, lr=2e-2, local_epochs=2)
    out = run_federated_emnist(cfg, datasets=make_datasets())
    assert out["final_acc"] > 1.0 / CLASSES + 0.2
