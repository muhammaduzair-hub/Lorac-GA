"""Tests for the tiny LoRA ViT, its client training loop and backbone IO."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.fl.emnist_client import evaluate, local_train  # noqa: E402
from src.models.emnist_lora import (TinyViT, build_lora_vit, closest_rank,  # noqa: E402
                                    save_backbone)
from src.models.lora_wrap import adapter_size_mb, get_adapter_state  # noqa: E402

SMALL = dict(dim=16, depth=2, heads=2, mlp_dim=32)


def make_loader(n=32, classes=5, seed=0, batch=16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def test_forward_shape():
    out = TinyViT(47, **SMALL)(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 47)


def test_only_lora_and_head_trainable():
    model = build_lora_vit(5, r=2, alpha=4, **SMALL)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or n.startswith("head.") for n in trainable)
    assert any("blocks.0.q.lora_A" == n for n in trainable)
    assert not any("blocks.0.k" in n for n in trainable)


def test_lora_is_noop_at_init(tmp_path):
    torch.manual_seed(0)
    plain = TinyViT(5, **SMALL)
    path = str(tmp_path / "bb.pt")
    save_backbone(plain, path)
    lora = build_lora_vit(5, r=2, alpha=4, backbone_path=path, **SMALL)
    lora.head.load_state_dict(plain.head.state_dict())
    x = torch.randn(4, 1, 28, 28)
    assert torch.allclose(plain(x), lora(x), atol=1e-6)


def test_backbone_excludes_head_and_mismatch_raises(tmp_path):
    path = str(tmp_path / "bb.pt")
    save_backbone(TinyViT(10, **SMALL), path)
    assert not any(k.startswith("head.") for k in torch.load(path))
    with pytest.raises(RuntimeError):
        build_lora_vit(5, r=2, alpha=4, backbone_path=path, dim=32, depth=2,
                       heads=2, mlp_dim=32)


def test_adapter_size_matches_param_count():
    model = build_lora_vit(5, r=2, alpha=4, **SMALL)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert adapter_size_mb(model) == pytest.approx(n * 4 / 1e6)


def test_closest_rank_matches_paper_payload_for_default_vit():
    # configs/m0_emnist.yaml uses r=7 for S ~= 0.0833 MB; keep them in sync.
    r, s = closest_rank(0.0833, 47)
    assert r == 7 and abs(s - 0.0833) < 0.005


def test_local_train_updates_only_trainable_and_returns_their_state():
    torch.manual_seed(0)
    model = build_lora_vit(5, r=2, alpha=4, **SMALL)
    frozen_before = {n: p.clone() for n, p in model.named_parameters()
                     if not p.requires_grad}
    state = local_train(model, make_loader(), lr=1e-2, local_epochs=2)
    assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(torch.equal(frozen_before[n], p) for n, p in model.named_parameters()
               if n in frozen_before)


def test_local_train_empty_loader_raises():
    model = build_lora_vit(5, r=2, alpha=4, **SMALL)
    with pytest.raises(ValueError):
        local_train(model, DataLoader(TensorDataset(torch.empty(0, 1, 28, 28),
                                                    torch.empty(0).long())))


def test_fedprox_pulls_towards_global():
    def drift(mu):
        torch.manual_seed(0)
        model = build_lora_vit(5, r=2, alpha=4, **SMALL)
        g = get_adapter_state(model)
        s = local_train(model, make_loader(), lr=1e-2, local_epochs=3, prox_mu=mu)
        return sum(((s[k] - g[k]) ** 2).sum() for k in g)
    assert drift(100.0) < drift(0.0)


def test_training_improves_and_evaluate_in_range():
    torch.manual_seed(0)
    model = build_lora_vit(5, r=4, alpha=8, **SMALL)
    loader = make_loader(n=32)
    before = evaluate(model, loader)
    local_train(model, loader, lr=2e-2, local_epochs=60)
    after = evaluate(model, loader)
    assert 0.0 <= before <= 1.0 and after > before
