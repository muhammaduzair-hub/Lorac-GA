"""Unit tests for the LoRA adapter wrapper.

All tests build a *tiny* randomly initialized DistilBERT locally, so nothing is
downloaded and everything runs on CPU in seconds.
"""

import pytest
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from src.models import lora_wrap


def tiny_base(num_labels: int = 2) -> DistilBertForSequenceClassification:
    config = DistilBertConfig(
        vocab_size=100,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
        num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(config)


@pytest.fixture
def model():
    return lora_wrap.apply_lora(tiny_base(), r=8, alpha=16, dropout=0.05)


class TestApplyLora:
    def test_only_lora_and_classifier_params_are_trainable(self, model):
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable
        assert all("lora_" in n or "classifier" in n or "pre_classifier" in n
                   for n in trainable)

    def test_adapters_attach_to_query_and_value_projections(self, model):
        lora_names = [n for n, _ in model.named_parameters() if "lora_A" in n]
        assert any("q_lin" in n for n in lora_names)
        assert any("v_lin" in n for n in lora_names)
        assert not any("k_lin" in n for n in lora_names)

    def test_forward_pass_produces_logits(self, model):
        out = model(input_ids=torch.ones(2, 8, dtype=torch.long))
        assert out.logits.shape == (2, 2)

    def test_rejects_non_positive_rank(self):
        with pytest.raises(ValueError):
            lora_wrap.apply_lora(tiny_base(), r=0)


class TestReportTrainable:
    def test_trainable_is_a_small_fraction_of_total(self, model):
        report = lora_wrap.report_trainable(model)
        assert 0 < report["trainable"] < report["total"]
        assert report["pct"] == pytest.approx(
            100.0 * report["trainable"] / report["total"]
        )


class TestAdapterSizeMb:
    def test_matches_manual_byte_count_of_travelling_tensors(self, model):
        state = lora_wrap.get_adapter_state(model)
        expected = sum(t.numel() * t.element_size() for t in state.values()) / 1e6
        assert lora_wrap.adapter_size_mb(model) == pytest.approx(expected)

    def test_scales_linearly_with_rank(self):
        # LoRA adds r*(d_in + d_out) params per target module, so S(r) = a + b*r
        # (the constant `a` is the rank-independent classification head).
        # Equal steps in r must give equal steps in S.
        s4 = lora_wrap.adapter_size_mb(lora_wrap.apply_lora(tiny_base(), r=4))
        s8 = lora_wrap.adapter_size_mb(lora_wrap.apply_lora(tiny_base(), r=8))
        s16 = lora_wrap.adapter_size_mb(lora_wrap.apply_lora(tiny_base(), r=16))
        assert s4 < s8 < s16
        assert (s16 - s8) == pytest.approx(2 * (s8 - s4), rel=1e-9)


class TestAdapterState:
    def test_state_holds_only_trainable_tensors(self, model):
        state = lora_wrap.get_adapter_state(model)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert set(state) == trainable

    def test_state_tensors_are_detached_cpu_copies(self, model):
        state = lora_wrap.get_adapter_state(model)
        for tensor in state.values():
            assert tensor.device.type == "cpu"
            assert not tensor.requires_grad

    def test_state_is_a_snapshot_not_a_live_view(self, model):
        state = lora_wrap.get_adapter_state(model)
        key = next(iter(state))
        before = state[key].clone()
        with torch.no_grad():
            dict(model.named_parameters())[key].add_(1.0)
        assert torch.equal(state[key], before)

    def test_set_adapter_state_round_trips(self, model):
        state = lora_wrap.get_adapter_state(model)
        bumped = {k: v + 1.0 for k, v in state.items()}
        lora_wrap.set_adapter_state(model, bumped)
        reloaded = lora_wrap.get_adapter_state(model)
        for key in bumped:
            assert torch.allclose(reloaded[key], bumped[key])

    def test_set_adapter_state_rejects_unknown_keys(self, model):
        with pytest.raises(KeyError):
            lora_wrap.set_adapter_state(model, {"not.a.real.param": torch.zeros(1)})


class TestBuildLoraModel:
    def test_loads_named_model_then_wraps_it(self, monkeypatch):
        seen = {}

        def fake_from_pretrained(name, num_labels=2, **kwargs):
            seen["name"] = name
            seen["num_labels"] = num_labels
            return tiny_base(num_labels)

        monkeypatch.setattr(
            lora_wrap.AutoModelForSequenceClassification,
            "from_pretrained",
            fake_from_pretrained,
        )
        model = lora_wrap.build_lora_model("distilbert-base-uncased", num_labels=3, r=4)
        assert seen == {"name": "distilbert-base-uncased", "num_labels": 3}
        assert any("lora_A" in n for n, _ in model.named_parameters())
