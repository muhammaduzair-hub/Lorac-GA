"""Unit tests for the peft/torchao dispatcher guard.

peft >= 0.19 ships a LoRA dispatcher for torchao-quantized layers. Its
availability check RAISES ImportError when torchao is installed but older than
0.16 (the Kaggle image ships 0.10), and the dispatcher runs for every adapted
module, so `get_peft_model` dies before any of our code executes. Our models are
never torchao-quantized, so the correct behaviour is "torchao not available".
"""

import sys
import types

import pytest

from src.models.lora_wrap import disable_incompatible_torchao_dispatch

MODULE = "peft.tuners.lora.torchao"


@pytest.fixture
def fake_dispatcher(monkeypatch):
    module = types.ModuleType(MODULE)
    monkeypatch.setitem(sys.modules, MODULE, module)
    return module


class TestDisableIncompatibleTorchaoDispatch:
    def test_patches_a_checker_that_raises_on_an_old_torchao(self, fake_dispatcher):
        def raising_checker():
            raise ImportError(
                "Found an incompatible version of torchao. Found version 0.10.0, "
                "but only versions above 0.16.0 are supported"
            )

        fake_dispatcher.is_torchao_available = raising_checker
        assert disable_incompatible_torchao_dispatch() is True
        assert fake_dispatcher.is_torchao_available() is False

    def test_leaves_a_healthy_checker_untouched(self, fake_dispatcher):
        healthy = lambda: True  # noqa: E731
        fake_dispatcher.is_torchao_available = healthy
        assert disable_incompatible_torchao_dispatch() is False
        assert fake_dispatcher.is_torchao_available is healthy

    def test_no_op_when_peft_has_no_torchao_dispatcher(self, monkeypatch):
        monkeypatch.setitem(sys.modules, MODULE, None)
        assert disable_incompatible_torchao_dispatch() is False

    def test_does_not_swallow_unrelated_import_errors(self, fake_dispatcher):
        def unrelated():
            raise ImportError("libcudart.so.12: cannot open shared object file")

        fake_dispatcher.is_torchao_available = unrelated
        with pytest.raises(ImportError, match="libcudart"):
            disable_incompatible_torchao_dispatch()


class TestApplyLoraRunsTheGuard:
    def test_apply_lora_disarms_the_dispatcher_before_wrapping(self, fake_dispatcher):
        from transformers import DistilBertConfig, DistilBertForSequenceClassification

        from src.models.lora_wrap import apply_lora

        calls: list[str] = []

        def raising_checker():
            calls.append("checked")
            raise ImportError("Found an incompatible version of torchao. Found "
                              "version 0.10.0, but only versions above 0.16.0")

        fake_dispatcher.is_torchao_available = raising_checker
        base = DistilBertForSequenceClassification(
            DistilBertConfig(vocab_size=50, dim=16, n_layers=1, n_heads=2,
                             hidden_dim=32, max_position_embeddings=32)
        )
        model = apply_lora(base, r=4)
        assert any("lora_A" in n for n, _ in model.named_parameters())
        assert fake_dispatcher.is_torchao_available() is False
