"""Unit tests for client-side local training and evaluation."""

import pytest
import torch
from torch.utils.data import DataLoader

from src.fl.client import evaluate, local_train
from src.models.lora_wrap import get_adapter_state


@pytest.fixture
def loader(tiny_dataset):
    return DataLoader(tiny_dataset, batch_size=8)


class _FixedLogitsModel(torch.nn.Module):
    """Model returning pre-set logits, to test the accuracy wiring exactly."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.cursor = 0

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch = input_ids.shape[0]
        chunk = self.logits[self.cursor:self.cursor + batch]
        self.cursor += batch
        return type("Out", (), {"logits": chunk, "loss": None})()


class TestLocalTrain:
    def test_returns_the_adapter_parameter_names(self, tiny_model, loader):
        state = local_train(tiny_model, loader, lr=1e-3, local_epochs=1, device="cpu")
        assert set(state) == set(get_adapter_state(tiny_model))

    def test_adapter_weights_actually_move(self, tiny_model, loader):
        before = get_adapter_state(tiny_model)
        after = local_train(tiny_model, loader, lr=1e-2, local_epochs=1, device="cpu")
        changed = [k for k in before if not torch.allclose(before[k], after[k])]
        assert changed, "local training left every adapter tensor untouched"

    def test_frozen_backbone_is_not_updated(self, tiny_model, loader):
        frozen = {
            n: p.detach().clone()
            for n, p in tiny_model.named_parameters()
            if not p.requires_grad
        }
        local_train(tiny_model, loader, lr=1e-2, local_epochs=1, device="cpu")
        for name, param in tiny_model.named_parameters():
            if name in frozen:
                assert torch.equal(param.detach(), frozen[name])

    def test_returned_state_is_on_cpu(self, tiny_model, loader):
        state = local_train(tiny_model, loader, lr=1e-3, local_epochs=1, device="cpu")
        assert all(t.device.type == "cpu" for t in state.values())

    def test_rejects_empty_loader(self, tiny_model, tiny_dataset):
        empty = DataLoader(torch.utils.data.Subset(tiny_dataset, []), batch_size=8)
        with pytest.raises(ValueError):
            local_train(tiny_model, empty, lr=1e-3, local_epochs=1, device="cpu")


class TestEvaluate:
    def test_computes_accuracy_from_argmax_of_logits(self, tiny_dataset):
        labels = tiny_dataset["labels"]
        # Logits that are right for the first half, wrong for the second.
        logits = torch.zeros(len(labels), 2)
        for i, label in enumerate(labels.tolist()):
            correct = label if i < len(labels) // 2 else 1 - label
            logits[i, correct] = 1.0
        model = _FixedLogitsModel(logits)
        loader = DataLoader(tiny_dataset, batch_size=8)
        assert evaluate(model, loader, device="cpu") == pytest.approx(0.5)

    def test_returns_a_probability_for_a_real_model(self, tiny_model, tiny_dataset):
        loader = DataLoader(tiny_dataset, batch_size=8)
        acc = evaluate(tiny_model, loader, device="cpu")
        assert 0.0 <= acc <= 1.0

    def test_leaves_model_in_eval_free_state_for_training(self, tiny_model, tiny_dataset):
        loader = DataLoader(tiny_dataset, batch_size=8)
        evaluate(tiny_model, loader, device="cpu")
        assert not tiny_model.training  # evaluation must not silently re-enable dropout
