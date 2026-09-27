"""Chat isolation and input routing, without downloading Qwen weights."""
from types import SimpleNamespace

import pytest
import torch

from scripts import chat_point_readout as chat


@pytest.mark.parametrize("response_format", ["json", "free"])
def test_only_baseline_receives_full_matrix(response_format):
    field = torch.tensor([[[1.125, 2.25], [3.5, 4.75]]])
    for mode in ("baseline", "interface"):
        messages = chat.initial_messages("Read row 1, column 1", [2, 2], field, mode, response_format)
        text = str(messages)
        assert ("1.125" in text) == (mode == "baseline")
        assert "Read row 1, column 1" in text
        assert ("Return only" in text) == (response_format == "json")


def test_histories_are_not_mutated_or_shared():
    baseline = [{"role": "assistant", "content": "baseline answer"}]
    interface = [{"role": "assistant", "content": "interface answer"}]
    field = torch.zeros(1, 2, 2)
    messages = chat.conversation_messages(baseline, "Why?", [2, 2], field, "baseline", "free")
    assert len(baseline) == 1
    assert messages[-1]["content"] == "Why?"
    assert "interface answer" not in str(messages)
    assert len(interface) == 1


@pytest.mark.parametrize("mode", ["baseline", "interface"])
@pytest.mark.parametrize("fails", [False, True])
def test_memory_routing_and_cleanup(monkeypatch, mode, fails):
    events = []
    class Sidecar:
        def clear(self):
            events.append("clear")
        def bind(self, field, mode):
            events.append((mode, None if field is None else tuple(field.shape)))
    def generate(*args, **kwargs):
        if fails:
            raise RuntimeError("generation failed")
        return {"prediction": "[1.0]"}
    monkeypatch.setattr(chat.trainer, "native_chat_ids", lambda *a, **k: [1, 2])
    monkeypatch.setattr(chat.trainer, "generate_from_prompt", generate)
    args = (SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100)), Sidecar(), None,
            [], torch.zeros(1, 2, 2), mode, torch.device("cpu"), torch.float32, 50, 10)
    if fails:
        with pytest.raises(RuntimeError, match="generation failed"):
            chat.reply(*args)
    else:
        assert chat.reply(*args)["prediction"] == "[1.0]"
    assert events == ["clear", ("correct", (1, 1, 2, 2)) if mode == "interface"
                      else ("no_tensor", None), "clear"]


def test_context_overflow_refuses_silent_truncation(monkeypatch):
    monkeypatch.setattr(chat.trainer, "native_chat_ids", lambda *a, **k: [1] * 20)
    with pytest.raises(ValueError, match="/reset"):
        chat.reply(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=25)), None,
                   None, [], torch.zeros(1, 2, 2), "baseline", torch.device("cpu"),
                   torch.float32, 100, 10)
