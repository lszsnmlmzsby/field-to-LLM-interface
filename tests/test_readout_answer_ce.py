"""Compare bounded batched answer CE to the established per-question objective."""
import copy

import pytest
import torch

from scripts import train_tensor_qwen_cross_attention as core


@pytest.mark.parametrize("cap", [1, 4, 128])
def test_batched_ce_preserves_unequal_answer_loss_and_gradients(cap):
    torch.manual_seed(19)
    hidden = torch.randn(3, 9, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    head = torch.nn.Linear(5, 13, bias=False)
    reference_head = copy.deepcopy(head)
    spans = [(2, 4), (3, 9), (1, 2)]
    labels = torch.full((3, 9), -100, dtype=torch.long)
    for row, (start, end) in enumerate(spans):
        labels[row, start:end] = torch.arange(end - start) % 13
    projection_sizes = []
    handle = head.register_forward_pre_hook(lambda module, args: projection_sizes.append(args[0].shape[0]))
    actual = core.batched_per_question_answer_ce(hidden, labels, head, answer_spans=spans,
                                                max_projection_tokens=cap)
    handle.remove()
    expected = torch.stack([core.full_answer_ce(reference_hidden[i:i+1], labels[i:i+1], reference_head)
                            for i in range(3)]).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(head.weight.grad, reference_head.weight.grad, rtol=1e-6, atol=1e-7)
    assert max(projection_sizes) <= cap
    assert sum(projection_sizes) == sum(end - start for start, end in spans)
    for row, (start, end) in enumerate(spans):
        assert torch.count_nonzero(hidden.grad[row, :start-1]) == 0
        assert torch.count_nonzero(hidden.grad[row, end-1:]) == 0


@pytest.mark.parametrize("spans", [[], [(0, 2)], [(1, 1)], [(1, 5)], [(True, 2)], [(1.0, 2)]])
def test_invalid_cpu_spans_are_rejected_before_projection(spans):
    hidden, labels = torch.randn(1, 4, 5), torch.zeros(1, 4, dtype=torch.long)
    head = torch.nn.Linear(5, 13)
    with pytest.raises(ValueError, match="span|offset|prompt"):
        core.batched_per_question_answer_ce(hidden, labels, head, answer_spans=spans)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_projection_limit_is_rejected(cap):
    with pytest.raises(ValueError, match="positive integer"):
        core.batched_per_question_answer_ce(torch.randn(1, 4, 5), torch.zeros(1, 4, dtype=torch.long),
            torch.nn.Linear(5, 13), answer_spans=[(1, 4)], max_projection_tokens=cap)
