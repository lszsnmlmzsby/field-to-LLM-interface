"""Deterministic local position-cache checks; no remote throughput claims."""
import copy

import pytest
import torch

from tensor_compression.workflows import train_tensor_qwen_cross_attention as core


def backbone(block=(1, 1), dim=18):
    return core.FullGridSpatialBackbone(latent_channels=3, latent_grid=(4, 4),
        adapter_dim=dim, adapter_layers=1, adapter_heads=2, dropout=0,
        dynamic_grid=True, block_shape=block)


@pytest.mark.parametrize("block", [(1, 1), (2, 2), (1, 2), (2, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dynamic_axes_preserve_exact_absolute_positions(block, dtype):
    model = backbone(block)
    for shape in ((9, 17), (1, 7), (15, 3), (2, 1), (9, 17)):
        actual = model._dynamic_position_encoding(shape, device="cpu", dtype=dtype)
        expected = model.position_encoding(shape).to(dtype=dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_axes_are_reused_and_only_growing_axis_is_rebuilt(monkeypatch):
    model = backbone()
    calls = []
    original = core.sinusoidal_position_encoding

    def counted(length, dim):
        calls.append((length, dim))
        return original(length, dim)

    monkeypatch.setattr(core, "sinusoidal_position_encoding", counted)
    for shape in ((9, 17), (9, 17), (3, 11), (15, 3), (9, 17)):
        model._dynamic_position_encoding(shape, device="cpu", dtype=torch.float32)
    assert calls == [(9, 9), (17, 9), (15, 9)]
    model._dynamic_position_encoding((9, 17), device="cpu", dtype=torch.bfloat16)
    assert calls[-2:] == [(9, 9), (17, 9)]
    assert all(axis.dtype == torch.bfloat16 for axis in model._position_axes_cache)
    model.double()
    assert model._position_axes_cache is None


def test_cache_is_bounded_to_two_axes_and_absent_from_checkpoints():
    model = backbone(dim=512)
    state = copy.deepcopy(model.state_dict())
    buffers = set(dict(model.named_buffers()))
    for shape in ((128, 1), (1, 128), (17, 93), (127, 127), (3, 11)):
        model._dynamic_position_encoding(shape, device="cpu", dtype=torch.bfloat16)
    rows, cols = model._position_axes_cache
    assert rows.shape == cols.shape == (128, 256)
    assert sum(axis.numel() * axis.element_size() for axis in (rows, cols)) == 128 * 1024
    assert set(dict(model.named_buffers())) == buffers
    assert set(model.state_dict()) == set(state)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    restored = backbone(dim=512)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored._position_axes_cache is None


@pytest.mark.parametrize("block", [(1, 1), (2, 2)])
def test_forward_gradients_and_legacy_base_buffer_are_unchanged(block):
    model = backbone(block)
    reference = copy.deepcopy(model)
    reference._dynamic_position_encoding = lambda grid, *, device, dtype: reference.position_encoding(grid).to(
        device=device, dtype=dtype)
    x = torch.randn(2, 3, 9, 7, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    outputs = model.spatial_input_states(x)
    expected = reference.spatial_input_states(expected_x)
    for actual, wanted in zip(outputs, expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    sum(value.square().sum() for value in outputs).backward()
    sum(value.square().sum() for value in expected).backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
    for (name, actual), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
        if wanted.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
    # A legacy checkpoint's registered base table remains authoritative.
    model.spatial_pos_encoding.fill_(7)
    reference.spatial_pos_encoding.fill_(7)
    base = torch.randn(1, 3, 4, 4)
    for actual, wanted in zip(model.spatial_input_states(base), reference.spatial_input_states(base)):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available locally")
def test_device_move_clears_axes_and_rebuilds_exact_cpu_formula_on_cuda():
    model = backbone()
    model._dynamic_position_encoding((9, 17), device="cpu", dtype=torch.float32)
    model.cuda()
    assert model._position_axes_cache is None
    actual = model._dynamic_position_encoding((9, 17), device=model.latent_projection.weight.device,
                                             dtype=torch.bfloat16)
    torch.testing.assert_close(actual.cpu(), model.position_encoding((9, 17)).bfloat16(), rtol=0, atol=0)
    model.cpu()
    assert model._position_axes_cache is None
