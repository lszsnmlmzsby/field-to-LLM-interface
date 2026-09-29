from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from tensor_compression.downstream.block_tokens import block_shape, pack_blocks, masked_reconstruction
from tensor_compression.downstream.point_readout_data import load_config
from tensor_compression.downstream.point_readout import make_record, normalize_field
from scripts import train_point_readout as trainer
from test_point_readout import tiny_components


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3), (4, 4)])
@pytest.mark.parametrize("hw", [(8, 8), (9, 17), (17, 90)])
def test_packing_preserves_every_cell_and_order(shape, hw):
    h, w = hw
    ph, pw = shape
    x = torch.arange(2 * 3 * h * w).reshape(2, 3, h, w).float().requires_grad_()
    packed, valid = pack_blocks(x, shape)
    assert packed.shape == (2, math.ceil(h/ph)*math.ceil(w/pw), ph*pw, 3)
    assert valid.sum().item() == 2*h*w
    for r in range(h):
        for c in range(w):
            token = (r//ph)*math.ceil(w/pw)+c//pw
            slot = (r % ph)*pw+c % pw
            assert torch.equal(packed[:, token, slot], x[:, :, r, c])
    assert torch.count_nonzero(packed[~valid]) == 0
    packed.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_padding_does_not_contribute_to_reconstruction():
    target = torch.tensor([[[1., 2., 0., 0.]]])
    prediction = torch.tensor([[[1., 2., 900., -900.]]], requires_grad=True)
    valid = torch.tensor([[[True, True, False, False]]])
    loss = masked_reconstruction(prediction, target, valid)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(prediction.grad) == 0


@pytest.mark.parametrize("shape", [[0, 2], [True, 2], [2], "2x2"])
def test_invalid_block_shape(shape):
    with pytest.raises(ValueError):
        block_shape(shape)


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3)])
def test_tiny_qwen_training_reload_and_dynamic_shape(shape):
    config = load_config(ROOT / "configs/field_to_llm_point_readout.yaml", "smoke")
    config["memory"]["block_shape"] = list(shape)
    trainer, llm, sidecar, tokenizer, config = tiny_components(config)
    z = normalize_field(torch.arange(9*17).reshape(9, 17).float())[0]
    row = make_record("state", z, "single_point", {"points": [[9, 17]]})
    frozen = {name: p.detach().clone() for name, p in llm.named_parameters() if not p.requires_grad}
    loss, _ = trainer.training_loss(llm, sidecar, tokenizer, [row], z[None, None],
                                    torch.device("cpu"), torch.float32, config["training"])
    loss.backward()
    memory = sidecar.memory
    assert memory.value_reconstruction.weight.grad.abs().sum() > 0
    assert memory.spatial_backbone.latent_projection.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=1e-3)
    optimizer.step()
    for name, p in llm.named_parameters():
        if name in frozen:
            assert p.grad is None and torch.equal(p, frozen[name])
    sidecar.clear()
    sidecar.eval()
    state_dict = copy.deepcopy(sidecar.state_dict())
    _, _, restored, _, _ = tiny_components(config)
    restored.load_state_dict(state_dict, strict=True)
    restored.eval()
    with torch.no_grad():
        a, b = sidecar.memory(z[None, None]), restored.memory(z[None, None])
        assert torch.equal(a.content, b.content)
        assert torch.equal(a.value, b.value)
        large = restored.memory(torch.randn(1, 1, 17, 90))
    assert large.content.shape[1] == math.ceil(17/shape[0])*math.ceil(90/shape[1])
    if shape == (1, 1):
        assert memory.value_reconstruction.out_features == 1
        assert memory.value_encoder[0].in_features == 6
    else:
        assert memory.value_reconstruction.out_features == math.prod(shape)
    wrong = copy.deepcopy(config)
    wrong["memory"]["block_shape"] = [4, 4]
    _, _, incompatible, _, _ = tiny_components(wrong)
    with pytest.raises(RuntimeError):
        incompatible.load_state_dict(state_dict, strict=True)


def test_one_cell_path_and_absolute_positions_remain_compatible():
    from scripts.train_tensor_qwen_cross_attention import FullGridSpatialBackbone, sinusoidal_2d_position_encoding
    args = dict(latent_channels=3, latent_grid=(8, 8), adapter_dim=16, adapter_layers=1,
                adapter_heads=2, dropout=0, dynamic_grid=True)
    legacy = FullGridSpatialBackbone(**args)
    explicit = FullGridSpatialBackbone(**args, block_shape=(1, 1))
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(1, 3, 9, 17)
    expected = legacy.latent_projection(x.flatten(2).transpose(1, 2)) + sinusoidal_2d_position_encoding(9, 17, 16)
    assert torch.equal(explicit.spatial_input_states(x)[0], expected)
    block = FullGridSpatialBackbone(**args, block_shape=(2, 2))
    positions = sinusoidal_2d_position_encoding(9, 17, 16).reshape(1, 9, 17, 16)
    assert torch.equal(block.position_encoding((9, 17)), positions[:, ::2, ::2].reshape(1, -1, 16))


def test_feasibility_budget_and_resume_contract():
    config = load_config(ROOT / "configs/field_to_llm_block_readout.yaml", "pilot")
    trainer.validate_config(config)
    n = config["generation"]["train_states"] * len(config["generation"]["tasks"])
    assert n == 11264
    assert n * config["training"]["epochs"] // config["training"]["gradient_accumulation_steps"] == 5632
    first = {"text_encoding": "same", "recipe": {"memory": {"block_shape": [1, 1]}}}
    second = copy.deepcopy(first)
    second["recipe"]["memory"]["block_shape"] = [2, 2]
    checkpoint = {"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": first}
    with pytest.raises(ValueError, match="differs"):
        trainer.validate_resume(checkpoint, second)


def test_comparison_requires_same_experiment_and_questions(tmp_path, capsys):
    from scripts.compare_block_readout import compare
    dirs = [tmp_path / "reference", tmp_path / "compressed"]
    metric = {"questions": 1, "answer_accuracy": .5, "unit_accuracy": .75,
              "macro_task_answer_accuracy": .5, "elapsed_seconds": 1.,
              "peak_allocated_gib": None, "generated_tokens": 4,
              "seen_shapes": None, "heldout_shapes": None}
    metric["by_task_type"] = {"single_point": {"answer_accuracy": .5, "unit_accuracy": .75}}
    for directory, block in zip(dirs, [1, 2]):
        directory.mkdir()
        contract = {"dataset": "same", "recipe": {"memory": {"block_shape": [block, block]}}}
        (directory / "contract.json").write_text(json.dumps(contract))
        (directory / "val_metrics.json").write_text(json.dumps(metric))
        (directory / "val_predictions.jsonl").write_text(json.dumps({"qa_id": "a", "field_tokens": 16//block**2})+"\n")
    compare(*dirs)
    assert "single_point" in capsys.readouterr().out
    (dirs[1] / "val_predictions.jsonl").write_text(json.dumps({"qa_id": "b", "field_tokens": 4})+"\n")
    with pytest.raises(ValueError, match="question sets"):
        compare(*dirs)
    contract["dataset"] = "different"
    (dirs[1] / "contract.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Dataset"):
        compare(*dirs)
