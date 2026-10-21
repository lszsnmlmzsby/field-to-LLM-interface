"""Actual training-loop integration with local tiny Qwen, never remote readiness."""
import copy
import json
import sys
from pathlib import Path

import torch

from test_multisource_readout import config as source_fixture
from test_point_readout import tiny_components
from tensor_compression.downstream.point_readout_data import build_dataset, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_v2_recipe_preserves_complete_sources_and_expands_crop_budget():
    original = load_config(ROOT / "configs/field_to_llm_multisource.yaml", "full")
    full = load_config(ROOT / "configs/field_to_llm_multisource_v2.yaml", "full")
    assert full["data"]["sources"] == original["data"]["sources"]
    assert full["data"]["shape_sampling"] == original["data"]["shape_sampling"]
    assert full["generation"]["split_seed"] == original["generation"]["split_seed"]
    counts = full["generation"]["train_states_by_source"]
    assert counts == {"ins_tracer": 32768, "acoustic_scattering_inclusions": 65536,
                      "euler_multi_quadrants_openBC": 262144, "gray_scott_reaction_diffusion": 65536}
    assert sum(counts.values()) == full["generation"]["train_states"] == 425984
    assert full["generation"]["eval_states"] == 2048
    assert full["training"]["epochs"] == 2 and full["training"]["max_updates"] is None
    assert full["generation"]["query_program"]["tasks_per_state"] == 11
    assert len(full["generation"]["tasks"]) == 14
    assert full["training"]["max_target_tokens"] == full["evaluation"]["max_new_tokens"] == 256
    assert full["data"]["qa_dir"] != original["data"]["qa_dir"]
    smoke = load_config(ROOT / "configs/field_to_llm_multisource_v2.yaml", "smoke")
    assert smoke["generation"]["query_program"]["tasks_per_state"] == 4
    assert "capability_weights" in smoke["generation"]["query_program"]


def test_program_training_resume_and_separate_language_diagnostics(source_fixture, tmp_path, monkeypatch):
    from transformers import Qwen2ForCausalLM
    from scripts import train_point_readout as trainer
    from scripts.score_point_readout import score_predictions
    from tensor_compression.downstream.point_readout_data import PointReadoutDataset, read_jsonl

    recipe = load_config(ROOT / "configs/field_to_llm_multisource_v2.yaml", "smoke")
    recipe["data"].update(copy.deepcopy(source_fixture["data"]))
    recipe["data"]["sources"] = recipe["data"]["sources"][:2]  # Tiny INS/acoustic test fixtures only.
    recipe["generation"].update(train_states=64, eval_states=80,
        tasks=["single_point", "point_difference", "threshold_fraction"])
    recipe["generation"]["query_program"].update(tasks_per_state=3, diagnostic_states_per_source=2)
    qa = tmp_path / "qa"
    build_dataset(recipe, None, qa)
    _, example_llm, _, tokenizer, recipe = tiny_components(recipe)
    # Keep >1 channel per GroupNorm group even for the legal 1x1 input.
    # The production recipe uses 32 channels; the shared tiny helper uses only 8.
    recipe["field_encoder"]["model"]["base_channels"] = 16
    llm_config = copy.deepcopy(example_llm.config)
    recipe["data"]["qa_dir"] = str(qa)
    recipe["experiment_profile"] = "smoke"
    recipe["training"].update(max_updates=2, log_interval=1, save_every_updates=1,
        eval_every_updates=1, batch_size=3, gradient_accumulation_steps=2)
    recipe["evaluation"]["max_new_tokens"] = recipe["training"]["max_target_tokens"]
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(recipe), encoding="utf-8")

    def load_model(args, device):
        torch.manual_seed(123)
        return Qwen2ForCausalLM(copy.deepcopy(llm_config)).to(device), torch.float32

    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", load_model)
    # Test data/score/suite integration without hundreds of meaningless random decodes.
    monkeypatch.setattr(trainer, "generate_from_prompt", lambda *a, **k:
        {"prediction": "[0.0]", "terminated": True, "stop_token_id": 2, "generated_tokens": 3})

    def run(out, *extra):
        monkeypatch.setattr(sys, "argv", ["train_point_readout.py", "--config", str(config_path),
            "--profile", "smoke", "--device", "cpu", "--output-dir", str(out), *extra])
        trainer.main()

    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    run(direct)
    run(resumed, "--stop-after-updates", "1")
    run(resumed, "--resume", str(resumed / "last.pt"))
    a = torch.load(direct / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    assert a["step"] == b["step"] == 2
    for name in a["sidecar"]:
        torch.testing.assert_close(a["sidecar"][name], b["sidecar"][name], rtol=0, atol=0)

    for mode in ("correct", "disabled"):
        out = tmp_path / f"language_{mode}"
        run(out, "--resume", str(resumed / "best.pt"), "--evaluate-only", "--split", "test",
            "--eval-suite", "language", "--input-mode", mode)
        metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
        assert metrics["input_mode"] == mode
        assert "paired_language" in metrics
        dataset = PointReadoutDataset(qa, None, "test", suite="language")
        try:
            predictions = read_jsonl(out / "test_predictions.jsonl")
            offline = score_predictions(dataset, predictions)
            assert offline["answer_accuracy"] == metrics["answer_accuracy"]
            assert offline["paired_language"] == metrics["paired_language"]
        finally:
            dataset.close()
