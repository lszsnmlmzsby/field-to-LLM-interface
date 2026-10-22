"""Diagnostics are locally tested with exact oracles and tiny, untrained models."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from test_point_readout import CharacterTokenizer, tiny_components
from scripts import train_point_readout as trainer
from scripts.score_point_readout import score_predictions
from tensor_compression.downstream.point_readout import make_record, score_answer
from tensor_compression.downstream.readout_evaluation import (
    FieldPairDataset, evaluation_annotations, paired_language_metrics, summarize_readout_evaluation,
    zero_baseline_metrics, _disjoint_reference_answers,
)


def rows_and_states():
    field = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    rows = [make_record("field", field, "single_point", {"points": [[2, 3]]}),
            make_record("field", field, "region_argmax", {"start": [1, 1], "size": [3, 4]})]
    states = {"field": {"field": "pressure", "source_id": "fixture", "family": "fixture",
                        "is_constant": False, "is_static": False}}
    return field, rows, states


def annotated(row, prediction):
    return {"qa_id": row["qa_id"], "shape": "3x4", "field": "pressure", "task_type": row["task_type"],
            "score": score_answer(row, prediction), **evaluation_annotations(row, (2, 2))}


def test_language_pair_accuracy_distinguishes_equal_wrong_answers_and_missing_members():
    _, rows, _ = rows_and_states()
    scored = []
    for semantic, answers in (("wrong-agreement", ["[0]", "[0]"]),
                              ("both-correct", [rows[0]["answer"]] * 2),
                              ("language-drop", [rows[0]["answer"], "[0]"])):
        for group, prediction in zip(("seen", "heldout"), answers):
            row = {**rows[0], "semantic_id": semantic, "language_group": group}
            scored.append(annotated(row, prediction))
    incomplete = annotated({**rows[0], "semantic_id": "incomplete", "language_group": "heldout"}, "[0]")
    result = paired_language_metrics(scored + [incomplete])
    assert result["complete_pairs"] == 3 and result["incomplete_pairs"] == 1
    assert result["both_correct_rate"] == pytest.approx(1 / 3)
    assert result["valid_prediction_agreement_rate"] == pytest.approx(2 / 3)
    assert result["seen_correct_heldout_wrong_rate"] == pytest.approx(1 / 3)
    altered = copy.deepcopy(scored)
    altered[1]["label_signature"] = "different-gold"
    with pytest.raises(ValueError, match="reference answer"):
        paired_language_metrics(altered)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_language_metrics(scored + scored[:1])


def test_query_diagnostics_count_real_block_members_and_keep_principal_score():
    field, rows, _ = rows_and_states()
    near = make_record("field", field, "multi_point", {"points": [[1, 1], [2, 2]]})
    far = make_record("field", field, "multi_point", {"points": [[1, 1], [3, 4]]})
    near_tags, far_tags = evaluation_annotations(near, (2, 2)), evaluation_annotations(far, (2, 2))
    assert near_tags["block_relation"] == "single_block"
    assert far_tags["block_relation"] == "multiple_blocks"
    assert near_tags["output_length_bucket"] == far_tags["output_length_bucket"] == "2-4"
    scored = [annotated(row, row["answer"]) for row in [*rows, near, far]]
    result = summarize_readout_evaluation(scored, [[3, 4]])
    assert result["macro_task_answer_accuracy"] == 1
    assert result["questions"] == 4
    assert sum(group["questions"] for group in result["by_block_relation"].values()) == 4
    assert "paired_language" not in result


def test_zero_baseline_keeps_actual_tolerances_and_constant_strata_separate():
    field, rows, states = rows_and_states()
    constant = make_record("constant", torch.zeros_like(field), "single_point", {"points": [[1, 1]]})
    states["constant"] = {**states["field"], "is_constant": True}
    result = zero_baseline_metrics([*rows, constant], states)
    assert result["diagnostic_only"]
    assert result["numerical_questions"] == 2 and result["excluded_coordinate_questions"] == 1
    assert result["metrics"]["answer_accuracy"] == .5
    assert result["metrics"]["by_is_constant"]["True"]["answer_accuracy"] == 1
    assert result["metrics"]["by_is_constant"]["False"]["answer_accuracy"] == 0


def test_offline_scores_recover_model_block_shape_and_reject_mixed_ablation():
    _, rows, states = rows_and_states()
    dataset = SimpleNamespace(records=rows, states=states, metadata={"train_shapes": [[3, 4]]})
    predictions = [{"qa_id": row["qa_id"], "prediction": row["answer"], "terminated": True,
                    "block_shape": [2, 2], "input_mode": "disabled"} for row in rows]
    metrics = score_predictions(dataset, predictions)
    assert metrics["input_mode"] == "disabled" and metrics["diagnostic_only"]
    assert metrics["macro_task_answer_accuracy"] == 1
    with pytest.raises(ValueError, match="block shape disagrees"):
        score_predictions(dataset, predictions, block_shape=(1, 1))
    predictions[0]["input_mode"] = "correct"
    with pytest.raises(ValueError, match="mix input modes"):
        score_predictions(dataset, predictions)


def test_disabled_input_never_calls_field_encoder_and_clears_after_failure(tmp_path, monkeypatch):
    _, llm, sidecar, tokenizer, config = tiny_components()
    field, rows, states = rows_and_states()
    dataset = SimpleNamespace(records=rows, states=states, metadata={"train_shapes": [[3, 4]]})
    # A disabled evaluation must not even ask the dataset to decode another field.
    class Dataset:
        records, states, metadata = dataset.records, dataset.states, dataset.metadata

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            raise AssertionError("Disabled input unexpectedly read field tensors")

    calls = []
    hook = sidecar.memory.register_forward_hook(lambda *args: calls.append(True))
    config["evaluation"]["max_new_tokens"] = 3
    try:
        metrics = trainer.evaluate(llm, sidecar, tokenizer, Dataset(), torch.device("cpu"), torch.float32,
                                   config, tmp_path / "disabled.jsonl", input_mode="disabled")
        assert not calls
        assert metrics["field_memory_encodes"] == metrics["field_memory_reuses"] == 0
        assert metrics["diagnostic_only"] and sidecar._bound_state is None
        assert all(bridge._memory is None for bridge in sidecar.bridges)

        def fail(*args, **kwargs):
            raise RuntimeError("test decode failure")

        monkeypatch.setattr(trainer, "generate_from_prompt", fail)
        with pytest.raises(RuntimeError, match="test decode failure"):
            trainer.evaluate(llm, sidecar, tokenizer, Dataset(), torch.device("cpu"), torch.float32,
                             config, tmp_path / "failure.jsonl", input_mode="disabled")
        assert sidecar._bound_state is None
        assert all(bridge._memory is None for bridge in sidecar.bridges)
    finally:
        hook.remove()


def test_preflight_checks_every_answer_and_preserves_bounded_token_reuse():
    _, rows, _ = rows_and_states()
    tokenizer = CharacterTokenizer()
    training = {"max_prompt_tokens": 2048, "max_target_tokens": 96}
    cache = trainer.TrainingTokenCache(tokenizer, training, max_bytes=100000, max_entries=1)
    report = trainer.preflight_token_budgets(rows, tokenizer, training, token_cache=cache)
    assert report["questions"] == len(rows)
    assert report["status"] == "ok" and report["overflow_questions"] == 0
    assert report["max_target_tokens_observed"] > 1
    assert len(cache.entries) == 1 and cache.bytes <= cache.max_bytes
    malformed = {**rows[-1], "answer": json.dumps([1.2345] * 100)}
    with pytest.raises(trainer.TokenBudgetError, match="max_target_tokens") as exc:
        trainer.preflight_token_budgets([*rows, malformed], tokenizer, training, token_cache=cache)
    assert exc.value.report["questions"] == 3
    assert exc.value.report["target_overflow_count"] == 1
    assert exc.value.report["prompt_overflow_count"] == 0


def test_preflight_collects_all_overflows_and_extrema_without_caching_or_truncating_invalid_records():
    _, rows, _ = rows_and_states()
    tokenizer = CharacterTokenizer()
    ample = {"max_prompt_tokens": 10000, "max_target_tokens": 1000}
    base_prompt, base_answer = trainer.encode_training_record(rows[0], tokenizer, ample)
    training = {"max_prompt_tokens": len(base_prompt), "max_target_tokens": len(base_answer)}
    expanded = []
    for index in range(12):
        expanded.append({**rows[0], "qa_id": f"overflow-{index}", "template_family": "test-expression",
                         "question": rows[0]["question"] + " extra" * (index + 1),
                         "answer": json.dumps([1] * (index + 10))})
    prompt, answer = trainer.encode_training_record(expanded[-1], tokenizer, ample)
    cache = trainer.TrainingTokenCache(tokenizer, training, max_bytes=100000, max_entries=1)
    report = trainer.preflight_token_budgets([*expanded, rows[0]], tokenizer, training,
                                            token_cache=cache, raise_on_overflow=False)
    assert report["questions"] == 13 and report["status"] == "overflow"
    assert report["overflow_questions"] == report["prompt_overflow_count"] == report["target_overflow_count"] == 12
    assert len(report["overflow_examples"]) == 8
    assert report["max_prompt_tokens_observed"] == len(prompt)
    assert report["max_target_tokens_observed"] == len(answer)
    assert report["max_prompt_record"]["qa_id"] == report["max_target_record"]["qa_id"] == "overflow-11"
    assert report["max_prompt_record"]["task_type"] == "single_point"
    assert report["max_prompt_record"]["template_family"] == "test-expression"
    assert len(cache.entries) == 1 and cache.bytes <= cache.max_bytes
    for encode in (trainer.encode_training_record, lambda row, tok, settings: cache.get(row, tok, settings),
                   lambda row, tok, settings: trainer.training_tensors([row], tok, settings)):
        with pytest.raises(trainer.TokenBudgetError, match="truncation is forbidden") as exc:
            encode(expanded[-1], tokenizer, training)
        assert exc.value.qa_id == "overflow-11"
        assert (exc.value.prompt_tokens, exc.value.target_tokens) == (len(prompt), len(answer))
        assert (exc.value.prompt_limit, exc.value.target_limit) == tuple(training.values())


@pytest.mark.parametrize("fault", ["prefix", "eos", "tokenizer"])
def test_preflight_never_hides_tokenizer_or_template_errors_behind_budget_overflow(fault):
    _, rows, _ = rows_and_states()

    class InvalidTokenizer(CharacterTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            if not kwargs["add_generation_prompt"]:
                if fault == "prefix":
                    result[0] = 99
                elif fault == "eos":
                    result = result[:-2]
                else:
                    raise RuntimeError("tokenizer failed")
            return result

    with pytest.raises((ValueError, RuntimeError), match={"prefix": "prefix", "eos": "terminate", "tokenizer": "tokenizer failed"}[fault]) as exc:
        trainer.preflight_token_budgets(rows, InvalidTokenizer(),
            {"max_prompt_tokens": 1, "max_target_tokens": 1}, raise_on_overflow=False)
    assert not isinstance(exc.value, trainer.TokenBudgetError)


def test_token_check_only_audits_all_main_splits_and_never_loads_llm(tmp_path, monkeypatch):
    _, _, _, tokenizer, config = tiny_components()
    config["data"].update(qa_dir=str(tmp_path / "qa"), hdf5_path=str(tmp_path / "source.hdf5"))
    _, rows, _ = rows_and_states()
    opened, closed = [], []
    corrupt_train = False

    class Dataset:
        identity = "fixture"
        metadata = {"scoring": config["generation"], "tasks": ["single_point"]}

        def __init__(self, qa, source, split, **kwargs):
            self.split = split
            self.records = copy.deepcopy(rows[:1])
            if corrupt_train and split == "train":
                self.records[0]["question"] += " extra" * 500
                self.records[0]["answer"] = json.dumps([1.12345] * 100)
            opened.append(split)

        def __len__(self):
            return len(self.records)

        def audit(self):
            return {"questions": len(self)}

        def close(self):
            closed.append(self.split)

    def forbidden_model_load(*args, **kwargs):
        raise AssertionError("--token-check-only attempted to load Qwen weights")

    monkeypatch.setattr(trainer, "PointReadoutDataset", Dataset)
    monkeypatch.setattr(trainer, "load_config", lambda *args: copy.deepcopy(config))
    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", forbidden_model_load)
    command = ["train_point_readout.py", "--profile", "smoke", "--token-check-only", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", command + ["--output-dir", str(tmp_path / "checked")])
    trainer.main()
    report = json.loads((tmp_path / "checked" / "token_preflight.json").read_text())
    assert set(report) == {"train", "val", "test"}
    assert opened == closed == ["train", "val", "test"]
    corrupt_train = True
    for token_check_only in (True, False):
        destination = tmp_path / f"rejected_{token_check_only}"
        options = command if token_check_only else [arg for arg in command if arg != "--token-check-only"]
        monkeypatch.setattr("sys.argv", options + ["--output-dir", str(destination)])
        with pytest.raises(ValueError, match="max_target_tokens") as exc:
            trainer.main()
        assert "token_preflight.json" in str(exc.value)
        report = json.loads((destination / "token_preflight.json").read_text())
        expected_splits = ["train", "val", "test"] if token_check_only else ["train", "val"]
        assert set(report) == set(expected_splits)
        assert all(report[split]["questions"] == 1 for split in expected_splits)
        assert report["train"]["prompt_overflow_count"] == report["train"]["target_overflow_count"] == 1
        assert report["val"]["status"] == "ok"
        assert opened[-len(expected_splits):] == closed[-len(expected_splits):] == expected_splits


def test_geometry_only_preflights_shared_boundary_probes_without_loading_val_or_changing_manifest(tmp_path, monkeypatch):
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["data"].update(qa_dir=str(tmp_path / "qa"), hdf5_path=str(tmp_path / "source.hdf5"))
    shapes = [(3, 4), (2, 5), (1, 8), (8, 1), (1, 1), (6, 6), (4, 4)]
    all_rows = [make_record(str(index), torch.zeros(shape), "single_point", {"points": [[1, 1]]})
                for index, shape in enumerate(shapes)]
    selected_indices = trainer.geometry_probe_indices(all_rows)
    assert selected_indices == [5, 2, 3, 4]
    expected_ids = [all_rows[index]["qa_id"] for index in selected_indices]
    opened, audited, checked, closed = [], [], [], []

    class Dataset:
        identity = "full-immutable-manifest-identity"
        metadata = {"scoring": config["generation"], "tasks": ["single_point"]}

        def __init__(self, qa, source, split, **kwargs):
            assert split == "train", "Geometry-only unexpectedly constructed a validation dataset"
            self.records = all_rows
            opened.append(split)

        def __len__(self):
            return len(self.records)

        def audit(self):
            audited.extend(row["qa_id"] for row in self.records)
            return {"questions": len(self.records), "identity": self.identity}

        def close(self):
            closed.append(self.records is all_rows)

    def geometry_check(*args, **kwargs):
        assert kwargs["parallel"].world_size == 1
        dataset = args[3]
        checked.extend(row["qa_id"] for row in dataset.records)
        assert dataset.identity == Dataset.identity
        return {"status": "geometry_forward_backward_only", "optimizer_updates": 0}

    monkeypatch.setattr(trainer, "PointReadoutDataset", Dataset)
    monkeypatch.setattr(trainer, "load_config", lambda *args: copy.deepcopy(config))
    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", lambda *args: (llm, torch.float32))
    monkeypatch.setattr(trainer.core, "build_scratch_memory_components", lambda *args: (None, None, None, None))
    monkeypatch.setattr(trainer.core, "build_sidecar", lambda *args: (sidecar, {}))
    monkeypatch.setattr(trainer, "check_geometry", geometry_check)
    destination = tmp_path / "geometry"
    monkeypatch.setattr("sys.argv", ["train_point_readout.py", "--profile", "smoke", "--geometry-check-only",
                                     "--device", "cpu", "--output-dir", str(destination)])
    trainer.main()
    assert opened == ["train"] and audited == checked == expected_ids
    assert closed == [True] and len(all_rows) == len(shapes)
    scope = json.loads((destination / "geometry_preflight_scope.json").read_text())
    assert not scope["full_dataset_audited"]
    assert scope["total_train_questions"] == len(all_rows) and scope["checked_questions"] == 4
    assert scope["dataset_identity"] == Dataset.identity
    tokens = json.loads((destination / "token_preflight.json").read_text())
    assert set(tokens) == {"train"} and tokens["train"]["questions"] == 4
    contract = json.loads((destination / "contract.json").read_text())
    assert contract["dataset"] == Dataset.identity


@pytest.mark.parametrize("option", [["--eval-suite", "language"], ["--input-mode", "disabled"],
                                    ["--eval-suite", "field-pairs"]])
def test_diagnostic_cli_cannot_change_training_by_accident(tmp_path, monkeypatch, option):
    monkeypatch.setattr("sys.argv", ["train_point_readout.py", "--profile", "smoke", "--output-dir",
                                     str(tmp_path / "unused"), *option])
    with pytest.raises(SystemExit) as exc:
        trainer.main()
    assert exc.value.code == 2
    assert not (tmp_path / "unused").exists()


def test_matched_field_pairs_recompute_oracle_only_from_current_split_and_are_bounded():
    from tensor_compression.downstream.field_query_program import make_program_record
    from tensor_compression.downstream.point_readout import validate_record

    program = {"version": 1, "task": "single_point", "expression": {"op": "read",
               "selector": {"kind": "points", "points": [[1, 1]]}}}
    fields = {"a": torch.tensor([[-1., 0.], [0., 1.]]), "b": torch.tensor([[1., 0.], [0., -1.]])}
    rows = [make_program_record(name, field, "single_point", program, "direct")
            for name, field in fields.items()]
    class Base:
        split, identity = "test", "unchanged-training-contract"
        metadata = {"train_shapes": [[2, 2]], "scoring": {"answer_decimals": 1}}
        records = rows
        # An all-splits state dictionary must not cause this train-only source to be read.
        states = {name: {"field": "pressure", "split": "train" if name == "train-secret" else "test"}
                  for name in ["a", "b", "train-secret"]}

        def __getitem__(self, index):
            row = self.records[index]
            return row, fields[row["state_ref"]][None]

    paired = FieldPairDataset(Base(), max_pairs=8, max_candidates=1, seed=4)
    assert len(paired) == 2 and paired.selection["attempted_candidates"] == 1
    assert paired.identity == Base.identity
    first, second = paired.records
    assert first["question"] == second["question"] and first["grid_shape"] == second["grid_shape"]
    assert first["state_ref"] != second["state_ref"]
    assert _disjoint_reference_answers(first, second)
    for index in range(2):
        row, field = paired[index]
        validate_record(row, field[0])
    scored = [annotated(row, row["answer"]) for row in paired.records]
    result = summarize_readout_evaluation(scored, [[2, 2]])["paired_fields"]
    assert result["both_correct_rate"] == 1
    wrong = [annotated(row, "[99]") for row in paired.records]
    result = summarize_readout_evaluation(wrong, [[2, 2]])["paired_fields"]
    assert result["both_correct_rate"] == result["valid_changed_prediction_rate"] == 0
    # A repeated deterministic invocation selects identical pairs.
    assert FieldPairDataset(Base(), max_pairs=8, max_candidates=1, seed=4).records == paired.records
    fields["b"] = fields["a"].clone()
    Base.records = [make_program_record(name, field, "single_point", program, "direct")
                    for name, field in fields.items()]
    missing = FieldPairDataset(Base(), max_pairs=8, max_candidates=2)
    assert len(missing) == 0 and missing.selection["status"] == "unavailable"
    assert missing.selection["attempted_candidates"] <= 2


@pytest.mark.parametrize("splits", [("train", "train"), ("val", "test"), (None, None)])
def test_field_pairs_reject_non_eval_or_mixed_actual_state_splits_without_base_split_attribute(splits):
    base = SimpleNamespace(records=[{"state_ref": "a"}, {"state_ref": "b"}],
        states={name: {"split": split} for name, split in zip(("a", "b"), splits)})
    assert not hasattr(base, "split")
    with pytest.raises(ValueError, match="one validation or test split"):
        FieldPairDataset(base)


def test_field_pair_separation_checks_whole_coordinate_tie_sets_and_tolerance_intervals():
    from tensor_compression.downstream.field_query_program import make_program_record

    program = {"version": 1, "task": "region_argmax", "expression": {"op": "argmax",
               "selector": {"kind": "region", "start": [1, 1], "size": [2, 2]}}}
    a = make_program_record("a", torch.tensor([[1., 1.], [0., 0.]]), "region_argmax", program, "direct")
    b = make_program_record("b", torch.tensor([[0., 1.], [0., 1.]]), "region_argmax", program, "direct")
    c = make_program_record("c", torch.tensor([[0., 0.], [1., 1.]]), "region_argmax", program, "direct")
    assert a["oracle"]["values"] != b["oracle"]["values"]
    assert not _disjoint_reference_answers(a, b)  # A shared accepted coordinate remains.
    assert _disjoint_reference_answers(a, c)
    numeric = make_record("a", torch.tensor([[0., 1.]]), "single_point", {"points": [[1, 1]]})
    close = make_record("b", torch.tensor([[.3, 1.]]), "single_point", {"points": [[1, 1]]})
    far = make_record("c", torch.tensor([[.5, 1.]]), "single_point", {"points": [[1, 1]]})
    assert not _disjoint_reference_answers(numeric, close)
    assert _disjoint_reference_answers(numeric, far)
