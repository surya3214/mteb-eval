"""Smoke tests for mteb_eval toolkit."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mteb_eval.cache import configure_cache
from mteb_eval.evaluate import _print_summary
from mteb_eval.model_loader import resolve_model_source, validate_local_checkpoint
from mteb_eval.tasks import expected_task_names, load_manifest, resolve_tasks, validate_against_manifest
from mteb.results.model_result import ModelResult
from mteb.results.task_result import TaskError, TaskResult


def _sample_task_result(name: str = "BIOSSES") -> TaskResult:
    return TaskResult(
        task_name=name,
        dataset_revision="rev",
        mteb_version="2.18.0",
        evaluation_time=1.0,
        scores={
            "test": [
                {
                    "main_score": 0.1,
                    "hf_subset": "default",
                    "languages": ["eng-Latn"],
                }
            ]
        },
    )


def test_manifest_has_19_tasks():
    manifest = load_manifest()
    assert manifest["expected_count"] == 19
    assert len(manifest["tasks"]) == 19


def test_resolve_sts_retrieval_task_count():
    tasks = resolve_tasks()
    assert len(tasks) == 19
    validate_against_manifest(tasks)


def test_expected_task_names_match_plan():
    names = set(expected_task_names())
    assert "FEVERHardNegatives" in names
    assert "HotpotQAHardNegatives" in names
    assert "BIOSSES" in names
    assert "STS22.v2" in names
    assert len(names) == 19


def test_cache_dir_sets_hf_env():
    with tempfile.TemporaryDirectory() as tmp:
        for key in (
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TRANSFORMERS_CACHE",
            "HF_DATASETS_OFFLINE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        ):
            os.environ.pop(key, None)

        cfg = configure_cache(cache_dir=tmp, offline=True)
        assert cfg.cache_dir == Path(tmp).resolve()
        assert os.environ["HF_HOME"] == str(Path(tmp).resolve())
        assert os.environ["HF_HUB_CACHE"].endswith("/hub")
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_default_cache_does_not_set_hf_home():
    saved = os.environ.get("HF_HOME")
    os.environ.pop("HF_HOME", None)
    try:
        cfg = configure_cache(default_cache=True)
        assert cfg.use_default_cache is True
        assert "HF_HOME" not in os.environ
    finally:
        if saved is not None:
            os.environ["HF_HOME"] = saved


def test_resolve_local_model_path(tmp_path: Path):
    model_dir = tmp_path / "mock-st"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (model_dir / "modules.json").write_text("{}")

    source = resolve_model_source(model_path=str(model_dir), model=None, hub_id=None)
    assert source.is_local is True
    assert source.path == str(model_dir.resolve())


def test_resolve_model_path_convenience_dir(tmp_path: Path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config_sentence_transformers.json").write_text("{}")

    source = resolve_model_source(model=str(model_dir), model_path=None, hub_id=None)
    assert source.is_local is True


def test_model_and_model_path_mutually_exclusive(tmp_path: Path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("{}")
    with pytest.raises(ValueError, match="not both"):
        resolve_model_source(model="org/model", model_path=str(model_dir), hub_id=None)


def test_missing_model_source_raises():
    with pytest.raises(ValueError, match="Provide --model"):
        resolve_model_source(model=None, model_path=None, hub_id=None)


def test_validate_local_checkpoint_st_layout(tmp_path: Path):
    model_dir = tmp_path / "st"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("{}")
    layout = validate_local_checkpoint(model_dir)
    assert layout.kind == "sentence-transformers"


def test_validate_local_checkpoint_transformers_layout(tmp_path: Path):
    model_dir = tmp_path / "tf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"")
    layout = validate_local_checkpoint(model_dir)
    assert layout.kind == "transformers"


def test_print_summary_includes_failed_tasks(capsys):
    task_result = MagicMock()
    task_result.task_name = "BIOSSES"
    task_result.get_score.return_value = 0.5
    results = SimpleNamespace(task_results=[task_result])
    _print_summary(
        results,
        {"BIOSSES": 1.0, "STS12": 0.5},
        failures={"STS12": "dataset not found"},
    )
    out = capsys.readouterr().out
    assert "BIOSSES" in out
    assert "0.5000" in out
    assert "STS12" in out
    assert "FAILED" in out
    assert "dataset not found" in out
    assert "Failed tasks: 1" in out


def test_continue_on_error_collects_failures(tmp_path: Path):
    from mteb_eval import evaluate as evaluate_module

    ok_task = MagicMock()
    ok_task.metadata.name = "BIOSSES"
    bad_task = MagicMock()
    bad_task.metadata.name = "STS12"

    ok_result = SimpleNamespace(
        task_results=[_sample_task_result("BIOSSES")],
        exceptions=None,
    )
    fail_result = SimpleNamespace(
        task_results=[],
        exceptions=[TaskError(task_name="STS12", exception="boom")],
    )

    def fake_evaluate(model, tasks, **kwargs):
        task = tasks[0]
        if task.metadata.name == "STS12":
            return fail_result
        return ok_result

    args = SimpleNamespace(
        cache_dir=None,
        default_cache=True,
        model="mteb/baseline-random-encoder",
        model_path=None,
        hub_id=None,
        model_type="auto",
        offline=False,
        benchmark="MTEB(eng, v2)",
        task_types=["STS"],
        tasks=None,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        batch_size=8,
        query_batch_size=None,
        corpus_batch_size=None,
        overwrite="only-missing",
        continue_on_error=True,
        verbose=False,
    )

    with (
        patch.object(evaluate_module, "build_parser") as mock_parser,
        patch.object(evaluate_module, "configure_cache"),
        patch.object(evaluate_module, "resolve_model_source") as mock_source,
        patch.object(evaluate_module, "load_embedding_model", return_value=MagicMock()),
        patch.object(evaluate_module, "resolve_tasks", return_value=[ok_task, bad_task]),
        patch("mteb.evaluate", side_effect=fake_evaluate),
    ):
        mock_parser.return_value.parse_args.return_value = args
        mock_source.return_value = SimpleNamespace(
            path="mteb/baseline-random-encoder",
            is_local=False,
            hub_id="mteb/baseline-random-encoder",
        )
        exit_code = evaluate_module.main([])

    assert exit_code == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["task_results"]) == 1
    assert summary["exceptions"][0]["task_name"] == "STS12"
