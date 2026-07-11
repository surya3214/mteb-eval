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
from mteb_eval.languages import ML16_LANGUAGES, resolve_languages
from mteb_eval.runner import release_task_memory
from mteb_eval.model_loader import (
    build_hf_model_kwargs,
    configure_max_seq_len,
    resolve_model_source,
    resolve_torch_dtype,
    validate_local_checkpoint,
)
from mteb_eval.prompts import configure_prompt_prefixes
from mteb_eval.summary import (
    build_summary_rows,
    merge_shard_results,
    print_summary,
    write_summary_csv,
)
from mteb_eval.tasks import (
    RETRIEVAL_FAST_OMITTED,
    RETRIEVAL_FAST_TASKS,
    expected_task_names,
    filter_tasks_by_languages,
    load_manifest,
    partition_task_names,
    resolve_task_names,
    resolve_tasks,
    validate_against_manifest,
)
from mteb.results.model_result import ModelResult
from mteb.results.task_result import TaskResult


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
    print_summary(
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
    assert "AVERAGE" in out
    assert "Failed tasks: 1" in out


def test_continue_on_error_collects_failures(tmp_path: Path):
    from mteb_eval.runner import EvalRunResult

    ok_task = MagicMock()
    ok_task.metadata.name = "BIOSSES"
    bad_task = MagicMock()
    bad_task.metadata.name = "STS12"

    ok_result = EvalRunResult(
        model_result=ModelResult(
            model_name="mteb/baseline-random-encoder",
            model_revision=None,
            task_results=[_sample_task_result("BIOSSES")],
            exceptions=None,
        ),
        timings={"BIOSSES": 1.0, "STS12": 0.5},
        failures={"STS12": "boom"},
        task_prompts={},
        exit_code=1,
        model_name="mteb/baseline-random-encoder",
    )

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
        tasks_preset=None,
        languages=None,
        languages_preset=None,
        exclusive_language_filter=False,
        output_dir=str(tmp_path / "out"),
        device="cpu",
        dtype="auto",
        attn_implementation=None,
        batch_size=8,
        query_batch_size=None,
        corpus_batch_size=None,
        overwrite="only-missing",
        continue_on_error=True,
        max_seq_len=512,
        query_prefix=None,
        document_prefix=None,
        verbose=False,
    )

    with patch("mteb_eval.evaluate.run_evaluation", return_value=ok_result) as mock_run, patch(
        "mteb_eval.evaluate.build_parser"
    ) as mock_parser:
        from mteb_eval import evaluate as evaluate_module

        mock_parser.return_value.parse_args.return_value = args
        exit_code = evaluate_module.main([])

    assert exit_code == 1
    mock_run.assert_called_once()
    assert (tmp_path / "out" / "summary.csv").exists()


def test_write_summary_csv_includes_average(tmp_path: Path):
    rows = [
        {
            "task": "STS12",
            "score": 0.8,
            "time_s": 2.0,
            "status": "ok",
            "query_prompt": "q1",
            "document_prompt": "d1",
            "error": "",
        },
        {
            "task": "STS13",
            "score": 0.6,
            "time_s": 4.0,
            "status": "ok",
            "query_prompt": "q2",
            "document_prompt": "d2",
            "error": "",
        },
    ]
    csv_path = tmp_path / "summary.csv"
    write_summary_csv(csv_path, rows)
    content = csv_path.read_text(encoding="utf-8")
    assert "AVERAGE" in content
    assert "0.700000" in content
    assert "3.000" in content


def test_configure_prompt_prefixes_on_sentence_transformer():
    st_model = SimpleNamespace(prompts={"query": "old-q", "document": "old-d"})
    wrapper = SimpleNamespace(model=st_model, model_prompts={"query": "old-q"})

    configure_prompt_prefixes(
        wrapper,
        query_prefix="query: ",
        document_prefix="document: ",
    )
    assert st_model.prompts["query"] == "query: "
    assert st_model.prompts["document"] == "document: "
    assert wrapper.model_prompts["query"] == "query: "
    assert wrapper.model_prompts["document"] == "document: "


def test_configure_max_seq_len_sentence_transformer():
    st_model = SimpleNamespace(max_seq_length=8192)
    wrapper = SimpleNamespace(model=st_model)

    configure_max_seq_len(wrapper, 512)
    assert st_model.max_seq_length == 512


def test_configure_max_seq_len_eurobert_wrapper():
    from mteb_eval.model_loader import EuroBertEncoderWrapper

    model = EuroBertEncoderWrapper(model=MagicMock(), tokenizer=MagicMock(), device="cpu")
    model.max_length = 2048
    configure_max_seq_len(model, 512)
    assert model.max_length == 512


def test_release_task_memory_runs():
    release_task_memory()


def test_ml16_preset_has_16_codes():
    assert len(ML16_LANGUAGES) == 16
    assert "eng-Latn" in ML16_LANGUAGES
    assert "jpn-Jpan" in ML16_LANGUAGES
    assert "zho-Hans" in ML16_LANGUAGES


def test_resolve_languages_preset():
    langs = resolve_languages(languages_preset="ml16")
    assert langs == list(ML16_LANGUAGES)


def test_resolve_languages_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_languages(languages=["eng-Latn"], languages_preset="ml16")


def test_filter_tasks_by_languages_sts17():
    import mteb

    task = mteb.get_task("STS17")
    before = len(task.hf_subsets)
    filtered = filter_tasks_by_languages([task], list(ML16_LANGUAGES))
    assert len(filtered) == 1
    assert len(filtered[0].hf_subsets) == before


def test_filter_skips_no_overlap_task():
    import mteb

    task = mteb.get_task("TwitterHjerneRetrieval")
    filtered = filter_tasks_by_languages([task], list(ML16_LANGUAGES))
    assert filtered == []


def test_partition_task_names_round_robin():
    names = ["C", "A", "B", "D", "E"]
    parts = partition_task_names(names, 3)
    assert parts == [["A", "D"], ["B", "E"], ["C"]]


def test_merge_shard_results(tmp_path: Path):
    shard_a = tmp_path / "shard0"
    shard_b = tmp_path / "shard1"
    shard_a.mkdir()
    shard_b.mkdir()

    result_a = _sample_task_result("STS12")
    result_b = _sample_task_result("STS13")
    model_dump_a = {
        "model_name": "test/model",
        "model_revision": None,
        "task_results": [result_a.model_dump()],
        "exceptions": None,
    }
    model_dump_b = {
        "model_name": "test/model",
        "model_revision": None,
        "task_results": [result_b.model_dump()],
        "exceptions": None,
    }
    (shard_a / "summary.json").write_text(json.dumps(model_dump_a), encoding="utf-8")
    (shard_b / "summary.json").write_text(json.dumps(model_dump_b), encoding="utf-8")
    (shard_a / "run_meta.json").write_text(
        json.dumps({"timings": {"STS12": 1.0}, "failures": {}, "task_prompts": {}, "exit_code": 0}),
        encoding="utf-8",
    )
    (shard_b / "run_meta.json").write_text(
        json.dumps({"timings": {"STS13": 2.0}, "failures": {}, "task_prompts": {}, "exit_code": 0}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "merged"
    merged = merge_shard_results([shard_a, shard_b], output_dir)
    assert len(merged.model_result.task_results) == 2
    assert merged.timings["STS12"] == 1.0
    assert merged.timings["STS13"] == 2.0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.csv").exists() is False  # CSV written by caller


def test_retrieval_fast_preset_membership():
    assert len(RETRIEVAL_FAST_TASKS) == 12
    assert len(RETRIEVAL_FAST_OMITTED) == 6
    assert set(RETRIEVAL_FAST_TASKS).isdisjoint(RETRIEVAL_FAST_OMITTED)
    assert "ArguAna" in RETRIEVAL_FAST_TASKS
    assert "BelebeleRetrieval" in RETRIEVAL_FAST_OMITTED
    assert "MIRACLRetrievalHardNegatives" in RETRIEVAL_FAST_OMITTED


def test_resolve_task_names_preset():
    names, from_preset = resolve_task_names(tasks_preset="retrieval-fast")
    assert from_preset is True
    assert names == list(RETRIEVAL_FAST_TASKS)


def test_resolve_task_names_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        resolve_task_names(tasks=["ArguAna"], tasks_preset="retrieval-fast")


def test_resolve_tasks_retrieval_fast_on_multilingual():
    tasks = resolve_tasks(
        benchmark="MTEB(Multilingual, v2)",
        task_types=["Retrieval"],
        task_names=list(RETRIEVAL_FAST_TASKS),
        allow_missing_task_names=True,
    )
    names = {t.metadata.name for t in tasks}
    assert names == set(RETRIEVAL_FAST_TASKS)
    assert names.isdisjoint(RETRIEVAL_FAST_OMITTED)


def test_build_hf_model_kwargs_auto_empty():
    assert build_hf_model_kwargs(dtype="auto") == {}


def test_build_hf_model_kwargs_bfloat16():
    import torch

    kwargs = build_hf_model_kwargs(dtype="bfloat16")
    assert kwargs == {"torch_dtype": torch.bfloat16}


def test_build_hf_model_kwargs_attn_only_when_set():
    import torch

    assert "attn_implementation" not in build_hf_model_kwargs(dtype="auto")
    kwargs = build_hf_model_kwargs(dtype="float16", attn_implementation="sdpa")
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["attn_implementation"] == "sdpa"


def test_resolve_torch_dtype_auto_is_none():
    assert resolve_torch_dtype("auto") is None


def test_load_embedding_model_passes_nested_model_kwargs():
    from mteb_eval.model_loader import ModelSource, load_embedding_model

    source = ModelSource(path="org/demo-model", is_local=False, hub_id="org/demo-model")
    with (
        patch("mteb.get_model_meta", return_value=None),
        patch("sentence_transformers.SentenceTransformer") as mock_st,
    ):
        mock_st.return_value = MagicMock()
        load_embedding_model(source, dtype="bfloat16", attn_implementation="sdpa")

    _, kwargs = mock_st.call_args
    import torch

    assert kwargs["model_kwargs"]["torch_dtype"] == torch.bfloat16
    assert kwargs["model_kwargs"]["attn_implementation"] == "sdpa"
    assert "torch_dtype" not in kwargs
