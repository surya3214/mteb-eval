"""Smoke tests for mteb_eval toolkit."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from mteb_eval.cache import configure_cache
from mteb_eval.model_loader import resolve_model_source, validate_local_checkpoint
from mteb_eval.tasks import expected_task_names, load_manifest, resolve_tasks, validate_against_manifest


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
