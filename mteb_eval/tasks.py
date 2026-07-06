"""MTEB(eng, v2) STS + Retrieval task resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import mteb
    from mteb.abstasks import AbsTask

DEFAULT_BENCHMARK = "MTEB(eng, v2)"
DEFAULT_TASK_TYPES = ("STS", "Retrieval")
MANIFEST_NAME = "eng_v2_sts_retrieval.json"


def manifest_path() -> Path:
    return Path(__file__).parent / "manifests" / MANIFEST_NAME


def load_manifest() -> dict:
    with manifest_path().open(encoding="utf-8") as f:
        return json.load(f)


def expected_task_names() -> list[str]:
    manifest = load_manifest()
    return [entry["name"] for entry in manifest["tasks"]]


def resolve_tasks(
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    task_types: Iterable[str] | None = None,
    task_names: Iterable[str] | None = None,
) -> list["AbsTask"]:
    """Resolve MTEB tasks from benchmark, optionally filtered by type and/or name."""
    import mteb

    bench = mteb.get_benchmark(benchmark)
    types = list(task_types) if task_types is not None else list(DEFAULT_TASK_TYPES)
    tasks = list(mteb.filter_tasks(bench, task_types=types))

    if task_names is not None:
        names = set(task_names)
        tasks = [t for t in tasks if t.metadata.name in names]
        missing = names - {t.metadata.name for t in tasks}
        if missing:
            raise ValueError(f"Unknown task name(s): {sorted(missing)}")

    return tasks


def validate_against_manifest(tasks: list["AbsTask"]) -> None:
    """Ensure resolved tasks match the shipped manifest."""
    expected = set(expected_task_names())
    actual = {t.metadata.name for t in tasks}
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"extra: {sorted(extra)}")
        raise ValueError("Task set does not match manifest — " + "; ".join(parts))


def dataset_info(task: "AbsTask") -> dict:
    """Extract HF dataset path and revision from task metadata."""
    ds = task.metadata.dataset
    if isinstance(ds, dict):
        return {
            "path": ds.get("path", ""),
            "revision": ds.get("revision"),
        }
    return {"path": str(ds), "revision": None}
