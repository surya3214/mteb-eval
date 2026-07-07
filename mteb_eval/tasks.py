"""MTEB(eng, v2) STS + Retrieval task resolution."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import mteb
    from mteb.abstasks import AbsTask

logger = logging.getLogger(__name__)

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


def filter_tasks_by_languages(
    tasks: list["AbsTask"],
    languages: Sequence[str],
    *,
    exclusive_language_filter: bool = False,
) -> list["AbsTask"]:
    """Filter each task's hf_subsets to the given languages; skip tasks with no match."""
    filtered: list[AbsTask] = []
    for task in tasks:
        task_copy = deepcopy(task)
        try:
            task_copy.filter_languages(
                languages,
                exclusive_language_filter=exclusive_language_filter,
            )
        except ValueError:
            logger.warning(
                "Skipping task %s: no subsets match languages %s",
                task.metadata.name,
                list(languages),
            )
            continue
        logger.info(
            "Task %s: %d subset(s) after language filter",
            task_copy.metadata.name,
            len(task_copy.hf_subsets),
        )
        filtered.append(task_copy)
    return filtered


def resolve_tasks(
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    task_types: Iterable[str] | None = None,
    task_names: Iterable[str] | None = None,
    languages: Sequence[str] | None = None,
    exclusive_language_filter: bool = False,
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

    if languages:
        tasks = filter_tasks_by_languages(
            tasks,
            languages,
            exclusive_language_filter=exclusive_language_filter,
        )
        if not tasks:
            raise ValueError(
                f"No tasks remain after language filter {list(languages)} "
                f"for benchmark {benchmark!r}."
            )

    return tasks


def partition_task_names(task_names: Sequence[str], num_partitions: int) -> list[list[str]]:
    """Split task names round-robin across partitions (deterministic)."""
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
    names = sorted(task_names)
    partitions: list[list[str]] = [[] for _ in range(num_partitions)]
    for idx, name in enumerate(names):
        partitions[idx % num_partitions].append(name)
    return partitions


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
