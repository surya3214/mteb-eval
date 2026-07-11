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

# Fast multilingual Retrieval subset: single-lang / small BEIR-style tasks.
# Omits the multi-subset heavyweights that dominate wall-clock time.
RETRIEVAL_FAST_TASKS: tuple[str, ...] = (
    "ArguAna",
    "SCIDOCS",
    "AILAStatutes",
    "LegalBenchCorporateLobbying",
    "SpartQA",
    "TempReasonL1",
    "WinoGrande",
    "StackOverflowQA",
    "HagridRetrieval",
    "StatcanDialogueDatasetRetrieval",
    "TRECCOVID",
    "LEMBPasskeyRetrieval",
)

# Remaining MTEB(Multilingual, v2) Retrieval tasks not in the fast preset.
RETRIEVAL_FAST_OMITTED: tuple[str, ...] = (
    "BelebeleRetrieval",
    "MIRACLRetrievalHardNegatives",
    "WikipediaRetrievalMultilingual",
    "MLQARetrieval",
    "TwitterHjerneRetrieval",
    "CovidRetrieval",
)

TASK_PRESETS: dict[str, tuple[str, ...]] = {
    "retrieval-fast": RETRIEVAL_FAST_TASKS,
}


def manifest_path() -> Path:
    return Path(__file__).parent / "manifests" / MANIFEST_NAME


def load_manifest() -> dict:
    with manifest_path().open(encoding="utf-8") as f:
        return json.load(f)


def expected_task_names() -> list[str]:
    manifest = load_manifest()
    return [entry["name"] for entry in manifest["tasks"]]


def resolve_task_names(
    *,
    tasks: Sequence[str] | None = None,
    tasks_preset: str | None = None,
) -> tuple[list[str] | None, bool]:
    """Resolve explicit task names or a named preset.

    Returns:
        (task_names, from_preset). ``task_names`` is None when neither was set.
    """
    if tasks is not None and tasks_preset is not None:
        raise ValueError("Use either --tasks or --tasks-preset, not both.")
    if tasks_preset is not None:
        try:
            return list(TASK_PRESETS[tasks_preset]), True
        except KeyError as exc:
            known = ", ".join(sorted(TASK_PRESETS))
            raise ValueError(
                f"Unknown tasks preset {tasks_preset!r}. Known presets: {known}"
            ) from exc
    if tasks is not None:
        return list(tasks), False
    return None, False


def task_names_from_args(args: object) -> tuple[list[str] | None, bool]:
    """Read resolved task names from parsed CLI args."""
    return resolve_task_names(
        tasks=getattr(args, "tasks", None),
        tasks_preset=getattr(args, "tasks_preset", None),
    )


def add_task_arguments(parser: object) -> None:
    """Register --tasks / --tasks-preset (mutually exclusive)."""
    import argparse

    assert isinstance(parser, argparse.ArgumentParser)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional explicit task name subset.",
    )
    task_group.add_argument(
        "--tasks-preset",
        choices=sorted(TASK_PRESETS),
        default=None,
        help=(
            "Named task preset. retrieval-fast = 12 quick multilingual Retrieval "
            "tasks (skips Belebele/MIRACL/Wikipedia/MLQA/TwitterHjerne/Covid)."
        ),
    )


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
    allow_missing_task_names: bool = False,
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
            if allow_missing_task_names:
                logger.warning(
                    "Preset/task names not in benchmark %r with types %s: %s",
                    benchmark,
                    types,
                    sorted(missing),
                )
            else:
                raise ValueError(f"Unknown task name(s): {sorted(missing)}")
        if not tasks:
            raise ValueError(
                f"No tasks remain after applying task names {sorted(names)} "
                f"for benchmark {benchmark!r} with types {types}."
            )

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
