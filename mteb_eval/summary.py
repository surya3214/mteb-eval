"""Evaluation summary formatting and CSV export."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from mteb_eval.runner import RUN_META_FILENAME, EvalRunResult

logger = logging.getLogger(__name__)


def build_summary_rows(
    results: Any,
    timings: dict[str, float],
    *,
    failures: dict[str, str] | None = None,
    task_prompts: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str | float | None]]:
    """Build tabular summary rows for console and CSV output."""
    failures = failures or {}
    task_prompts = task_prompts or {}
    rows: list[dict[str, str | float | None]] = []

    seen: set[str] = set()
    for task_result in results.task_results:
        name = task_result.task_name
        seen.add(name)
        prompts = task_prompts.get(name, {})
        rows.append(
            {
                "task": name,
                "score": task_result.get_score(),
                "time_s": timings.get(name, 0.0),
                "status": "ok",
                "query_prompt": prompts.get("query", ""),
                "document_prompt": prompts.get("document", ""),
                "error": "",
            }
        )

    for name, error in failures.items():
        if name in seen:
            continue
        prompts = task_prompts.get(name, {})
        rows.append(
            {
                "task": name,
                "score": None,
                "time_s": timings.get(name, 0.0),
                "status": "failed",
                "query_prompt": prompts.get("query", ""),
                "document_prompt": prompts.get("document", ""),
                "error": error,
            }
        )

    return rows


def _average_row(rows: list[dict[str, str | float | None]]) -> dict[str, str | float | None]:
    scores = [float(row["score"]) for row in rows if row["score"] is not None]
    times = [float(row["time_s"]) for row in rows]
    return {
        "task": "AVERAGE",
        "score": sum(scores) / len(scores) if scores else None,
        "time_s": sum(times) / len(times) if times else 0.0,
        "status": "summary",
        "query_prompt": "",
        "document_prompt": "",
        "error": "",
    }


def iter_subset_score_entries(results: Any) -> list[dict[str, Any]]:
    """Flatten per-subset / per-language score entries from ModelResult task_results."""
    entries: list[dict[str, Any]] = []
    for task_result in results.task_results:
        task_name = task_result.task_name
        scores = getattr(task_result, "scores", None) or {}
        if isinstance(scores, dict):
            split_items = scores.items()
        else:
            continue
        for split, split_scores in split_items:
            if not isinstance(split_scores, list):
                continue
            for entry in split_scores:
                if not isinstance(entry, dict):
                    # pydantic ScoreDict-like
                    entry = dict(entry) if hasattr(entry, "items") else {}
                main = entry.get("main_score")
                if main is None:
                    continue
                hf_subset = entry.get("hf_subset", "default")
                langs = entry.get("languages") or []
                if not langs:
                    langs = [str(hf_subset)]
                for lang in langs:
                    entries.append(
                        {
                            "task": task_name,
                            "split": split,
                            "hf_subset": hf_subset,
                            "language": lang,
                            "score": float(main),
                        }
                    )
    return entries


def build_language_summary_rows(results: Any) -> list[dict[str, Any]]:
    """Aggregate main_score by language across all tasks/subsets."""
    from collections import defaultdict

    by_lang: dict[str, list[float]] = defaultdict(list)
    tasks_by_lang: dict[str, set[str]] = defaultdict(set)
    for entry in iter_subset_score_entries(results):
        lang = str(entry["language"])
        by_lang[lang].append(float(entry["score"]))
        tasks_by_lang[lang].add(str(entry["task"]))

    rows: list[dict[str, Any]] = []
    for lang in sorted(by_lang):
        scores = by_lang[lang]
        rows.append(
            {
                "language": lang,
                "mean_score": sum(scores) / len(scores),
                "n_scores": len(scores),
                "n_tasks": len(tasks_by_lang[lang]),
            }
        )
    return rows


def write_language_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write language-aggregated summary CSV with an AVERAGE row."""
    fieldnames = ["language", "mean_score", "n_scores", "n_tasks"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "language": row["language"],
                    "mean_score": f"{float(row['mean_score']):.6f}",
                    "n_scores": int(row["n_scores"]),
                    "n_tasks": int(row["n_tasks"]),
                }
            )
        if rows:
            all_scores = [float(r["mean_score"]) for r in rows]
            writer.writerow(
                {
                    "language": "AVERAGE",
                    "mean_score": f"{sum(all_scores) / len(all_scores):.6f}",
                    "n_scores": sum(int(r["n_scores"]) for r in rows),
                    "n_tasks": "",
                }
            )


def write_language_detail_csv(path: Path, results: Any) -> None:
    """Write per-task × language × subset score rows."""
    fieldnames = ["task", "split", "hf_subset", "language", "score"]
    entries = iter_subset_score_entries(results)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "task": entry["task"],
                    "split": entry["split"],
                    "hf_subset": entry["hf_subset"],
                    "language": entry["language"],
                    "score": f"{float(entry['score']):.6f}",
                }
            )


def print_language_summary(results: Any) -> None:
    """Print a compact language-wise score table."""
    rows = build_language_summary_rows(results)
    if not rows:
        return
    print("\n" + "=" * 64)
    print(f"{'Language':<20} {'Mean score':>12} {'n_scores':>10} {'n_tasks':>8}")
    print("-" * 64)
    for row in rows:
        print(
            f"{row['language']:<20} {float(row['mean_score']):>12.4f} "
            f"{int(row['n_scores']):>10} {int(row['n_tasks']):>8}"
        )
    means = [float(r["mean_score"]) for r in rows]
    print("-" * 64)
    print(
        f"{'AVERAGE':<20} {sum(means) / len(means):>12.4f} "
        f"{sum(int(r['n_scores']) for r in rows):>10}"
    )
    print("=" * 64)


def write_language_outputs(output_dir: Path, results: Any) -> tuple[Path, Path] | None:
    """Write language aggregate + detail CSVs; return paths or None if no subset scores."""
    lang_rows = build_language_summary_rows(results)
    if not lang_rows:
        logger.info("No per-language subset scores found; skipping language CSVs")
        return None
    lang_path = output_dir / "summary_by_language.csv"
    detail_path = output_dir / "summary_by_language_detail.csv"
    write_language_summary_csv(lang_path, lang_rows)
    write_language_detail_csv(detail_path, results)
    logger.info("Wrote language summary CSV to %s", lang_path)
    logger.info("Wrote language detail CSV to %s", detail_path)
    return lang_path, detail_path


def write_summary_csv(
    path: Path,
    rows: list[dict[str, str | float | None]],
    *,
    include_average: bool = True,
) -> None:
    """Write summary rows to CSV, optionally appending an average row."""
    fieldnames = [
        "task",
        "score",
        "time_s",
        "status",
        "query_prompt",
        "document_prompt",
        "error",
    ]
    output_rows = list(rows)
    if include_average and rows:
        output_rows.append(_average_row(rows))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "score": (
                        f"{row['score']:.6f}"
                        if row["score"] is not None and row["task"] != "AVERAGE"
                        else (
                            f"{row['score']:.6f}"
                            if row["score"] is not None
                            else ""
                        )
                    ),
                    "time_s": f"{float(row['time_s']):.3f}",
                    "status": row["status"],
                    "query_prompt": row.get("query_prompt", ""),
                    "document_prompt": row.get("document_prompt", ""),
                    "error": row.get("error", ""),
                }
            )


def print_summary(
    results: Any,
    timings: dict[str, float],
    *,
    failures: dict[str, str] | None = None,
    task_prompts: dict[str, dict[str, str]] | None = None,
) -> None:
    """Print a human-readable summary table to stdout."""
    failures = failures or {}
    rows = build_summary_rows(
        results,
        timings,
        failures=failures,
        task_prompts=task_prompts,
    )

    print("\n" + "=" * 72)
    print(f"{'Task':<35} {'Score':>12} {'Time (s)':>10}")
    print("-" * 72)

    for row in rows:
        score = row["score"]
        score_str = f"{float(score):.4f}" if score is not None else "FAILED"
        print(f"{row['task']:<35} {score_str:>12} {float(row['time_s']):>10.1f}")
        if row["status"] == "failed":
            print(f"  error: {row['error']}")

    if rows:
        avg = _average_row(rows)
        print("-" * 72)
        avg_score = avg["score"]
        avg_score_str = f"{float(avg_score):.4f}" if avg_score is not None else "n/a"
        print(f"{'AVERAGE':<35} {avg_score_str:>12} {float(avg['time_s']):>10.1f}")

    print("=" * 72)
    total = sum(float(row["time_s"]) for row in rows)
    print(f"Total evaluation time: {total:.1f}s ({total / 60:.1f} min)")
    if failures:
        print(f"Failed tasks: {len(failures)}")

    if task_prompts:
        print("\nPrompts used per task:")
        for row in rows:
            name = str(row["task"])
            prompts = task_prompts.get(name)
            if not prompts:
                continue
            print(f"  {name}")
            print(f"    query:    {prompts.get('query', '')}")
            print(f"    document: {prompts.get('document', '')}")


def _load_shard_run(shard_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = shard_dir / "summary.json"
    meta_path = shard_dir / RUN_META_FILENAME
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing shard summary: {summary_path}")
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    meta: dict[str, Any] = {"timings": {}, "failures": {}, "task_prompts": {}}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    return summary, meta


def _copy_result_cache_files(shard_dir: Path, output_dir: Path) -> None:
    """Copy MTEB per-task result JSON files from a shard into the merged output dir."""
    for path in shard_dir.rglob("*.json"):
        if path.name in {"summary.json", RUN_META_FILENAME, "model_meta.json"}:
            continue
        if path.name == "run_settings.jsonl":
            continue
        rel = path.relative_to(shard_dir)
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def merge_shard_results(shard_dirs: list[Path], output_dir: Path) -> EvalRunResult:
    """Merge parallel GPU shard outputs into a single EvalRunResult and unified cache."""
    from mteb.results.model_result import ModelResult
    from mteb.results.task_result import TaskError, TaskResult

    if not shard_dirs:
        raise ValueError("merge_shard_results requires at least one shard directory")

    all_task_results: list[TaskResult] = []
    all_exceptions: list[TaskError] = []
    timings: dict[str, float] = {}
    failures: dict[str, str] = {}
    task_prompts: dict[str, dict[str, str]] = {}
    model_name = ""
    shard_exit_codes: list[int] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_dir in shard_dirs:
        summary, meta = _load_shard_run(shard_dir)
        if not model_name:
            model_name = summary.get("model_name", "")
        shard_exit_codes.append(int(meta.get("exit_code", 0)))

        for raw in summary.get("task_results", []):
            all_task_results.append(TaskResult.model_validate(raw))
        for raw in summary.get("exceptions") or []:
            all_exceptions.append(TaskError.model_validate(raw))

        timings.update(meta.get("timings", {}))
        failures.update(meta.get("failures", {}))
        task_prompts.update(meta.get("task_prompts", {}))

        _copy_result_cache_files(shard_dir, output_dir)

    combined = ModelResult(
        model_name=model_name,
        model_revision=None,
        task_results=all_task_results,
        exceptions=all_exceptions or None,
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined.model_dump(), f, indent=2, default=str)

    meta_path = output_dir / RUN_META_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timings": timings,
                "failures": failures,
                "task_prompts": task_prompts,
            },
            f,
            indent=2,
        )

    exit_code = 1 if failures or any(code != 0 for code in shard_exit_codes) else 0
    return EvalRunResult(
        model_result=combined,
        timings=timings,
        failures=failures,
        task_prompts=task_prompts,
        exit_code=exit_code,
        model_name=model_name,
    )
