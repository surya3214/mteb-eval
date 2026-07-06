"""Evaluation summary formatting and CSV export."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


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
