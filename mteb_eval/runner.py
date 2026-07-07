"""Shared MTEB evaluation loop used by single- and multi-GPU CLIs."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mteb_eval.cache import configure_cache
from mteb_eval.languages import languages_from_args
from mteb_eval.model_loader import (
    DEFAULT_MAX_SEQ_LEN,
    configure_max_seq_len,
    load_embedding_model,
    resolve_model_source,
)
from mteb_eval.prompts import configure_prompt_prefixes, print_task_prompts, resolve_task_prompts
from mteb_eval.tasks import resolve_tasks

if TYPE_CHECKING:
    from mteb.abstasks import AbsTask
    from mteb.results.model_result import ModelResult

logger = logging.getLogger(__name__)

RUN_META_FILENAME = "run_meta.json"


@dataclass
class EvalRunResult:
    """Outcome of a single evaluation run (one GPU / one process)."""

    model_result: ModelResult
    timings: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    task_prompts: dict[str, dict[str, str]] = field(default_factory=dict)
    exit_code: int = 0
    model_name: str = ""


def release_task_memory() -> None:
    """Best-effort release of GPU/CPU memory between benchmark tasks."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass


def build_encode_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "processing_kwargs": {
            "text": {
                "max_length": args.max_seq_len,
                "truncation": True,
            }
        },
    }
    if args.query_batch_size is not None:
        kwargs["query_batch_size"] = args.query_batch_size
    if args.corpus_batch_size is not None:
        kwargs["corpus_batch_size"] = args.corpus_batch_size
    return kwargs


def write_run_meta(
    output_dir: Path,
    *,
    timings: dict[str, float],
    failures: dict[str, str],
    task_prompts: dict[str, dict[str, str]],
    exit_code: int = 0,
) -> None:
    """Persist per-run metadata alongside summary.json for shard merging."""
    meta_path = output_dir / RUN_META_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timings": timings,
                "failures": failures,
                "task_prompts": task_prompts,
                "exit_code": exit_code,
            },
            f,
            indent=2,
        )


def run_evaluation(
    args: argparse.Namespace,
    *,
    task_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> EvalRunResult:
    """Run MTEB evaluation for the given args and optional task-name subset."""
    configure_cache(
        cache_dir=args.cache_dir,
        default_cache=args.default_cache,
        offline=args.offline,
    )

    source = resolve_model_source(
        model=args.model,
        model_path=args.model_path,
        hub_id=args.hub_id,
    )
    if source.hub_id is None and args.hub_id:
        source = type(source)(path=source.path, is_local=source.is_local, hub_id=args.hub_id)

    import mteb
    from mteb.cache import ResultCache
    from mteb.results.model_result import ModelResult

    resolved_output = output_dir or Path(args.output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s (local=%s)", source.path, source.is_local)
    model = load_embedding_model(
        source,
        model_type=args.model_type,
        device=args.device,
    )
    configure_max_seq_len(model, args.max_seq_len)
    configure_prompt_prefixes(
        model,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    logger.info("Using max sequence length: %d", args.max_seq_len)

    languages = languages_from_args(args)
    if languages:
        logger.info(
            "Language filter: %d code(s), exclusive=%s",
            len(languages),
            args.exclusive_language_filter,
        )

    names = task_names if task_names is not None else args.tasks
    tasks: list[AbsTask] = resolve_tasks(
        benchmark=args.benchmark,
        task_types=args.task_types,
        task_names=names,
        languages=languages,
        exclusive_language_filter=args.exclusive_language_filter,
    )
    logger.info("Evaluating %d task(s)...", len(tasks))

    encode_kwargs = build_encode_kwargs(args)
    result_cache = ResultCache(cache_path=resolved_output)

    timings: dict[str, float] = {}
    all_task_results = []
    all_exceptions = []
    failures: dict[str, str] = {}
    task_prompts: dict[str, dict[str, str]] = {}
    start_all = time.perf_counter()

    for task in tasks:
        t0 = time.perf_counter()
        prompts = resolve_task_prompts(model, task)
        task_prompts[task.metadata.name] = prompts
        print_task_prompts(task.metadata.name, prompts)
        logger.info("Running task: %s", task.metadata.name)
        task_result = mteb.evaluate(
            model,
            [task],
            encode_kwargs=encode_kwargs,
            cache=result_cache,
            overwrite_strategy=args.overwrite,
            show_progress_bar=not args.verbose,
            raise_error=not args.continue_on_error,
        )
        timings[task.metadata.name] = time.perf_counter() - t0
        all_task_results.extend(task_result.task_results)
        if task_result.exceptions:
            all_exceptions.extend(task_result.exceptions)
            for err in task_result.exceptions:
                failures[err.task_name] = err.exception
                logger.error("Task %s failed: %s", err.task_name, err.exception)

        release_task_memory()

    elapsed_all = time.perf_counter() - start_all
    if failures:
        logger.warning(
            "Finished with %d failed task(s) in %.1fs",
            len(failures),
            elapsed_all,
        )
    else:
        logger.info("All tasks finished in %.1fs", elapsed_all)

    model_name = source.hub_id or source.path
    combined = ModelResult(
        model_name=model_name,
        model_revision=None,
        task_results=all_task_results,
        exceptions=all_exceptions or None,
    )

    summary_path = resolved_output / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined.model_dump(), f, indent=2, default=str)
    logger.info("Wrote summary to %s", summary_path)

    write_run_meta(
        resolved_output,
        timings=timings,
        failures=failures,
        task_prompts=task_prompts,
        exit_code=1 if failures else 0,
    )

    return EvalRunResult(
        model_result=combined,
        timings=timings,
        failures=failures,
        task_prompts=task_prompts,
        exit_code=1 if failures else 0,
        model_name=model_name,
    )
