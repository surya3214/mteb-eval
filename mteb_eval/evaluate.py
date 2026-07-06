"""Evaluate embedding models on MTEB(eng, v2) STS + Retrieval tasks."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

from mteb_eval.cache import configure_cache
from mteb_eval.model_loader import (
    DEFAULT_MAX_SEQ_LEN,
    configure_max_seq_len,
    load_embedding_model,
    resolve_model_source,
)
from mteb_eval.prompts import configure_prompt_prefixes, print_task_prompts, resolve_task_prompts
from mteb_eval.summary import build_summary_rows, print_summary, write_summary_csv
from mteb_eval.tasks import resolve_tasks

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an embedding model on MTEB STS + Retrieval tasks.",
    )

    cache = parser.add_mutually_exclusive_group(required=True)
    cache.add_argument("--cache-dir", type=str, help="Portable HF cache root.")
    cache.add_argument(
        "--default-cache",
        action="store_true",
        help="Use system default ~/.cache/huggingface.",
    )

    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", type=str, help="Hugging Face Hub model id.")
    model.add_argument(
        "--model-path",
        type=str,
        help="Absolute path to a local model checkpoint folder.",
    )

    parser.add_argument(
        "--hub-id",
        type=str,
        default=None,
        help="Canonical Hub id for local Qwen3 checkpoints (MTEB instruct wrapper).",
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "qwen3", "harrier", "sentence-transformer", "eurobert-base"],
        default="auto",
        help="Model family preset (default: auto).",
    )
    parser.add_argument("--offline", action="store_true", help="Enable HF offline mode.")
    parser.add_argument(
        "--benchmark",
        default="MTEB(eng, v2)",
        help="MTEB benchmark (default: MTEB(eng, v2)).",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["STS", "Retrieval"],
        help="Task types to evaluate.",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task subset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for MTEB result cache and summary JSON.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps).")
    parser.add_argument("--batch-size", type=int, default=32, help="Default encode batch size.")
    parser.add_argument("--query-batch-size", type=int, default=None, help="Query batch size.")
    parser.add_argument(
        "--corpus-batch-size",
        type=int,
        default=None,
        help="Corpus batch size (use 1-4 for large decoder embedders).",
    )
    parser.add_argument(
        "--overwrite",
        choices=["only-missing", "always", "never"],
        default="only-missing",
        help="MTEB result cache overwrite strategy.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log task failures and continue with remaining tasks (default: stop on first error).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Maximum input sequence length / truncation limit (default: {DEFAULT_MAX_SEQ_LEN}).",
    )
    parser.add_argument(
        "--query-prefix",
        type=str,
        default=None,
        help="Optional prefix prepended to queries (SentenceTransformer prompts['query']).",
    )
    parser.add_argument(
        "--document-prefix",
        type=str,
        default=None,
        help="Optional prefix prepended to documents/passages (prompts['document']).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _release_task_memory() -> None:
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


def _build_encode_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    tasks = resolve_tasks(
        benchmark=args.benchmark,
        task_types=args.task_types,
        task_names=args.tasks,
    )
    logger.info("Evaluating %d task(s)...", len(tasks))

    encode_kwargs = _build_encode_kwargs(args)
    result_cache = ResultCache(cache_path=output_dir)

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

        _release_task_memory()

    elapsed_all = time.perf_counter() - start_all
    if failures:
        logger.warning(
            "Finished with %d failed task(s) in %.1fs",
            len(failures),
            elapsed_all,
        )
    else:
        logger.info("All tasks finished in %.1fs", elapsed_all)

    from mteb.results.model_result import ModelResult

    combined = ModelResult(
        model_name=source.hub_id or source.path,
        model_revision=None,
        task_results=all_task_results,
        exceptions=all_exceptions or None,
    )
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined.model_dump(), f, indent=2, default=str)
    logger.info("Wrote summary to %s", summary_path)

    summary_rows = build_summary_rows(
        combined,
        timings,
        failures=failures,
        task_prompts=task_prompts,
    )
    csv_path = output_dir / "summary.csv"
    write_summary_csv(csv_path, summary_rows, include_average=True)
    logger.info("Wrote summary CSV to %s", csv_path)

    print_summary(
        combined,
        timings,
        failures=failures,
        task_prompts=task_prompts,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
