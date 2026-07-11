"""Evaluate embedding models on MTEB STS + Retrieval tasks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mteb_eval.languages import add_language_arguments
from mteb_eval.model_loader import (
    DEFAULT_MAX_SEQ_LEN,
    VALID_ATTN_IMPLEMENTATIONS,
    VALID_DTYPES,
)
from mteb_eval.runner import run_evaluation
from mteb_eval.summary import (
    build_summary_rows,
    print_language_summary,
    print_summary,
    write_language_outputs,
    write_summary_csv,
)
from mteb_eval.tasks import add_task_arguments


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
        default="MTEB(Multilingual, v2)",
        help='MTEB benchmark (default: "MTEB(Multilingual, v2)").',
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["STS", "Retrieval"],
        help="Task types to evaluate.",
    )
    add_task_arguments(parser)
    add_language_arguments(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for MTEB result cache and summary JSON.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, cpu, mps).")
    parser.add_argument(
        "--dtype",
        choices=list(VALID_DTYPES),
        default="bfloat16",
        help="Model weight dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=list(VALID_ATTN_IMPLEMENTATIONS),
        default="sdpa",
        help="HF attention backend (default: sdpa).",
    )
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after task failures (default: on). Use --no-continue-on-error to stop.",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = run_evaluation(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = build_summary_rows(
        result.model_result,
        result.timings,
        failures=result.failures,
        task_prompts=result.task_prompts,
    )
    csv_path = output_dir / "summary.csv"
    write_summary_csv(csv_path, summary_rows, include_average=True)
    logging.getLogger(__name__).info("Wrote summary CSV to %s", csv_path)
    write_language_outputs(output_dir, result.model_result)

    print_summary(
        result.model_result,
        result.timings,
        failures=result.failures,
        task_prompts=result.task_prompts,
    )
    print_language_summary(result.model_result)

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
