"""Multi-GPU parallel MTEB evaluation coordinator."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from argparse import Namespace
from pathlib import Path

from mteb_eval.evaluate import build_parser as build_eval_parser
from mteb_eval.languages import languages_from_args
from mteb_eval.offline_compat import apply_mteb_offline_compat
from mteb_eval.runner import run_evaluation
from mteb_eval.summary import (
    build_summary_rows,
    merge_shard_results,
    print_language_summary,
    print_summary,
    write_language_outputs,
    write_results_workbook,
    write_summary_csv,
)
from mteb_eval.tasks import partition_task_names, resolve_tasks, task_selection_from_args

logger = logging.getLogger(__name__)

SHARDS_DIRNAME = ".shards"


def build_parser() -> argparse.ArgumentParser:
    parser = build_eval_parser()
    parser.description = "Evaluate an embedding model on MTEB tasks in parallel across GPUs."
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="Comma-separated GPU ids (e.g. 0,1,2,3) or 'auto' for all visible CUDA devices.",
    )
    return parser


def resolve_gpus(gpus_arg: str) -> list[str]:
    """Resolve --gpus into a list of device id strings."""
    if gpus_arg.strip().lower() == "auto":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for GPU auto-detection. Install torch or pass --gpus explicitly."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPUs available for parallel evaluation.")
        return [str(i) for i in range(torch.cuda.device_count())]

    gpu_ids = [part.strip() for part in gpus_arg.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError(f"Invalid --gpus value: {gpus_arg!r}")
    return gpu_ids


def _namespace_to_dict(args: Namespace) -> dict:
    return {key: getattr(args, key) for key in vars(args)}


def _worker(
    gpu_id: str,
    task_names: list[str],
    args_dict: dict,
    shard_dir: str,
    task_types: list[str],
) -> int:
    """Run evaluation on one GPU for a disjoint task subset."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    worker_args = Namespace(**args_dict)
    worker_args.device = "cuda"
    worker_args.tasks = task_names
    worker_args.tasks_preset = None
    # Presets (e.g. mteb-eval-all) override CLI --task-types; keep the
    # coordinator-resolved types so Classification/Clustering/Reranking names
    # resolve instead of crashing before summary.json is written.
    worker_args.task_types = list(task_types)

    logging.basicConfig(
        level=logging.DEBUG if worker_args.verbose else logging.INFO,
        format=f"%(asctime)s GPU{gpu_id} %(levelname)s %(message)s",
    )

    result = run_evaluation(
        worker_args,
        task_names=task_names,
        task_types=list(task_types),
        output_dir=Path(shard_dir),
    )
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    apply_mteb_offline_compat()

    gpu_ids = resolve_gpus(args.gpus)
    logger.info("Using %d GPU(s): %s", len(gpu_ids), ", ".join(gpu_ids))

    languages = languages_from_args(args)
    types, names, from_preset = task_selection_from_args(args)
    all_tasks = resolve_tasks(
        benchmark=args.benchmark,
        task_types=types,
        task_names=names,
        languages=languages,
        exclusive_language_filter=args.exclusive_language_filter,
        allow_missing_task_names=from_preset,
    )
    task_types_by_name = {t.metadata.name: t.metadata.type for t in all_tasks}
    task_names = [t.metadata.name for t in all_tasks]
    if not task_names:
        raise ValueError("No tasks to evaluate after resolution.")

    partitions = partition_task_names(task_names, len(gpu_ids))
    output_dir = Path(args.output_dir).expanduser().resolve()
    shards_root = output_dir / SHARDS_DIRNAME
    shards_root.mkdir(parents=True, exist_ok=True)

    args_dict = _namespace_to_dict(args)
    # Ensure workers inherit coordinator-resolved types (preset overrides).
    args_dict["task_types"] = list(types)
    args_dict["tasks_preset"] = None
    processes: list[mp.Process] = []
    shard_dirs: list[Path] = []

    for gpu_id, names in zip(gpu_ids, partitions):
        if not names:
            logger.info("GPU %s: no tasks assigned, skipping worker", gpu_id)
            continue
        shard_dir = shards_root / f"gpu{gpu_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_dirs.append(shard_dir)
        logger.info(
            "GPU %s: %d task(s) -> %s",
            gpu_id,
            len(names),
            ", ".join(names),
        )
        proc = mp.Process(
            target=_worker,
            args=(gpu_id, names, args_dict, str(shard_dir), list(types)),
            name=f"mteb-gpu{gpu_id}",
        )
        processes.append(proc)
        proc.start()

    exit_codes: list[int] = []
    for proc in processes:
        proc.join()
        exit_codes.append(proc.exitcode if proc.exitcode is not None else 1)

    if any(code != 0 for code in exit_codes):
        logger.error("One or more GPU workers failed: %s", exit_codes)

    if not shard_dirs:
        raise RuntimeError("No GPU workers were started (empty task list?).")

    missing = [str(d) for d in shard_dirs if not (d / "summary.json").exists()]
    if missing:
        raise RuntimeError(
            "One or more GPU workers exited without writing summary.json "
            f"(exit_codes={exit_codes}). Missing: {missing}. "
            "Check worker logs above; common cause was presets like "
            "mteb-eval-all not propagating task_types to workers."
        )

    merged = merge_shard_results(shard_dirs, output_dir)

    summary_rows = build_summary_rows(
        merged.model_result,
        merged.timings,
        failures=merged.failures,
        task_prompts=merged.task_prompts,
    )
    csv_path = output_dir / "summary.csv"
    write_summary_csv(csv_path, summary_rows, include_average=True)
    logger.info("Wrote merged summary CSV to %s", csv_path)
    write_language_outputs(output_dir, merged.model_result)
    # Prefer coordinator-resolved types; fall back to merged shard meta.
    types_map = task_types_by_name or merged.task_types_by_name
    xlsx_path = output_dir / "results.xlsx"
    write_results_workbook(
        xlsx_path,
        summary_rows,
        merged.model_result,
        task_types_by_name=types_map,
    )
    logger.info("Wrote results workbook to %s", xlsx_path)

    print_summary(
        merged.model_result,
        merged.timings,
        failures=merged.failures,
        task_prompts=merged.task_prompts,
    )
    print_language_summary(merged.model_result)

    final_exit = 1 if any(code != 0 for code in exit_codes) or merged.exit_code else 0
    return final_exit


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
