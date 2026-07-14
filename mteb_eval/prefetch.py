"""Prefetch MTEB datasets (and optional models) into a portable HF cache."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from mteb_eval.cache import configure_cache
from mteb_eval.languages import (
    DEFAULT_LANGUAGES_PRESET,
    ML16_LANGUAGES,
    add_language_arguments,
    languages_from_args,
)
from mteb_eval.offline_compat import apply_mteb_offline_compat
from mteb_eval.tasks import (
    MANIFEST_NAME,
    ML16_CLF_CLUST_RERANK_MANIFEST,
    add_task_arguments,
    dataset_info,
    is_clf_clust_rerank_types,
    resolve_tasks,
    task_names_from_args,
    validate_against_manifest,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prefetch MTEB datasets into HF cache. "
            "Default: STS + Retrieval. Also supports Classification, Clustering, "
            "and Reranking via --task-types."
        ),
    )
    cache = parser.add_mutually_exclusive_group(required=True)
    cache.add_argument(
        "--cache-dir",
        type=str,
        help="Portable HF cache root (sets HF_HOME, HF_HUB_CACHE, etc.).",
    )
    cache.add_argument(
        "--default-cache",
        action="store_true",
        help="Use system default ~/.cache/huggingface (no env override).",
    )
    parser.add_argument(
        "--benchmark",
        default="MTEB(Multilingual, v2)",
        help='MTEB benchmark name (default: "MTEB(Multilingual, v2)").',
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=["STS", "Retrieval"],
        help=(
            "Task types to prefetch (default: STS Retrieval). "
            "Also supports Classification Clustering Reranking."
        ),
    )
    add_task_arguments(parser)
    add_language_arguments(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional Hub model ids to snapshot_download into the cache.",
    )
    parser.add_argument(
        "--validate-manifest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Validate against a shipped task inventory manifest. "
            "Default: on for MTEB(eng, v2) STS+Retrieval without overrides, "
            "and for Multilingual v2 Classification+Clustering+Reranking with ml16."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def prefetch_models(model_ids: list[str], cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download

    for model_id in model_ids:
        logger.info("Downloading model weights: %s", model_id)
        kwargs: dict = {"repo_id": model_id}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir / "hub")
        snapshot_download(**kwargs)
        logger.info("Finished model: %s", model_id)


def prefetch_datasets(tasks: list) -> None:
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        info = dataset_info(task)
        logger.info(
            "[%d/%d] Loading %s (dataset=%s)",
            idx,
            total,
            task.metadata.name,
            info["path"],
        )
        start = time.perf_counter()
        task.load_data()
        elapsed = time.perf_counter() - start
        logger.info("  done in %.1fs", elapsed)


def _resolve_manifest_validation(
    *,
    should_validate: bool | None,
    benchmark: str,
    task_types: list[str],
    names: list[str] | None,
    languages: list[str] | None,
    languages_preset: str | None,
) -> str | None:
    """Return manifest filename to validate against, or None to skip."""
    if should_validate is False:
        return None

    if should_validate is True:
        if (
            benchmark == "MTEB(Multilingual, v2)"
            and is_clf_clust_rerank_types(task_types)
            and names is None
            and languages is not None
            and set(languages) == set(ML16_LANGUAGES)
        ):
            return ML16_CLF_CLUST_RERANK_MANIFEST
        return MANIFEST_NAME

    # Auto mode
    if (
        benchmark == "MTEB(eng, v2)"
        and names is None
        and languages is None
        and languages_preset == "none"
    ):
        return MANIFEST_NAME

    if (
        benchmark == "MTEB(Multilingual, v2)"
        and is_clf_clust_rerank_types(task_types)
        and names is None
        and languages is not None
        and set(languages) == set(ML16_LANGUAGES)
    ):
        return ML16_CLF_CLUST_RERANK_MANIFEST

    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_cfg = configure_cache(
        cache_dir=args.cache_dir,
        default_cache=args.default_cache,
    )
    apply_mteb_offline_compat()

    names, from_preset = task_names_from_args(args)
    languages = languages_from_args(args)
    tasks = resolve_tasks(
        benchmark=args.benchmark,
        task_types=args.task_types,
        task_names=names,
        languages=languages,
        exclusive_language_filter=args.exclusive_language_filter,
        allow_missing_task_names=from_preset,
    )

    manifest_name = _resolve_manifest_validation(
        should_validate=args.validate_manifest,
        benchmark=args.benchmark,
        task_types=args.task_types,
        names=names,
        languages=languages,
        languages_preset=getattr(args, "languages_preset", DEFAULT_LANGUAGES_PRESET),
    )
    if manifest_name is not None:
        validate_against_manifest(tasks, manifest_name=manifest_name)
        logger.info("Validated against manifest %s", manifest_name)

    logger.info("Prefetching %d task(s)...", len(tasks))
    prefetch_datasets(tasks)

    if args.models:
        prefetch_models(args.models, cache_cfg.cache_dir)

    logger.info("Prefetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
