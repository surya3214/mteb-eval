"""Compatibility shims for offline HF datasets + MTEB Retrieval loading.

Some Retrieval Hub datasets store qrels under config ``qrels`` (e.g. SpartQA,
WinoGrande, StackOverflowQA). MTEB's loader prefers config ``default`` when the
subset is unnamed, which fails offline when only ``corpus`` / ``qrels`` /
``queries`` exist in the datasets cache:

    Couldn't find cache for config 'default'
    Available configs in the cache: ['corpus', 'qrels', 'queries']

This module patches MTEB to prefer ``qrels`` when that config is advertised
(or present in the local arrow cache), and to fall back if ``default`` load fails.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_PATCHED = False


def _datasets_cache_root() -> Path:
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "datasets"
    return Path.home() / ".cache" / "huggingface" / "datasets"


def list_cached_dataset_configs(dataset_name: str) -> list[str]:
    """List config names present under HF_DATASETS_CACHE for a Hub dataset id."""
    # datasets stores mteb/Foo as mteb___foo (camelCase → snake_case for last part)
    from datasets.naming import camelcase_to_snakecase

    parts = dataset_name.split("/")
    parts = list(parts)
    parts[-1] = camelcase_to_snakecase(parts[-1])
    root = _datasets_cache_root() / "___".join(parts)
    if not root.is_dir():
        return []

    configs: set[str] = set()
    for info_path in glob.glob(str(root / "*" / "*" / "*" / "dataset_info.json")):
        try:
            info = json.loads(Path(info_path).read_text(encoding="utf-8"))
            name = info.get("config_name") or Path(info_path).parts[-4]
            configs.add(str(name))
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    return sorted(configs)


def resolve_qrels_config(
    dataset_configs: Sequence[str],
    *,
    subset_config: str | None,
    dataset_name: str | None = None,
) -> str:
    """Pick the HF config name that holds qrels for a Retrieval dataset."""
    if subset_config is not None:
        return f"{subset_config}-qrels"

    configs = set(dataset_configs)
    if dataset_name:
        configs.update(list_cached_dataset_configs(dataset_name))

    # Prefer explicit ``qrels`` (newer Hub layout) over classic BEIR ``default``.
    if "qrels" in configs:
        return "qrels"
    if "default" in configs:
        return "default"
    raise ValueError(
        "No qrels or default config found. Please specify a valid config or "
        "ensure the dataset has qrels."
    )


def apply_mteb_offline_compat() -> None:
    """Monkeypatch MTEB RetrievalDatasetLoader for offline qrels config resolution."""
    global _PATCHED
    if _PATCHED:
        return

    from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader

    original_load_qrels = RetrievalDatasetLoader._load_qrels

    def _load_qrels(self, num_proc):  # type: ignore[no-untyped-def]
        if self.config is not None:
            return original_load_qrels(self, num_proc)

        try:
            config = resolve_qrels_config(
                self.dataset_configs,
                subset_config=None,
                dataset_name=self.hf_repo,
            )
        except ValueError:
            return original_load_qrels(self, num_proc)

        # Reuse original implementation path by temporarily aligning configs.
        # Prefer calling the low-level loader with the resolved config.
        from datasets import Features, Value

        logger.info("Loading qrels subset: %s", config)
        try:
            qrels_ds = self._load_dataset_split(config, num_proc)
        except (ValueError, FileNotFoundError, OSError) as exc:
            # Hub metadata may advertise ``default`` while only ``qrels`` is cached.
            if config == "default" and "qrels" in set(self.dataset_configs).union(
                list_cached_dataset_configs(self.hf_repo)
            ):
                logger.warning(
                    "Failed to load qrels config %r (%s); retrying with 'qrels'",
                    config,
                    exc,
                )
                qrels_ds = self._load_dataset_split("qrels", num_proc)
            else:
                raise

        qrels_ds = qrels_ds.select_columns(["query-id", "corpus-id", "score"])
        qrels_ds = qrels_ds.cast(
            Features(
                {
                    "query-id": Value("string"),
                    "corpus-id": Value("string"),
                    "score": Value("int32"),
                }
            )
        )
        qrels_ds = qrels_ds.to_polars()
        qrels_dict = {
            query_id[0]: dict(zip(group["corpus-id"], group["score"]))
            for query_id, group in qrels_ds.group_by("query-id", maintain_order=False)
        }
        logger.info("Loaded %d %s qrels.", len(qrels_dict), self.split.upper())
        return qrels_dict

    RetrievalDatasetLoader._load_qrels = _load_qrels  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied MTEB offline Retrieval qrels compatibility patch")
