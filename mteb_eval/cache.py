"""Hugging Face cache configuration — must run before importing mteb/datasets/transformers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheConfig:
    """Resolved cache configuration."""

    cache_dir: Path | None
    offline: bool
    use_default_cache: bool


def configure_cache(
    *,
    cache_dir: str | Path | None = None,
    default_cache: bool = False,
    offline: bool = False,
) -> CacheConfig:
    """Configure HF cache environment variables.

    Two mutually exclusive modes:
    - ``cache_dir``: portable cache rooted at the given directory.
    - ``default_cache``: leave ``~/.cache/huggingface`` untouched (no env override).

    When ``offline`` is True, datasets/hub/transformers offline flags are set.
    Local ``--model-path`` checkpoints still load from disk.
    """
    if cache_dir is not None and default_cache:
        raise ValueError("Use either --cache-dir or --default-cache, not both.")

    if default_cache:
        return CacheConfig(cache_dir=None, offline=offline, use_default_cache=True)

    if cache_dir is None:
        raise ValueError("Provide --cache-dir or pass --default-cache.")

    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    hub = root / "hub"
    datasets = root / "datasets"
    models = root / "models"
    for sub in (hub, datasets, models):
        sub.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_DATASETS_CACHE"] = str(datasets)
    os.environ["TRANSFORMERS_CACHE"] = str(models)

    if offline:
        _set_offline(True)

    return CacheConfig(cache_dir=root, offline=offline, use_default_cache=False)


def _set_offline(enabled: bool) -> None:
    value = "1" if enabled else "0"
    os.environ["HF_DATASETS_OFFLINE"] = value
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ["TRANSFORMERS_OFFLINE"] = value


def apply_offline(offline: bool) -> None:
    """Set or clear offline flags without changing cache location."""
    _set_offline(offline)


def get_configured_cache_dir() -> Path | None:
    """Return the configured HF_HOME if set, else None."""
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) if hf_home else None
