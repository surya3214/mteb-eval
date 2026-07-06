"""Model loading for Hub IDs and local checkpoint folders."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LEN = 512


QWEN3_HUB_IDS = (
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
)


@dataclass(frozen=True)
class ModelSource:
    """Resolved model source (Hub id or local folder)."""

    path: str
    is_local: bool
    hub_id: str | None


@dataclass(frozen=True)
class CheckpointLayout:
    """Detected on-disk checkpoint layout."""

    kind: str  # sentence-transformers | transformers | unknown


def resolve_model_source(
    *,
    model: str | None,
    model_path: str | None,
    hub_id: str | None,
) -> ModelSource:
    """Resolve --model vs --model-path into a unified ModelSource."""
    if model_path and model:
        raise ValueError("Use either --model or --model-path, not both.")

    if model_path:
        p = Path(model_path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"--model-path does not exist or is not a directory: {p}\n"
                "For Hub models use --model <repo_id> instead."
            )
        validate_local_checkpoint(p)
        return ModelSource(path=str(p), is_local=True, hub_id=hub_id)

    if model:
        expanded = Path(model).expanduser()
        if expanded.is_dir():
            p = expanded.resolve()
            validate_local_checkpoint(p)
            return ModelSource(path=str(p), is_local=True, hub_id=hub_id or model)
        return ModelSource(path=model, is_local=False, hub_id=model)

    raise ValueError("Provide --model <hub_id> or --model-path <local_dir>.")


def validate_local_checkpoint(path: Path) -> CheckpointLayout:
    """Verify a local folder looks loadable; raise with actionable message if not."""
    st_markers = (
        "modules.json",
        "config_sentence_transformers.json",
    )
    has_st = any((path / m).exists() for m in st_markers) or (
        (path / "config.json").exists() and (path / "1_Pooling").exists()
    )
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    has_transformers = (path / "config.json").exists() and bool(weight_files)

    if has_st:
        layout = CheckpointLayout(kind="sentence-transformers")
    elif has_transformers:
        layout = CheckpointLayout(kind="transformers")
    else:
        layout = CheckpointLayout(kind="unknown")
        logger.warning(
            "Could not detect a standard layout in %s (expected ST or transformers "
            "artifacts). Will attempt SentenceTransformer load anyway.",
            path,
        )

    logger.info("Detected checkpoint layout: %s (%s)", layout.kind, path)
    return layout


def detect_model_type(
    source: ModelSource,
    model_type: str,
) -> str:
    """Resolve model_type=auto from path name, config.json, or hub_id."""
    if model_type != "auto":
        return model_type

    haystack = " ".join(
        filter(
            None,
            [
                source.path.lower(),
                (source.hub_id or "").lower(),
            ],
        )
    )
    if "qwen3" in haystack or "qwen/qwen3" in haystack:
        return "qwen3"
    if "harrier" in haystack:
        return "harrier"
    if "eurobert" in haystack and "sentence" not in haystack:
        return "eurobert-base"

    if source.is_local:
        config_path = Path(source.path) / "config.json"
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as f:
                    cfg = json.load(f)
                model_type_field = str(cfg.get("model_type", "")).lower()
                name_or_path = str(cfg.get("_name_or_path", "")).lower()
                if "qwen3" in model_type_field or "qwen3" in name_or_path:
                    return "qwen3"
                if "eurobert" in model_type_field or "eurobert" in name_or_path:
                    return "eurobert-base"
            except (json.JSONDecodeError, OSError):
                pass

    return "sentence-transformer"


def _read_config_name_or_path(local_path: Path) -> str | None:
    config_path = local_path / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def infer_qwen3_hub_id(source: ModelSource) -> str | None:
    """Best-effort Qwen3 hub id for MTEB instruct wrapper lookup."""
    if source.hub_id and source.hub_id in QWEN3_HUB_IDS:
        return source.hub_id
    if source.hub_id and "qwen3" in source.hub_id.lower():
        return source.hub_id

    if source.is_local:
        name_or_path = _read_config_name_or_path(Path(source.path))
        if name_or_path:
            for hid in QWEN3_HUB_IDS:
                if hid.lower() in name_or_path.lower():
                    return hid
            if "qwen3" in name_or_path.lower():
                return name_or_path

    if not source.is_local and source.path in QWEN3_HUB_IDS:
        return source.path

    return source.hub_id


def load_qwen3_local(
    source: ModelSource,
    *,
    device: str | None,
    hub_id: str | None,
    model_kwargs: dict[str, Any],
) -> Any:
    """Load local Qwen3 weights with MTEB instruct wrapper when possible."""
    from mteb.models.model_implementations.qwen3_models import q3e_instruct_loader

    resolved_hub = hub_id or infer_qwen3_hub_id(source)
    if resolved_hub:
        import mteb

        meta = mteb.get_model_meta(resolved_hub)
        if meta is not None:
            logger.info(
                "Loading local Qwen3 via MTEB instruct loader (hub ref: %s)",
                resolved_hub,
            )
            kwargs = {**model_kwargs, "device": device}
            return q3e_instruct_loader(source.path, revision=None, **kwargs)

    logger.warning(
        "No Qwen3 hub id resolved; loading SentenceTransformer with left padding. "
        "Scores may differ from the MTEB leaderboard. Pass --hub-id explicitly."
    )
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        source.path,
        device=device,
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        **model_kwargs,
    )


class EuroBertEncoderWrapper:
    """Mean-pooled encoder wrapper for base EuroBERT checkpoints (not ST-packaged)."""

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str | None = None,
        max_length: int = 512,
        **kwargs: Any,
    ) -> "EuroBertEncoderWrapper":
        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=kwargs.get("torch_dtype"),
        )
        model.to(resolved_device)
        model.eval()
        wrapper = cls(model, tokenizer, resolved_device)
        wrapper.max_length = max_length
        return wrapper

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> Any:
        import torch

        all_embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=getattr(self, "max_length", 512),
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                embeddings = summed / counts
            all_embeddings.append(embeddings.cpu().numpy())
        import numpy as np

        return np.vstack(all_embeddings)


def load_embedding_model(
    source: ModelSource,
    *,
    model_type: str = "auto",
    device: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load an embedding model from Hub or local path."""
    import mteb
    from sentence_transformers import SentenceTransformer

    kwargs = dict(model_kwargs or {})
    resolved_type = detect_model_type(source, model_type)

    if not source.is_local:
        meta = mteb.get_model_meta(source.path)
        if meta is not None:
            logger.info("Loading Hub model via MTEB registry: %s", source.path)
            return meta.load_model(device=device, **kwargs)
        logger.info("Loading Hub model via SentenceTransformer: %s", source.path)
        return SentenceTransformer(
            source.path,
            device=device,
            trust_remote_code=True,
            **kwargs,
        )

    if resolved_type == "qwen3":
        return load_qwen3_local(
            source,
            device=device,
            hub_id=source.hub_id,
            model_kwargs=kwargs,
        )

    if resolved_type == "eurobert-base":
        logger.info("Loading local EuroBERT base encoder: %s", source.path)
        return EuroBertEncoderWrapper.from_pretrained(
            source.path,
            device=device,
            **kwargs,
        )

    logger.info(
        "Loading local SentenceTransformer (%s): %s",
        resolved_type,
        source.path,
    )
    return SentenceTransformer(
        source.path,
        device=device,
        trust_remote_code=True,
        **kwargs,
    )


def configure_max_seq_len(model: Any, max_seq_len: int) -> None:
    """Apply a sequence-length cap to the loaded encoder when supported."""
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    if isinstance(model, EuroBertEncoderWrapper):
        model.max_length = max_seq_len
        logger.info("Set max_length=%d on EuroBERT encoder", max_seq_len)
        return

    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "max_seq_length"):
        inner.max_seq_length = max_seq_len
        logger.info("Set max_seq_length=%d on encoder", max_seq_len)
        return

    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq_len
        logger.info("Set max_seq_length=%d on encoder", max_seq_len)
        return

    logger.warning(
        "Encoder %s does not expose max_seq_length; relying on encode-time truncation only",
        type(model).__name__,
    )
