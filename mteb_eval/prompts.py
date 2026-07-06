"""Prompt resolution and optional query/document prefix configuration."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _sentence_transformer(model: Any) -> Any | None:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "prompts"):
        return inner
    if hasattr(model, "prompts"):
        return model
    return None


def configure_prompt_prefixes(
    model: Any,
    *,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> None:
    """Set global query/document prefixes on SentenceTransformer-style encoders."""
    if query_prefix is None and document_prefix is None:
        return

    st = _sentence_transformer(model)
    if st is not None:
        prompts = dict(getattr(st, "prompts", None) or {})
        if query_prefix is not None:
            prompts["query"] = query_prefix
        if document_prefix is not None:
            prompts["document"] = document_prefix
        st.prompts = prompts
        logger.info(
            "Set SentenceTransformer prompts: query=%r document=%r",
            prompts.get("query"),
            prompts.get("document"),
        )

    if hasattr(model, "model_prompts"):
        if model.model_prompts is None:
            model.model_prompts = {}
        if query_prefix is not None:
            model.model_prompts["query"] = query_prefix
        if document_prefix is not None:
            model.model_prompts["document"] = document_prefix
        logger.info(
            "Set model_prompts: query=%r document=%r",
            model.model_prompts.get("query"),
            model.model_prompts.get("document"),
        )

    if st is None and not hasattr(model, "model_prompts"):
        logger.warning(
            "Model %s does not expose ST prompts; --query-prefix/--document-prefix "
            "may have no effect. Instruct models use per-task instructions instead.",
            type(model).__name__,
        )


def resolve_task_prompts(model: Any, task: Any) -> dict[str, str]:
    """Resolve the effective query/document prompt text for a task."""
    from mteb.models.abs_encoder import get_prompt
    from mteb.types import PromptType

    task_metadata = task.metadata
    resolved = {"query": "", "document": ""}

    if hasattr(model, "get_task_instruction"):
        for prompt_type, key in (
            (PromptType.query, "query"),
            (PromptType.document, "document"),
        ):
            try:
                text = model.get_task_instruction(task_metadata, prompt_type)
                resolved[key] = text or ""
            except (ValueError, TypeError, AttributeError):
                resolved[key] = ""

    model_prompts = getattr(model, "prompts_dict", None) or getattr(
        model, "model_prompts", None
    )
    if model_prompts:
        for prompt_type, key in (
            (PromptType.query, "query"),
            (PromptType.document, "document"),
        ):
            if resolved[key]:
                continue
            prompt_text = get_prompt(model_prompts, task_metadata, prompt_type)
            if prompt_text:
                resolved[key] = prompt_text

    st = _sentence_transformer(model)
    if st is not None:
        st_prompts = getattr(st, "prompts", None) or {}
        if not resolved["query"] and st_prompts.get("query"):
            resolved["query"] = str(st_prompts["query"])
        if not resolved["document"] and st_prompts.get("document"):
            resolved["document"] = str(st_prompts["document"])

    if task_metadata.type == "STS" and not any(resolved.values()):
        from mteb.get_tasks import get_task

        abstask = get_task(task_name=task_metadata.name)
        instruction = abstask.abstask_prompt
        resolved["query"] = instruction
        resolved["document"] = f"{instruction} (symmetric STS — same for both sides)"

    if not resolved["query"] and not resolved["document"]:
        resolved["query"] = "(none)"
        resolved["document"] = "(none)"
    elif not resolved["document"]:
        resolved["document"] = "(none — not applied to passages/documents)"
    elif not resolved["query"]:
        resolved["query"] = "(none)"

    return resolved


def print_task_prompts(
    task_name: str,
    prompts: dict[str, str],
    *,
    logger_only: bool = False,
) -> None:
    """Log or print query/document prompts for a task."""
    lines = [
        f"Prompts for {task_name}:",
        f"  query:    {prompts.get('query', '')}",
        f"  document: {prompts.get('document', '')}",
    ]
    message = "\n".join(lines)
    logger.info(message)
    if not logger_only:
        print(message)
