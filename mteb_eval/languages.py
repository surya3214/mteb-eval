"""Language presets and CLI resolution for MTEB task filtering."""

from __future__ import annotations

import argparse
from typing import Sequence

# EN, KO, AR, ZH, FR, DE, HI, ID, IT, JP, PT, RU, ES, VI, TH, PL
ML16_LANGUAGES: tuple[str, ...] = (
    "eng-Latn",
    "kor-Hang",
    "ara-Arab",
    "zho-Hans",
    "fra-Latn",
    "deu-Latn",
    "hin-Deva",
    "ind-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "por-Latn",
    "rus-Cyrl",
    "spa-Latn",
    "vie-Latn",
    "tha-Latn",
    "pol-Latn",
)

LANGUAGE_PRESETS: dict[str, tuple[str, ...]] = {
    "ml16": ML16_LANGUAGES,
}

DEFAULT_LANGUAGES_PRESET = "ml16"


def resolve_languages(
    *,
    languages: Sequence[str] | None = None,
    languages_preset: str | None = DEFAULT_LANGUAGES_PRESET,
) -> list[str] | None:
    """Resolve explicit language codes or a named preset into a language list.

    Priority: explicit ``--languages`` wins over ``--languages-preset``.
    Preset ``none`` (or None) disables filtering.
    """
    if languages is not None:
        return list(languages)
    if languages_preset is None or languages_preset == "none":
        return None
    try:
        return list(LANGUAGE_PRESETS[languages_preset])
    except KeyError as exc:
        known = ", ".join([*sorted(LANGUAGE_PRESETS), "none"])
        raise ValueError(
            f"Unknown languages preset {languages_preset!r}. Known presets: {known}"
        ) from exc


def languages_from_args(args: argparse.Namespace) -> list[str] | None:
    """Read resolved languages from parsed CLI args."""
    return resolve_languages(
        languages=getattr(args, "languages", None),
        languages_preset=getattr(args, "languages_preset", DEFAULT_LANGUAGES_PRESET),
    )


def add_language_arguments(parser: argparse.ArgumentParser) -> None:
    """Register --languages / --languages-preset / --exclusive-language-filter."""
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help=(
            "Language-script codes to keep (e.g. eng-Latn deu-Latn). "
            "Overrides --languages-preset when set."
        ),
    )
    parser.add_argument(
        "--languages-preset",
        choices=[*sorted(LANGUAGE_PRESETS), "none"],
        default=DEFAULT_LANGUAGES_PRESET,
        help=(
            "Named language preset (default: ml16). "
            "Use 'none' to evaluate all language subsets."
        ),
    )
    parser.add_argument(
        "--exclusive-language-filter",
        action="store_true",
        help="Keep only subsets where ALL languages are in --languages / preset.",
    )
