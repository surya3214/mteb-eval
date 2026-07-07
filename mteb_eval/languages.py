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


def resolve_languages(
    *,
    languages: Sequence[str] | None = None,
    languages_preset: str | None = None,
) -> list[str] | None:
    """Resolve explicit language codes or a named preset into a language list."""
    if languages is not None and languages_preset is not None:
        raise ValueError("Use either --languages or --languages-preset, not both.")
    if languages_preset is not None:
        try:
            return list(LANGUAGE_PRESETS[languages_preset])
        except KeyError as exc:
            known = ", ".join(sorted(LANGUAGE_PRESETS))
            raise ValueError(
                f"Unknown languages preset {languages_preset!r}. Known presets: {known}"
            ) from exc
    if languages is not None:
        return list(languages)
    return None


def languages_from_args(args: argparse.Namespace) -> list[str] | None:
    """Read resolved languages from parsed CLI args."""
    return resolve_languages(
        languages=getattr(args, "languages", None),
        languages_preset=getattr(args, "languages_preset", None),
    )


def add_language_arguments(parser: argparse.ArgumentParser) -> None:
    """Register --languages / --languages-preset / --exclusive-language-filter."""
    lang_group = parser.add_mutually_exclusive_group()
    lang_group.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language-script codes to keep (e.g. eng-Latn deu-Latn).",
    )
    lang_group.add_argument(
        "--languages-preset",
        choices=sorted(LANGUAGE_PRESETS),
        default=None,
        help="Named language preset (ml16 = 16-language multilingual set).",
    )
    parser.add_argument(
        "--exclusive-language-filter",
        action="store_true",
        help="Keep only subsets where ALL languages are in --languages / preset.",
    )
