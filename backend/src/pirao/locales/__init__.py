"""Translations of user-facing text.  English is the source language."""

from __future__ import annotations

from .pt import MESSAGES as _PT

SUPPORTED = ("en", "pt")


def translate(text: str, lang: str) -> str:
    """``text`` in ``lang``, or unchanged when there is no translation."""
    if lang == "pt" and text:
        return _PT.get(text, text)
    return text


def missing(texts: list[str], lang: str = "pt") -> list[str]:
    """Which of ``texts`` have no translation -- for the tests."""
    catalogue = _PT if lang == "pt" else {}
    return [t for t in texts if t and t not in catalogue]
