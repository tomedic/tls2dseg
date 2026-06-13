"""Dot-separated class-prompt splitting — single source of truth (stdlib only)."""

from __future__ import annotations


def split_class_keys(text: str) -> list[str]:
    """Split a dot-separated class prompt into clean class keys.

    Drops empty/whitespace-only segments (stray dots, leading/trailing dots,
    the mandatory trailing-period artifact) and strips each key.
    """
    return [k.strip() for k in text.split(".") if k.strip()]
