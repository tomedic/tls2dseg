"""Tier-A tests for empty-class cleaning in prompt class splitting.

Covers the shared :func:`split_class_keys` helper, the ``PromptConfig.text``
normalizer, and ``_derive_class_id_map`` — a dirty prompt must never yield an
empty or whitespace-padded class key, and an all-empty prompt must still raise.

These tests are explicitly ``tier_a``: no pchandler/pc2img/torch imports.
"""

from __future__ import annotations

import pytest

from tls2dseg.config.models import PromptConfig
from tls2dseg.config.text import split_class_keys
from tls2dseg.runtime.context import _derive_class_id_map

DIRTY_INPUTS = [
    ("tree.. pole.", ["tree", "pole"], "tree. pole."),
    (".tree. pole.", ["tree", "pole"], "tree. pole."),
    (" tree .  pole . ", ["tree", "pole"], "tree. pole."),
]


@pytest.mark.tier_a
@pytest.mark.parametrize(("raw", "keys", "_canonical"), DIRTY_INPUTS)
def test_split_class_keys_drops_empty_and_padded(raw: str, keys: list[str], _canonical: str) -> None:
    """Dirty inputs yield clean key lists with no '' and no whitespace-padded key."""
    result = split_class_keys(raw)
    assert result == keys
    assert "" not in result
    assert all(k == k.strip() for k in result)


@pytest.mark.tier_a
def test_split_class_keys_all_empty_returns_empty() -> None:
    """All-empty / trailing-period-only inputs split to []."""
    assert split_class_keys("....") == []
    assert split_class_keys("") == []
    assert split_class_keys("   .") == []


@pytest.mark.tier_a
@pytest.mark.parametrize(("raw", "_keys", "canonical"), DIRTY_INPUTS)
def test_promptconfig_normalizes_dirty_input(raw: str, _keys: list[str], canonical: str) -> None:
    """PromptConfig.text drops empty segments and emits a clean canonical prompt."""
    text = PromptConfig(text=raw).text
    assert text == canonical
    assert "" not in text.split(".")[:-1]


@pytest.mark.tier_a
def test_derive_class_id_map_has_no_empty_key() -> None:
    """Class-id map from a dirty-prompt canonical string has no '' key; background maps to 0."""
    canonical = PromptConfig(text="tree.. pole.").text
    id_map = _derive_class_id_map(canonical)
    assert "" not in id_map
    assert all(k == k.strip() for k in id_map)
    assert id_map["background"] == 0


@pytest.mark.tier_a
@pytest.mark.parametrize("raw", ["....", "   ."])
def test_promptconfig_all_empty_raises(raw: str) -> None:
    """An all-empty prompt still raises ValueError('prompt.text must be non-empty')."""
    with pytest.raises(ValueError, match="must be non-empty"):
        PromptConfig(text=raw)
