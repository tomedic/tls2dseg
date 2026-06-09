"""Tier-A MODE-01 assertion — ``mode`` is required and Literal-constrained.

D-A1-06 locks ``mode: Literal["single-view", "multi-view"]`` with NO
default. A YAML without the ``mode:`` key must fail loudly at load time.

Covers:

- Omitting ``mode:`` raises ``ValidationError`` mentioning the field name.
- Setting ``mode: junk-value`` raises ``ValidationError`` mentioning the
  literal constraint or the offending value.
- The two valid values (``single-view`` + ``multi-view``) both round-trip
  cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config.loader import load_config

_YAML_WITH_PLACEHOLDER = """\
{MODE_LINE}
io:
  input_path: /tmp/in
  output_dir: /tmp/out
prompt:
  text: wheat
preprocessing:
  output_resolution_m: 0.05
projection:
  features: [intensity]
inference:
  sam2_checkpoint: /tmp/sam2.pt
d3d_extraction: {{}}
fusion: {{}}
"""


@pytest.mark.tier_a
def test_missing_mode_raises_validation_error(tmp_yaml_path: Path) -> None:
    """A YAML without ``mode:`` raises ValidationError naming ``mode``."""
    yaml_text = _YAML_WITH_PLACEHOLDER.format(MODE_LINE="")
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "mode" in msg


@pytest.mark.tier_a
def test_invalid_mode_value_raises_validation_error(tmp_yaml_path: Path) -> None:
    """A YAML with ``mode: junk-value`` raises ValidationError naming the
    Literal constraint or the offending value.
    """
    yaml_text = _YAML_WITH_PLACEHOLDER.format(MODE_LINE="mode: junk-value")
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "mode" in msg
    # pydantic v2 surfaces 'literal_error' for invalid Literal values.
    assert "literal_error" in msg or "junk-value" in msg


@pytest.mark.tier_a
def test_single_view_mode_round_trips(tmp_yaml_path: Path) -> None:
    """``mode: single-view`` round-trips cleanly."""
    yaml_text = _YAML_WITH_PLACEHOLDER.format(MODE_LINE="mode: single-view")
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(tmp_yaml_path)
    assert cfg.mode == "single-view"


@pytest.mark.tier_a
def test_multi_view_mode_round_trips(tmp_yaml_path: Path) -> None:
    """``mode: multi-view`` round-trips cleanly."""
    yaml_text = _YAML_WITH_PLACEHOLDER.format(MODE_LINE="mode: multi-view")
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(tmp_yaml_path)
    assert cfg.mode == "multi-view"
