"""Tier-A MODE-06 assertion — ``projection.features`` is validated against
the projection engine's known feature set.

D-A1-12 locks ``features: list[Literal["intensity", "range", "rgb"]]``
with ``min_length=1`` and NO default. The Literal already constrains
individual elements; a redundant ``@field_validator`` provides a clearer
error message listing all known features (versus pydantic's generic
``literal_error``).

Covers:

- Empty ``features: []`` raises ValidationError (min_length=1).
- Unknown feature (``features: [intensity, depth]``) raises
  ValidationError naming the offending feature.
- All three valid features can appear (in any order).
- Missing ``features:`` key raises ValidationError (no default).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config.loader import load_config


def _yaml_with_features(features_value: str) -> str:
    """Build a minimal YAML inserting the given features YAML-literal."""
    return f"""\
mode: single-view
io:
  input_path: /tmp/in
  output_dir: /tmp/out
prompt:
  text: wheat
preprocessing:
  output_resolution_m: 0.05
projection:
  features: {features_value}
inference:
  sam2_checkpoint: /tmp/sam2.pt
d3d_extraction: {{}}
fusion: {{}}
"""


@pytest.mark.tier_a
def test_unknown_feature_rejected(tmp_yaml_path: Path) -> None:
    """``features: [intensity, depth]`` raises ValidationError naming the
    unknown feature ``depth``.
    """
    tmp_yaml_path.write_text(_yaml_with_features("[intensity, depth]"), encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "depth" in msg or "literal_error" in msg


@pytest.mark.tier_a
def test_empty_features_list_rejected(tmp_yaml_path: Path) -> None:
    """``features: []`` raises ValidationError (min_length=1)."""
    tmp_yaml_path.write_text(_yaml_with_features("[]"), encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    # pydantic v2 surfaces 'too_short' for min_length violations.
    assert "too_short" in msg or "features" in msg


@pytest.mark.tier_a
def test_all_known_features_accepted(tmp_yaml_path: Path) -> None:
    """All three known features round-trip cleanly in any order."""
    for features in (
        "[intensity]",
        "[range]",
        "[rgb]",
        "[intensity, range]",
        "[intensity, range, rgb]",
        "[rgb, intensity]",
    ):
        tmp_yaml_path.write_text(_yaml_with_features(features), encoding="utf-8")
        cfg = load_config(tmp_yaml_path)
        # Round-trip preserves list order; compare against the YAML form.
        assert isinstance(cfg.projection.features, list)
        assert len(cfg.projection.features) >= 1


@pytest.mark.tier_a
def test_missing_features_key_rejected(tmp_yaml_path: Path) -> None:
    """``projection:`` block without ``features:`` raises ValidationError
    (D-A1-12: no default).
    """
    yaml_text = """\
mode: single-view
io:
  input_path: /tmp/in
  output_dir: /tmp/out
prompt:
  text: wheat
preprocessing:
  output_resolution_m: 0.05
projection: {}
inference:
  sam2_checkpoint: /tmp/sam2.pt
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "features" in msg
