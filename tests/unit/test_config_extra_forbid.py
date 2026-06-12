"""Tier-A CFG-02 assertion — unknown YAML keys raise ``ValidationError``.

The ``extra='forbid'`` invariant locked in D-A1-17 is the load-bearing
typo-catcher for the YAML schema. Pydantic v2's default is
``extra='ignore'`` (silently drops unknown keys), which would let typos
like ``box_threshhold`` (extra ``h``) slip through unnoticed.

This file regression-locks ``ValidationError`` (type ``extra_forbidden``)
on:

- Unknown key at the top level of RunConfig (``typo_key:`` at root).
- Unknown key inside a sub-model (``box_threshhold:`` inside
  ``inference:``).
- Legacy hyphenated key (``sam2-checkpoint:`` inside ``inference:``).
- Unknown key inside the deeply-nested SlicingConfig.

These tests are explicitly ``tier_a``: no pchandler/pc2img imports,
filesystem confined to ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config.loader import load_config


@pytest.mark.tier_a
def test_typo_at_root_raises_extra_forbidden(tmp_yaml_path: Path) -> None:
    """A top-level typo'd key raises ValidationError mentioning extra_forbidden
    or the offending key name.
    """
    yaml_text = """\
mode: single-view
typo_key_at_root: oops
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
  type: grounded_sam2
  sam2_checkpoint: /tmp/sam2.pt
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "extra_forbidden" in msg or "typo_key_at_root" in msg


@pytest.mark.tier_a
def test_typo_in_inference_block_raises_extra_forbidden(tmp_yaml_path: Path) -> None:
    """The canonical typo case — ``box_threshhold`` (extra h) inside
    ``inference:`` — raises ValidationError loudly.
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
projection:
  features: [intensity]
inference:
  type: grounded_sam2
  sam2_checkpoint: /tmp/sam2.pt
  box_threshhold: 0.10
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "extra_forbidden" in msg or "box_threshhold" in msg


@pytest.mark.tier_a
def test_legacy_hyphenated_key_rejected(tmp_yaml_path: Path) -> None:
    """Legacy ``sam2-checkpoint`` (hyphenated) is rejected — snake_case only
    per D-A1-03. The schema canonicalizes to ``sam2_checkpoint``.
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
projection:
  features: [intensity]
inference:
  type: grounded_sam2
  sam2-checkpoint: /tmp/sam2.pt
  sam2_checkpoint: /tmp/sam2.pt
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "extra_forbidden" in msg or "sam2-checkpoint" in msg


@pytest.mark.tier_a
def test_typo_in_slicing_subblock_raises_extra_forbidden(tmp_yaml_path: Path) -> None:
    """``extra='forbid'`` applies to the deeply-nested SlicingConfig too
    (D-A1-17: "No exceptions for v1").
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
projection:
  features: [intensity]
inference:
  type: grounded_sam2
  sam2_checkpoint: /tmp/sam2.pt
  slicing:
    enabled: true
    unknown_slicing_field: 1
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_config(tmp_yaml_path)
    msg = str(exc_info.value)
    assert "extra_forbidden" in msg or "unknown_slicing_field" in msg
