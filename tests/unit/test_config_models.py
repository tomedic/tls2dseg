"""Tier-A tests for :mod:`tls2dseg.config.models` — round-trip locks for
all 10 top-level RunConfig blocks (CFG-01).

Covers:

- :class:`RunConfig.model_config` frozen + extra='forbid' invariant.
- All 11 model classes are instantiable through ``load_config`` with the
  minimal YAML covering every required-no-default field.
- Sub-model defaults flow correctly when omitted from YAML (``runtime``
  and ``logging`` blocks may be absent — the others have required fields).
- Required-no-default fields raise :class:`pydantic.ValidationError` when
  omitted.
- Frozen-instance behavior: attribute assignment on a constructed
  RunConfig raises ``ValidationError(type='frozen_instance')``, NOT
  AttributeError (Pitfall 4 in RESEARCH.md).
- Discriminator default values for the Phase 4 ENG-* expansion slots
  (``projection.type='spherical'``, ``inference.type='grounded_sam2'``,
  ``fusion.type='graph_cluster'``).

These tests are explicitly ``tier_a``: no pchandler/pc2img imports, no
GPU, no network, no disk I/O outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tls2dseg.config import (
    D3DExtractionConfig,
    FusionConfig,
    InferenceConfig,
    IOConfig,
    LoggingConfig,
    PreprocessingConfig,
    ProjectionConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
    SlicingConfig,
)
from tls2dseg.config.loader import load_config
from tls2dseg.config.models import GroundedSAM2Config, GroundedSAM2HFConfig, InferenceSharedConfig


@pytest.mark.tier_a
def test_runconfig_frozen_and_extra_forbid_invariants() -> None:
    """RunConfig advertises frozen=True + extra='forbid' (CFG-02, D-A1-05/17)."""
    assert RunConfig.model_config.get("frozen") is True
    assert RunConfig.model_config.get("extra") == "forbid"


@pytest.mark.tier_a
def test_every_sub_model_has_extra_forbid_and_frozen() -> None:
    """Every sub-model has extra='forbid' + frozen=True (D-A1-17).

    InferenceConfig is now a discriminated union (type alias); the concrete
    sub-models (InferenceSharedConfig, GroundedSAM2Config, GroundedSAM2HFConfig)
    are checked individually (D-C-02).
    """
    sub_models = [
        IOConfig,
        RuntimeConfig,
        PromptConfig,
        PreprocessingConfig,
        ProjectionConfig,
        SlicingConfig,
        # Inference discriminated union — concrete sub-models (D-C-02)
        InferenceSharedConfig,
        GroundedSAM2Config,
        GroundedSAM2HFConfig,
        D3DExtractionConfig,
        FusionConfig,
        LoggingConfig,
    ]
    for m in sub_models:
        assert m.model_config.get("extra") == "forbid", f"{m.__name__} missing extra=forbid"
        assert m.model_config.get("frozen") is True, f"{m.__name__} missing frozen=True"


@pytest.mark.tier_a
def test_every_field_carries_tag_annotation() -> None:
    """Every field on every model has a ``tag`` annotation in the
    locked vocabulary {primary, tuning, pipings, other} (D-A1-04, prep for DOC-02).

    InferenceConfig is a discriminated union (type alias); the concrete
    sub-models are checked individually (D-C-02).
    """
    allowed_tags = {"primary", "tuning", "pipings", "other"}
    all_models = [
        RunConfig,
        IOConfig,
        RuntimeConfig,
        PromptConfig,
        PreprocessingConfig,
        ProjectionConfig,
        SlicingConfig,
        # Inference discriminated union — concrete sub-models (D-C-02)
        InferenceSharedConfig,
        GroundedSAM2Config,
        GroundedSAM2HFConfig,
        D3DExtractionConfig,
        FusionConfig,
        LoggingConfig,
    ]
    for m in all_models:
        for fname, finfo in m.model_fields.items():
            extra = finfo.json_schema_extra
            assert isinstance(extra, dict), f"{m.__name__}.{fname} missing json_schema_extra dict"
            tag = extra.get("tag")
            assert tag in allowed_tags, f"{m.__name__}.{fname} tag={tag!r} not in {allowed_tags}"


@pytest.mark.tier_a
def test_load_config_minimal_yaml_round_trip(minimal_yaml_path: Path) -> None:
    """The minimal YAML covering every required-no-default field
    round-trips through ``load_config`` into a fully-validated RunConfig
    with all 10 top-level blocks populated.
    """
    cfg = load_config(minimal_yaml_path)

    # All 10 top-level blocks resolved.
    assert isinstance(cfg.io, IOConfig)
    assert isinstance(cfg.runtime, RuntimeConfig)
    assert isinstance(cfg.prompt, PromptConfig)
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert isinstance(cfg.projection, ProjectionConfig)
    # InferenceConfig is a discriminated union type alias; use the base shared class
    assert isinstance(cfg.inference, InferenceSharedConfig)
    assert isinstance(cfg.inference.slicing, SlicingConfig)
    assert isinstance(cfg.d3d_extraction, D3DExtractionConfig)
    assert isinstance(cfg.fusion, FusionConfig)
    assert isinstance(cfg.logging, LoggingConfig)

    # mode flowed in
    assert cfg.mode == "single-view"

    # Required-no-default fields populated.
    assert cfg.io.input_path == Path("/tmp/tls2dseg_test_input")
    assert cfg.io.output_dir == Path("/tmp/tls2dseg_test_output")
    assert cfg.prompt.text == "wheat."  # normalizer added trailing period
    assert cfg.preprocessing.output_resolution_m == 0.05
    assert cfg.projection.features == ["intensity", "range"]
    assert cfg.inference.sam2_checkpoint == Path("/tmp/tls2dseg_test_sam2.pt")


@pytest.mark.tier_a
def test_sub_block_defaults_flow_when_omitted(minimal_yaml_path: Path) -> None:
    """RuntimeConfig + LoggingConfig sub-blocks default cleanly when
    absent from YAML (both consist entirely of defaulted fields).
    """
    cfg = load_config(minimal_yaml_path)

    # runtime defaults (D-A1-08)
    assert cfg.runtime.device == "auto"
    assert cfg.runtime.n_workers == 12
    assert cfg.runtime.accept_cpu_fallback is True

    # logging defaults (D-A1-10)
    assert cfg.logging.level == "INFO"
    assert cfg.logging.per_package == {}
    assert cfg.logging.log_to_file is True

    # slicing nested default (D-A1-14)
    assert cfg.inference.slicing.enabled is True
    assert cfg.inference.slicing.iou_threshold == 0.80

    # io defaults
    assert cfg.io.file_format == "e57"
    assert cfg.io.resume_from_checkpoint is True

    # d3d_extraction defaults (D-A1-15)
    assert cfg.d3d_extraction.bounding_box_type == "obb"
    assert cfg.d3d_extraction.min_point_count == 50

    # fusion defaults (D-A1-16)
    assert cfg.fusion.graph_clustering_method == "hcs"
    assert cfg.fusion.outlier_detection_method == "negative_binomial"


@pytest.mark.tier_a
def test_phase4_discriminator_defaults_present(minimal_yaml_path: Path) -> None:
    """Phase 4 ENG-* discriminator fields ship with the expected single-value
    Literal defaults (D-A1-12/13/16).
    """
    cfg = load_config(minimal_yaml_path)
    assert cfg.projection.type == "spherical"
    assert cfg.inference.type == "grounded_sam2"
    assert cfg.fusion.type == "graph_cluster"


@pytest.mark.tier_a
def test_runconfig_frozen_assignment_raises_validation_error(
    minimal_yaml_path: Path,
) -> None:
    """Attribute assignment on a constructed RunConfig raises pydantic
    ValidationError (NOT AttributeError per Pitfall 4 in RESEARCH.md).
    """
    cfg = load_config(minimal_yaml_path)
    with pytest.raises(ValidationError) as exc_info:
        cfg.mode = "multi-view"  # type: ignore[misc]
    # pydantic v2 uses error type 'frozen_instance' for this case.
    assert "frozen" in str(exc_info.value).lower()


@pytest.mark.tier_a
def test_prompt_text_lowercase_and_trailing_period() -> None:
    """PromptConfig.text validator enforces lowercase + trailing period (D-A1-09)."""
    assert PromptConfig(text="WHEAT").text == "wheat."
    assert PromptConfig(text="wheat. wheat ear.").text == "wheat. wheat ear."
    assert PromptConfig(text="  Wheat Ear  ").text == "wheat ear."


@pytest.mark.tier_a
def test_image_width_value_type_union_dispatch() -> None:
    """ProjectionConfig.image_width accepts int, 'scan_resolution', and '<frac>-scan_resolution'.

    CR-04: bare floats are rejected — use '<frac>-scan_resolution' for fractional widths.
    """
    from pydantic import ValidationError

    p_int = ProjectionConfig(features=["intensity"], image_width=800)
    assert p_int.image_width == 800
    assert isinstance(p_int.image_width, int)

    p_lit = ProjectionConfig(features=["intensity"], image_width="scan_resolution")
    assert p_lit.image_width == "scan_resolution"

    p_frac = ProjectionConfig(features=["intensity"], image_width="0.5-scan_resolution")
    assert p_frac.image_width == "0.5-scan_resolution"

    with pytest.raises(ValidationError):
        ProjectionConfig(features=["intensity"], image_width=0.5)


@pytest.mark.tier_a
def test_rotate_pcd_false_is_preserved_as_bool() -> None:
    """rotate_pcd accepts ``False`` as a bool (not coerced to 0.0).

    Per feedback_yaml_natural_over_pythonic memory: ``false`` is the
    YAML-natural off-sentinel. pydantic v2 smart mode preserves bools
    correctly across Union[float, Literal["auto", False]].
    """
    p = ProjectionConfig(features=["intensity"], rotate_pcd=False)
    assert p.rotate_pcd is False
    p_auto = ProjectionConfig(features=["intensity"], rotate_pcd="auto")
    assert p_auto.rotate_pcd == "auto"
    p_float = ProjectionConfig(features=["intensity"], rotate_pcd=42.5)
    assert p_float.rotate_pcd == 42.5


@pytest.mark.tier_a
def test_scan_resolution_string_regex_validation() -> None:
    """scan_resolution string arm matches `<num><unit>@<dist>m` regex (D-A1-12)."""
    # Accepted forms
    p = ProjectionConfig(features=["intensity"], scan_resolution="1.6mm@10m")
    assert p.scan_resolution == "1.6mm@10m"
    p = ProjectionConfig(features=["intensity"], scan_resolution="auto")
    assert p.scan_resolution == "auto"
    # Numeric forms also accepted (validator only kicks in on string arm)
    p = ProjectionConfig(features=["intensity"], scan_resolution=0.001)
    assert p.scan_resolution == 0.001

    # Rejected — string but not matching regex
    with pytest.raises(ValidationError):
        ProjectionConfig(features=["intensity"], scan_resolution="oops_no_unit")


@pytest.mark.tier_a
def test_yaml_env_var_interpolation_with_var_set(tmp_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``${ENV_VAR}`` in a YAML string field expands to the env-var value
    (D-CD-05). Tests the loader-level interpolation, not pydantic's.
    """
    monkeypatch.setenv("SAM2_CHECKPOINT_PATH_TEST", "/tmp/expanded_sam2.pt")
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
  sam2_checkpoint: ${SAM2_CHECKPOINT_PATH_TEST}
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    cfg = load_config(tmp_yaml_path)
    assert cfg.inference.sam2_checkpoint == Path("/tmp/expanded_sam2.pt")


@pytest.mark.tier_a
def test_yaml_env_var_unresolved_raises_value_error(tmp_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``${UNSET_VAR}`` in YAML raises a ValueError whose message contains the
    literal ``${UNSET_VAR}`` reference (Pitfall 6 in RESEARCH.md).
    """
    monkeypatch.delenv("TLS2DSEG_UNSET_TEST_VAR", raising=False)
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
  sam2_checkpoint: ${TLS2DSEG_UNSET_TEST_VAR}
d3d_extraction: {}
fusion: {}
"""
    tmp_yaml_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_config(tmp_yaml_path)
    assert "${TLS2DSEG_UNSET_TEST_VAR}" in str(exc_info.value)


@pytest.mark.tier_a
def test_cli_overrides_take_precedence_over_yaml(minimal_yaml_path: Path) -> None:
    """``cli_overrides`` win over YAML values (D-A3-04)."""
    # Baseline from YAML
    cfg_base = load_config(minimal_yaml_path)
    assert cfg_base.inference.box_threshold == 0.10  # schema default; YAML omits it

    # CLI override at nested path
    cfg_cli = load_config(
        minimal_yaml_path,
        cli_overrides={
            "inference": {
                "box_threshold": 0.25,
                "sam2_checkpoint": "/tmp/tls2dseg_test_sam2.pt",
            }
        },
    )
    assert cfg_cli.inference.box_threshold == 0.25


@pytest.mark.tier_a
def test_env_vars_override_yaml_but_cli_overrides_env(minimal_yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-var ``TLS2DSEG_INFERENCE__BOX_THRESHOLD`` overrides YAML; CLI
    overrides override env vars (D-A3-04 precedence + Pitfall 5).
    """
    # Env over YAML
    monkeypatch.setenv("TLS2DSEG_INFERENCE__BOX_THRESHOLD", "0.30")
    cfg_env = load_config(minimal_yaml_path)
    assert cfg_env.inference.box_threshold == 0.30

    # CLI over env
    cfg_cli = load_config(
        minimal_yaml_path,
        cli_overrides={
            "inference": {
                "box_threshold": 0.40,
                "sam2_checkpoint": "/tmp/tls2dseg_test_sam2.pt",
            }
        },
    )
    assert cfg_cli.inference.box_threshold == 0.40
