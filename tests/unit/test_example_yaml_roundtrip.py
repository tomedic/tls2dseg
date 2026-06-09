"""CFG-07 — all 3 shipped example YAMLs round-trip cleanly through load_config().

Locks the worked-example contract for downstream users: a researcher cloning
the repo gets three runnable starting points (one per reference dataset per
D-A3-05), and each one parses through :func:`tls2dseg.config.loader.load_config`
without raising :class:`pydantic.ValidationError`.

Each test sets ``SAM2_CHECKPOINT_PATH`` via :meth:`pytest.MonkeyPatch.setenv`
BEFORE calling :func:`load_config` so the ``${SAM2_CHECKPOINT_PATH}``
interpolation in the YAML resolves (D-A1-13 + CFG-03 — the YAML never ships a
hardcoded ``/scratch/`` path, the credential lives in the user's env).

Path resolution uses ``Path(__file__).parents[2] / "examples" / "configs"`` so
the test is cwd-independent — running from anywhere under ``tls2dseg/`` works.

Marked ``tier_a`` per Phase 1 D-17 — pure-Python, no GPU, no pchandler/pc2img
imports, no I/O outside the read-only YAML file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tls2dseg.config.loader import load_config

# Repo root: tests/unit/test_X.py → parents[0]=unit, parents[1]=tests, parents[2]=tls2dseg/.
_EXAMPLES_CONFIGS_DIR = Path(__file__).parents[2] / "examples" / "configs"


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("yaml_name", "expected_mode"),
    [
        ("office.yaml", "multi-view"),
        ("agricultural.yaml", "multi-view"),
        ("mountain.yaml", "single-view"),
    ],
)
def test_example_yaml_roundtrips_through_load_config(
    yaml_name: str,
    expected_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each shipped example YAML parses cleanly with SAM2_CHECKPOINT_PATH set.

    Asserts:

    1. The YAML file actually exists under ``tls2dseg/examples/configs/`` —
       guards against accidental rename / delete from a future cleanup pass.
    2. :func:`load_config` returns a valid :class:`RunConfig` instance — no
       :class:`pydantic.ValidationError` and no :class:`ValueError` from the
       :class:`ExpandingYamlSource` env-var guard.
    3. ``cfg.mode`` matches the dataset-specific expectation (D-A3-05).
    4. ``cfg.prompt.text`` is non-empty (PromptConfig validator already
       lowercases + ensures trailing period; here we lock that *something*
       lands in the field, not the exact text — that would couple too
       tightly to the YAML's class-list copy and brittle the test on
       future class-list tweaks).
    5. ``cfg.projection.features`` is a non-empty list (MODE-06; required
       min_length=1 already enforced by the model, this is a smoke check).
    """
    # ── Pre-flight: SAM2 env var (required by ${SAM2_CHECKPOINT_PATH} in every
    # ── shipped YAML). Use a fake path inside tmp_path — D-CD-05 makes Path
    # ── validation lazy, so a non-existent file is fine at config-load time.
    monkeypatch.setenv("SAM2_CHECKPOINT_PATH", str(tmp_path / "dummy_sam2.pt"))

    yaml_path = _EXAMPLES_CONFIGS_DIR / yaml_name
    assert yaml_path.exists(), (
        f"Example YAML missing: {yaml_path} — CFG-07 requires all 3 ship under tls2dseg/examples/configs/."
    )

    cfg = load_config(yaml_path)

    assert cfg.mode == expected_mode, (
        f"{yaml_name}: expected mode={expected_mode!r}, got {cfg.mode!r} — "
        f"D-A3-05 maps office=multi-view, agricultural=multi-view, mountain=single-view."
    )
    assert cfg.prompt.text, f"{yaml_name}: prompt.text must be non-empty (D-A1-09)."
    assert cfg.projection.features, f"{yaml_name}: projection.features must be a non-empty list (MODE-06)."
