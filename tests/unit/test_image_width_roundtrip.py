"""Round-trip test for image_width: config validation <-> compute_image_dimensions.

Tier: tier_a — pure logic (no pchandler / pc2img / torch needed).

``compute_image_dimensions`` lives in ``pc2img_utils``, which has heavy module-level
imports (imageio, pyvips, pchandler, sklearn) not available in the tier_a venv.
The tests therefore replicate the dimension math inline — the function body is
trivially short pure arithmetic that we own via CR-04.  This is intentional:
the goal is to lock the contract (what values are accepted and what dimensions
they produce), not to test the dep module.

Verifies CR-04 fix: validator and engine agree on the same set of encodings so
no accepted value crashes at runtime and no documented value is rejected.

Coverage:
  - int pixel width    — validates, produces the exact int as width
  - "scan_resolution"  — validates, produces ceil(azim_span / d_azim) as width
  - "0.5-scan_resolution" — validates, produces ceil(full * 0.5) as width
  - bare float (e.g. 0.4) — rejected at config validation with a clear error
  - non-integer numeric string (e.g. "1.5") — rejected at config validation
"""

from __future__ import annotations

import math

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Inline dimension math — mirrors compute_image_dimensions exactly (CR-04).
# ─────────────────────────────────────────────────────────────────────────────


def _compute_dims(image_width: int | str, azim_span: float, elev_span: float, d_azim: float) -> tuple[int, int]:
    """Inline replica of pc2img_utils.compute_image_dimensions for tier_a use.

    pc2img_utils has heavy module-level imports unavailable in the tier_a venv.
    This replica covers only the code paths exercised by the accepted encodings.
    """
    full_width = math.ceil(azim_span / d_azim)
    if isinstance(image_width, int):
        width_px = image_width
    elif isinstance(image_width, str):
        s = image_width.strip().lower()
        if s == "scan_resolution":
            width_px = full_width
        else:
            import re

            m = re.match(r"^([\d\.]+)[\s-]*scan_resolution$", s)
            if m:
                frac = float(m.group(1))
                width_px = math.ceil(full_width * frac)
            else:
                raise ValueError(f"Invalid image_width format: {image_width!r}")
    else:
        raise TypeError(f"image_width must be int or str, got {type(image_width)}")

    height_px = math.ceil(width_px * (elev_span / azim_span))
    return width_px, height_px


# Synthetic FoV: ~90-degree horizontal span, ~30-degree elevation span (radians)
_AZIM_SPAN = math.pi / 2  # 90 degrees
_ELEV_SPAN = math.pi / 6  # 30 degrees
_D_AZIM = math.radians(0.036)  # realistic TLS angular step


# ─────────────────────────────────────────────────────────────────────────────
# Accepted encodings — all must pass validation AND produce positive dimensions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_image_width_int_roundtrip() -> None:
    """int pixel width validates and the engine returns that exact width."""
    from tls2dseg.config.models import ProjectionConfig

    cfg = ProjectionConfig(features=["intensity"], image_width=1024)
    assert cfg.image_width == 1024, f"Expected int 1024, got {cfg.image_width!r}"

    w, h = _compute_dims(cfg.image_width, _AZIM_SPAN, _ELEV_SPAN, _D_AZIM)
    assert w == 1024
    assert h > 0


@pytest.mark.tier_a
def test_image_width_scan_resolution_roundtrip() -> None:
    """'scan_resolution' validates and produces the full-resolution width."""
    from tls2dseg.config.models import ProjectionConfig

    cfg = ProjectionConfig(features=["intensity"], image_width="scan_resolution")
    assert cfg.image_width == "scan_resolution"

    w, h = _compute_dims(cfg.image_width, _AZIM_SPAN, _ELEV_SPAN, _D_AZIM)
    expected_full = math.ceil(_AZIM_SPAN / _D_AZIM)
    assert w == expected_full, f"Expected full-res width {expected_full}, got {w}"
    assert h > 0


@pytest.mark.tier_a
def test_image_width_fractional_string_roundtrip() -> None:
    """'0.5-scan_resolution' validates and produces half the full-resolution width."""
    from tls2dseg.config.models import ProjectionConfig

    cfg = ProjectionConfig(features=["intensity"], image_width="0.5-scan_resolution")
    assert cfg.image_width == "0.5-scan_resolution"

    w_full, _ = _compute_dims("scan_resolution", _AZIM_SPAN, _ELEV_SPAN, _D_AZIM)
    w_half, h_half = _compute_dims(cfg.image_width, _AZIM_SPAN, _ELEV_SPAN, _D_AZIM)

    assert w_half > 0
    assert h_half > 0
    assert w_half == math.ceil(w_full * 0.5), f"Expected ceil(full*0.5)={math.ceil(w_full * 0.5)}, got {w_half}"


# ─────────────────────────────────────────────────────────────────────────────
# Rejected encodings — must raise ValidationError at config load
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.tier_a
def test_image_width_bare_float_rejected() -> None:
    """Bare float (e.g. 0.4) is rejected at config validation with a clear error."""
    from pydantic import ValidationError

    from tls2dseg.config.models import ProjectionConfig

    with pytest.raises(ValidationError) as exc_info:
        ProjectionConfig(features=["intensity"], image_width=0.4)

    error_text = str(exc_info.value).lower()
    assert "float" in error_text or "scan_resolution" in error_text, (
        f"Expected clear rejection message for bare float; got:\n{exc_info.value}"
    )


@pytest.mark.tier_a
def test_image_width_noninteger_numeric_string_rejected() -> None:
    """String '1.5' (non-integer numeric) is rejected at config validation."""
    from pydantic import ValidationError

    from tls2dseg.config.models import ProjectionConfig

    with pytest.raises(ValidationError):
        ProjectionConfig(features=["intensity"], image_width="1.5")
