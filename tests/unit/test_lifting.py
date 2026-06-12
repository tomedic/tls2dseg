"""TEST-05 — img_1to3_channels_encoding edge-case coverage + lifting import guard.

Phase 4 plan 02 Task 1. Locks the tls2dseg-side rasterization helper
``img_1to3_channels_encoding`` against four edge cases that are critical for
stable DL-framework ingestion (D-D-04 wrappers-only rule):

* NaN-containing float array → valid 3-channel float32 in [0,1]
* Constant image (zero variance) → no divide-by-zero; output is all-zeros
* broadcast=True vs broadcast=False → correct output shapes
* output_dtype enforcement → returned array has the requested dtype

NOTE: ``img_1to3_channels_encoding`` currently lives in ``tls2dseg.pc2img_utils``
which has module-level ``pyvips``/``pc2img``/``pchandler`` imports.  Therefore the
four encoding tests are marked ``tier_b_light`` until Plan 04-04 Task 3 moves the
function to ``tls2dseg.engines.inference.shared`` (pure numpy, no heavy deps).
After 04-04, the import path in each test function is updated and the marker
upgrades to ``tier_a``.  Do NOT remove this comment before 04-04 is done.

pc2img_utils requires pyvips + pchandler (+ pchandler.geometry → cudf on GPU
machines) at module load.  If any of those fail the four encoding tests are
skipped so that ``nox -s tier_b_light`` remains green on machines without
RAPIDS/cudf.  The broad try/except follows the PATTERNS.md "pchandler import
guard" pattern.

The lifting module import-cost guard (``test_lifting_module_importable_no_heavy_deps``)
IS tier_a — it only imports ``tls2dseg.lifting.masks_to_pcd`` which must be
importable in a no-deps venv.
"""

from __future__ import annotations

import numpy as np
import pytest

# ------------------------------------------------------------------
# lifting/ module import-cost guard (tier_a)
# ------------------------------------------------------------------


@pytest.mark.tier_a
def test_lifting_module_importable_no_heavy_deps() -> None:
    """``tls2dseg.lifting.masks_to_pcd`` is importable with no pchandler/pc2img at module level.

    Locks: D-A-05 heavy-import contract for the lifting module.
    The lift_* functions need pchandler only at call time, not at import time.
    """
    import sys

    before = set(sys.modules)
    from tls2dseg.lifting.masks_to_pcd import lift_mask_to_pcd, lift_masks_to_pcd

    leaked = {"pchandler", "pc2img", "pyvips", "torch", "sam2", "transformers"} & (set(sys.modules) - before)
    assert not leaked, f"Heavy imports leaked at lifting module load: {leaked}"


# ------------------------------------------------------------------
# img_1to3_channels_encoding — TEST-05 edge cases
#
# Marked tier_b_light; import is attempted at module level with a broad
# try/except so that collection does not fail when pyvips/cudf is absent.
# Plan 04-04 Task 3 will re-point to tls2dseg.engines.inference.shared
# (pure numpy) and upgrade these tests to tier_a.
# ------------------------------------------------------------------

try:
    from tls2dseg.pc2img_utils import img_1to3_channels_encoding as _img_encode

    _SKIP_REASON = ""
except Exception as _import_exc:
    _img_encode = None  # type: ignore[assignment]
    _SKIP_REASON = (
        f"pc2img_utils not importable (pyvips/pchandler/cudf absent): {_import_exc!r}. "
        "04-04 will re-point import to engines/inference/shared (tier_a)."
    )

_skip_img = pytest.mark.skipif(_img_encode is None, reason=_SKIP_REASON)


@pytest.mark.tier_b_light
@_skip_img
def test_img_1to3_nan_input_returns_valid_float32_in_0_1() -> None:
    """NaN-containing input array → valid 3-channel float32 in [0, 1].

    Locks: NaN handling in step 1 (replace_nan_with='max' default) +
    normalise-to-0-1 in step 2. Output must have no NaNs, dtype=float32,
    shape (H, W, 3), values in [0, 1].

    04-04 re-points this import to tls2dseg.engines.inference.shared and
    upgrades this test to tier_a.
    """
    img_1to3_channels_encoding = _img_encode

    rng = np.random.default_rng(42)
    img = rng.random((8, 12)).astype(np.float64)
    img[2, 3] = np.nan
    img[5, 7] = np.nan

    result = img_1to3_channels_encoding(img, output_dtype="float32", normalize="0-1", broadcast=False)

    assert result.shape == (8, 12, 3), f"Expected (8, 12, 3), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    assert not np.any(np.isnan(result)), "NaNs survived encoding"
    assert float(result.min()) >= 0.0, "Values below 0.0 after normalisation"
    assert float(result.max()) <= 1.0, f"Values above 1.0 after normalisation: {result.max()}"


@pytest.mark.tier_b_light
@_skip_img
def test_img_1to3_constant_image_no_divide_by_zero() -> None:
    """Constant image (img_max == img_min) → output is all-zeros, no ZeroDivisionError.

    Locks: the constant-slice guard at step 2:
        ``np.zeros_like(img) if img_max == img_min else (img - img_min) / (img_max - img_min)``
    A constant array would divide by zero without the guard.

    04-04 re-points this import to tls2dseg.engines.inference.shared and
    upgrades this test to tier_a.
    """
    img_1to3_channels_encoding = _img_encode

    img = np.full((4, 6), fill_value=7.5, dtype=np.float32)

    # Must not raise; constant image → normalised to all-zeros
    result = img_1to3_channels_encoding(img, output_dtype="float32", normalize="0-1", broadcast=False)

    assert result.shape == (4, 6, 3)
    assert float(result.max()) == 0.0, "Constant image should normalise to all-zeros"
    assert not np.any(np.isnan(result))


@pytest.mark.tier_b_light
@_skip_img
def test_img_1to3_broadcast_true_vs_false_shapes() -> None:
    """broadcast=True returns a read-only broadcast view; broadcast=False returns full copy.

    Locks: step 4 — both paths must produce shape (H, W, 3) and identical values.

    04-04 re-points this import to tls2dseg.engines.inference.shared and
    upgrades this test to tier_a.
    """
    img_1to3_channels_encoding = _img_encode

    rng = np.random.default_rng(0)
    img = rng.random((5, 7)).astype(np.float32)

    result_bcast = img_1to3_channels_encoding(img.copy(), output_dtype="float32", normalize="0-1", broadcast=True)
    result_full = img_1to3_channels_encoding(img.copy(), output_dtype="float32", normalize="0-1", broadcast=False)

    assert result_bcast.shape == (5, 7, 3), f"broadcast=True shape wrong: {result_bcast.shape}"
    assert result_full.shape == (5, 7, 3), f"broadcast=False shape wrong: {result_full.shape}"
    np.testing.assert_array_almost_equal(
        result_bcast,
        result_full,
        decimal=6,
        err_msg="broadcast=True and broadcast=False must produce same values",
    )


@pytest.mark.tier_b_light
@_skip_img
def test_img_1to3_output_dtype_enforced() -> None:
    """output_dtype parameter is respected for both 'float32' and 'uint8'.

    Locks: step 3 dtype cast. float32 and uint8 are the two DL-framework-relevant
    dtypes; None (no cast) is also supported.

    04-04 re-points this import to tls2dseg.engines.inference.shared and
    upgrades this test to tier_a.
    """
    img_1to3_channels_encoding = _img_encode

    rng = np.random.default_rng(1)
    img = rng.random((4, 4)).astype(np.float64)

    result_f32 = img_1to3_channels_encoding(img.copy(), output_dtype="float32", normalize="0-1")
    result_u8 = img_1to3_channels_encoding(img.copy(), output_dtype="uint8", normalize="0-255")

    assert result_f32.dtype == np.float32, f"Expected float32, got {result_f32.dtype}"
    assert result_u8.dtype == np.uint8, f"Expected uint8, got {result_u8.dtype}"
