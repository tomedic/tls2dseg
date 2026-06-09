"""CPU-06 lock-in — pytest.mark.gpu + pytest.mark.cpu_only registration.

Phase 3 plan 02 (Task 2). Phase 1 D-17 declared the markers
``tier_a / tier_b_light / tier_b_heavy / gpu / cpu_only`` in
``pyproject.toml`` ``[tool.pytest.ini_options].markers``. CPU-06 is the
"explicit-skip behavior" part of that contract:

* ``@pytest.mark.cpu_only`` marks tests that must NEVER attempt GPU access
  (cloud CI tier_a path; safe under ``pip install --no-deps``).
* ``@pytest.mark.gpu`` marks tests that require CUDA; the marker is
  registered and discoverable but cannot itself enforce a skip — that
  decision belongs to whichever tier the test is collected under
  (tier_b_heavy ↔ self-hosted runner, tier_a ↔ skip in cloud CI).

This module ships three trivial tests:

1. A ``@pytest.mark.cpu_only`` always-pass (verifies the marker is
   acceptable to pytest collection — an unknown marker triggers a
   ``PytestUnknownMarkWarning`` and, with ``-W error::pytest.PytestUnknownMarkWarning``,
   a collection failure).
2. A ``@pytest.mark.gpu`` always-pass with the same purpose for the gpu
   marker.
3. A registration check that inspects ``request.config.getini("markers")``
   and asserts both ``"gpu"`` and ``"cpu_only"`` strings appear — proves
   Phase 1 D-17 markers survive Phase 3 modifications to pyproject.toml.

All three carry ``@pytest.mark.tier_a`` so they ship in the always-on
cloud CI job. The ``gpu`` test does NOT touch torch/cuda — it is a marker
acceptance test, not a GPU test.
"""

from __future__ import annotations

import pytest


@pytest.mark.tier_a
@pytest.mark.cpu_only
def test_cpu_only_marker_is_collected() -> None:
    """A ``@pytest.mark.cpu_only`` test is collected and runs without warning.

    If the ``cpu_only`` marker is missing from pyproject.toml, pytest emits a
    ``PytestUnknownMarkWarning`` here. Under CI's strict-warnings config that
    surfaces as a collection failure, locking the Phase 1 D-17 declaration in
    place.
    """
    assert True


@pytest.mark.tier_a
@pytest.mark.gpu
def test_gpu_marker_is_collected() -> None:
    """A ``@pytest.mark.gpu`` test is collected and runs without warning.

    Same lock-in as cpu_only above but for the ``gpu`` marker. This test does
    NOT actually require a GPU — it's a marker-registration lock-in. Tests
    that genuinely require CUDA go in tier_b_heavy and are gated by tier
    selection at the pytest invocation level (Phase 1 D-17), not by this
    marker's presence/absence.
    """
    assert True


@pytest.mark.tier_a
def test_d17_markers_declared_in_pyproject(request: pytest.FixtureRequest) -> None:
    """Phase 1 D-17 markers (``gpu``, ``cpu_only``) are still in pyproject.toml.

    Reads the ``markers`` ini setting from the current pytest config and
    asserts both ``gpu`` and ``cpu_only`` appear as leading tokens on
    registered marker lines. This is the CPU-06 lock-in test — if a
    future refactor strips either marker from
    ``[tool.pytest.ini_options].markers``, this test fails loudly.
    """
    markers = request.config.getini("markers")
    # Each entry is a string like "gpu: requires CUDA" — match on the prefix.
    marker_names = {entry.split(":", 1)[0].strip() for entry in markers}

    assert "gpu" in marker_names, (
        f"Phase 1 D-17 marker 'gpu' missing from pyproject.toml — registered markers: {sorted(marker_names)}"
    )
    assert "cpu_only" in marker_names, (
        f"Phase 1 D-17 marker 'cpu_only' missing from pyproject.toml — registered markers: {sorted(marker_names)}"
    )
