"""CPU-05 lock-in — Runtime dataclass + probe_all() with monkeypatched find_spec.

Phase 3 plan 02 (Task 2). Locks in the contract of
``tls2dseg.runtime.capability``:

* :class:`Runtime` is a ``@dataclasses.dataclass(frozen=True)`` with the seven
  fields specified in CONTEXT.md D-A2-03 + REQUIREMENTS.md CPU-05.
* :func:`probe_all` returns a populated :class:`Runtime` whose ``*_available``
  bool fields are driven by ``importlib.util.find_spec`` — verifiable
  deterministically by monkeypatching ``find_spec``.
* Mutation of a :class:`Runtime` instance raises
  ``dataclasses.FrozenInstanceError`` (NOT ``pydantic.ValidationError`` per
  RESEARCH.md Pitfall 4 — Runtime is a stdlib dataclass, not a pydantic model).
* ``_safe_version`` returns None for missing distributions (never raises
  ``PackageNotFoundError``) per RESEARCH.md §Pattern 3.

Marked ``tier_a`` per CONTEXT.md D-A4-01 — these tests import ONLY stdlib and
``tls2dseg.runtime`` (never pchandler/pc2img), so they are safe to run in
cloud CI under ``pip install --no-deps``. Determinism comes from
``monkeypatch.setattr`` on ``importlib.util.find_spec``; no real GPU, no
network, no large fixtures.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

from tls2dseg.runtime import Runtime, probe_all
from tls2dseg.runtime.capability import _probe, _safe_version


@pytest.mark.tier_a
def test_runtime_is_frozen_dataclass() -> None:
    """``Runtime`` is a frozen stdlib dataclass with the seven CPU-05 fields.

    Locks: CONTEXT.md D-A2-03 field set, dataclass-not-pydantic shape.
    """
    assert dataclasses.is_dataclass(Runtime), (
        "Runtime must be a dataclass (not a pydantic BaseModel) per RESEARCH "
        "Pitfall 4 — mutation must raise FrozenInstanceError, not ValidationError"
    )
    assert Runtime.__dataclass_params__.frozen, (
        "Runtime must be frozen=True so RunContext can stamp a single snapshot at process start (CPU-05)"
    )

    field_names = {f.name for f in dataclasses.fields(Runtime)}
    expected = {
        "cuml_available",
        "torch_cuda_available",
        "sam2_available",
        "libvips_available",
        "numpy_version",
        "torch_version",
        "cuml_version",
    }
    assert field_names == expected, f"Runtime fields drifted from D-A2-03 spec: {field_names} != {expected}"


@pytest.mark.tier_a
def test_runtime_mutation_raises_frozen_instance_error() -> None:
    """``rt.cuml_available = True`` raises ``FrozenInstanceError`` (not AttributeError).

    Locks: RESEARCH Pitfall 4 — Runtime is a dataclass, not a pydantic model,
    so the immutability test must assert ``dataclasses.FrozenInstanceError``
    specifically (a plain ``AttributeError`` would accept a non-frozen
    dataclass with read-only properties — that is NOT the contract).
    """
    rt = probe_all()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rt.cuml_available = True  # type: ignore[misc]


@pytest.mark.tier_a
def test_probe_all_returns_runtime_with_bool_fields() -> None:
    """``probe_all()`` returns a Runtime whose availability fields are real bools.

    Locks: CPU-05 — the availability snapshot is a flat dataclass with bool
    fields (not Optional[bool], not int, not str). RunContext.device
    resolution in plan 04 will do ``"cuda" if rt.torch_cuda_available else "cpu"``
    — a non-bool would silently miscompare.
    """
    rt = probe_all()
    assert isinstance(rt, Runtime)
    assert isinstance(rt.cuml_available, bool)
    assert isinstance(rt.torch_cuda_available, bool)
    assert isinstance(rt.sam2_available, bool)
    assert isinstance(rt.libvips_available, bool)


@pytest.mark.tier_a
def test_probe_all_all_modules_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_spec returning None for every module → all availability fields False.

    Drives ``importlib.util.find_spec`` to always return ``None`` (simulates a
    minimal cloud-CI environment with no GPU stack installed). All four
    ``*_available`` bools must be False — including ``torch_cuda_available``,
    because ``probe_all`` gates the ``torch.cuda.is_available()`` call behind
    a ``_probe("torch")`` check, which now returns False.
    """
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    rt = probe_all()

    assert rt.cuml_available is False
    assert rt.torch_cuda_available is False
    assert rt.sam2_available is False
    assert rt.libvips_available is False


@pytest.mark.tier_a
def test_probe_all_partial_modules_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only sam2 + pyvips present → sam2_available + libvips_available True; others False.

    Demonstrates per-module dispatch: ``_probe`` consults the module name
    passed to ``find_spec``, so different return values for different names
    produce different ``Runtime`` field values. cuml False (no module).
    torch_cuda False (no torch module → no inner import → False).
    """
    present = {"sam2", "pyvips"}

    def fake_find_spec(name: str, package: str | None = None) -> object | None:
        # find_spec returns a ModuleSpec object on hit, None on miss. _probe
        # only cares about non-None — any sentinel object suffices.
        return object() if name in present else None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    rt = probe_all()

    assert rt.cuml_available is False
    assert rt.torch_cuda_available is False  # no torch → inner import skipped
    assert rt.sam2_available is True
    assert rt.libvips_available is True


@pytest.mark.tier_a
def test_probe_returns_bool_for_missing_module() -> None:
    """``_probe`` for an obviously-missing module name returns False (does not raise).

    Locks: ``find_spec`` is the canonical "is this importable?" probe per
    RESEARCH §"Don't Hand-Roll" — it MUST NOT execute the target module's
    code, so the impl must use ``find_spec`` (not ``try: import X``).
    """
    # A module name with characters illegal in Python identifiers and a
    # dotted-path that cannot exist on disk — find_spec returns None.
    assert _probe("definitely-not-a-real-module-xyz123") is False


@pytest.mark.tier_a
def test_safe_version_returns_none_for_missing_package() -> None:
    """``_safe_version`` for a missing distribution returns None (not PackageNotFoundError).

    Locks: RESEARCH §Pattern 3 — version probes are used to populate provenance
    ``env.txt``; raising on missing packages would force every call-site to
    wrap the lookup, defeating the helper's purpose.
    """
    assert _safe_version("definitely-not-a-real-pkg-xyz-987") is None


@pytest.mark.tier_a
def test_runtime_re_export_surface() -> None:
    """``from tls2dseg.runtime import Runtime, probe_all`` succeeds (re-export contract).

    Locks: ``runtime/__init__.py`` exports the two public symbols this plan
    delivers. Plan 04 will extend ``__all__`` with ``RunContext`` and
    ``build_context``; until then, this is the full surface.
    """
    from tls2dseg.runtime import __all__ as runtime_all

    assert "Runtime" in runtime_all
    assert "probe_all" in runtime_all
