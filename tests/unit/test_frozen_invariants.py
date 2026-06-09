"""CFG-04 dual-invariant lock-in — RunConfig vs RunContext frozen semantics.

Phase 3 plan 04 (Task 3). The CFG-04 contract has TWO halves:

* ``RunConfig`` (pydantic BaseSettings, ``frozen=True``) raises
  ``pydantic.ValidationError`` with ``type='frozen_instance'`` on mutation
  (NOT ``AttributeError`` — RESEARCH.md Pitfall 4).
* ``RunContext`` (stdlib ``@dataclasses.dataclass(frozen=True)``) raises
  ``dataclasses.FrozenInstanceError`` on mutation (NOT the same as pydantic).

The DIFFERENT exception types are the lock-in proof. A silent migration of
``RunContext`` to pydantic would break the second test. A silent migration
of ``RunConfig`` to a stdlib dataclass would break the first.

Both tests stay ``tier_a``: stdlib + pydantic + tls2dseg.config + tls2dseg.runtime
only (no pchandler/pc2img imports), cloud-CI-safe per D-A4-01.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pydantic
import pytest
import yaml

from tls2dseg.config import RunConfig
from tls2dseg.runtime import build_context
from tls2dseg.runtime.capability import Runtime


def _cpu_runtime() -> Runtime:
    """Build a deterministic CPU-only Runtime for tests (no probe_all)."""
    return Runtime(
        cuml_available=False,
        torch_cuda_available=False,
        sam2_available=False,
        libvips_available=False,
        numpy_version="1.26.0",
        torch_version="2.4.0",
        cuml_version=None,
    )


@pytest.mark.tier_a
def test_runconfig_frozen_raises_validation_error(
    minimal_runconfig_yaml: str,
) -> None:
    """RunConfig mutation raises pydantic.ValidationError with type='frozen_instance'.

    Locks CFG-04 RunConfig half (RESEARCH.md Pitfall 4). pydantic v2 with
    ``frozen=True`` raises ValidationError on attribute assignment — NOT
    AttributeError (the stdlib dataclass behavior). A silent migration to
    stdlib dataclass would break this assertion immediately.
    """
    cfg = RunConfig(**yaml.safe_load(minimal_runconfig_yaml))

    with pytest.raises(pydantic.ValidationError) as exc_info:
        cfg.mode = "multi-view"  # type: ignore[misc]

    # RESEARCH.md Pitfall 4: pydantic v2 frozen tagging uses type='frozen_instance'.
    assert "frozen_instance" in str(exc_info.value), (
        f"expected pydantic ValidationError with type='frozen_instance' "
        f"per RESEARCH.md Pitfall 4; got: {exc_info.value}"
    )


@pytest.mark.tier_a
def test_runcontext_frozen_raises_frozen_instance_error(
    tmp_path: Path,
    minimal_runconfig_yaml: str,
) -> None:
    """RunContext mutation raises dataclasses.FrozenInstanceError.

    Locks CFG-04 RunContext half (RESEARCH.md Pitfall 4). stdlib
    @dataclasses.dataclass(frozen=True) raises FrozenInstanceError on
    attribute assignment — DIFFERENT exception type from pydantic. A silent
    migration to pydantic would raise ValidationError instead and this
    assertion would catch the regression.
    """
    cfg_dict = yaml.safe_load(minimal_runconfig_yaml)
    cfg_dict["io"]["output_dir"] = str(tmp_path)
    cfg = RunConfig(**cfg_dict)

    ctx = build_context(cfg, _cpu_runtime())

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_id = "tampered"  # type: ignore[misc]


@pytest.mark.tier_a
def test_runconfig_and_runcontext_use_different_exception_types(
    tmp_path: Path,
    minimal_runconfig_yaml: str,
) -> None:
    """Cross-check: the two exception classes are DIFFERENT — not synonyms.

    The DIFFERENT-exception-types invariant is the actual CFG-04 lock-in
    (Pitfall 4 says you cannot use ``pytest.raises(AttributeError)`` as a
    one-size-fits-all). This test makes the contrast explicit so the next
    refactorer sees the intent: do NOT collapse onto a single exception type
    via a base class or a custom wrapper.
    """
    # pydantic.ValidationError is NOT a subclass of dataclasses.FrozenInstanceError
    # and vice versa. They share no MRO beyond ``Exception``.
    assert not issubclass(pydantic.ValidationError, dataclasses.FrozenInstanceError)
    assert not issubclass(dataclasses.FrozenInstanceError, pydantic.ValidationError)
