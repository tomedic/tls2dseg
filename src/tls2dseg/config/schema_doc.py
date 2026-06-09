"""Schema-documentation hook (Phase 8 DOC-02 target).

Phase 3 ships the entry-point signature only. Phase 8 DOC-02 implements:
walk every field on every model in :mod:`tls2dseg.config.models`, read the
``json_schema_extra["tag"]`` annotation, group by tag (primary → tuning →
pipings → other per D-A1-04), and render Markdown reference docs.

The ``tag`` annotations are mandatory in Phase 3 specifically so DOC-02 has
something to group by — see ``models.py`` for the 4-tag vocabulary contract.
"""

from __future__ import annotations

__all__ = ["generate_schema_doc"]


def generate_schema_doc() -> str:
    """Stub — Phase 8 DOC-02 implements.

    Will return a Markdown reference grouped by tag, walking
    :class:`tls2dseg.config.models.RunConfig` and every nested sub-model.
    """
    raise NotImplementedError(
        "schema_doc.generate_schema_doc is a Phase 3 stub. Phase 8 DOC-02 implements the actual schema-doc generator."
    )
