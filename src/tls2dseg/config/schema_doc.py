"""Schema-documentation generator (DOC-02).

Walks RunConfig + all 13 sub-models via ``model_fields``, reads the
``json_schema_extra["tag"]`` annotation on every field, groups by tag
(primary → tuning → pipings → other), and renders Markdown reference docs.

The discriminated InferenceConfig union (GroundedSAM2Config |
GroundedSAM2HFConfig) is unpacked via ``typing.get_args()`` so fields from
both members appear in the output.
"""

from __future__ import annotations

import typing

from pydantic import BaseModel

from tls2dseg.config.models import (
    D3DExtractionConfig,
    FusionConfig,
    GroundedSAM2Config,
    GroundedSAM2HFConfig,
    InferenceSharedConfig,
    IOConfig,
    LoggingConfig,
    MultiZoomConfig,
    PreprocessingConfig,
    ProjectionConfig,
    PromptConfig,
    RunConfig,
    RuntimeConfig,
    SlicingConfig,
)

__all__ = ["generate_schema_doc"]

TAG_ORDER = ["primary", "tuning", "pipings", "other"]

# Explicit traversal order for top-level RunConfig sub-blocks.
_SUB_MODELS = [
    RunConfig,
    IOConfig,
    RuntimeConfig,
    PromptConfig,
    PreprocessingConfig,
    ProjectionConfig,
    SlicingConfig,
    MultiZoomConfig,
    InferenceSharedConfig,
    GroundedSAM2Config,
    GroundedSAM2HFConfig,
    D3DExtractionConfig,
    FusionConfig,
    LoggingConfig,
]


def _is_model(tp: object) -> bool:
    """Return True if tp is a pydantic BaseModel subclass."""
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _iter_union_members(annotation: object) -> list[type]:
    """Unpack Annotated[A | B, ...] → [A, B]; or [] if not a union."""
    args = typing.get_args(annotation)
    if not args:
        return []
    inner = args[0]
    union_members = typing.get_args(inner)
    return [m for m in union_members if isinstance(m, type)]


def _type_name(annotation: object) -> str:
    """Return a compact human-readable name for a field annotation."""
    if annotation is None:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__  # type: ignore[union-attr]
    return str(annotation).replace("typing.", "").replace("tls2dseg.config.models.", "")


def _default_str(finfo: object) -> str:
    """Return the field default as a string, or '' if required."""
    import pydantic.fields as _pf

    if not isinstance(finfo, _pf.FieldInfo):
        return ""
    if finfo.is_required():
        return "*(required)*"
    default = finfo.default
    if default is None:
        return "null"
    if isinstance(default, bool):
        return str(default).lower()
    return repr(default)


def _collect_fields(
    model: type,
    source_label: str,
    seen: set[tuple[str, str]],
) -> list[tuple[str, str, str, str, str, str]]:
    """Return list of (tag, field_name, type_str, default_str, description, source).

    Recurses into nested BaseModel fields; unpacks the InferenceConfig
    discriminated union via _iter_union_members.
    Skips fields already seen (model_name, field_name) to avoid duplicates
    from inheritance.
    """
    rows: list[tuple[str, str, str, str, str, str]] = []
    for fname, finfo in model.model_fields.items():
        key = (source_label, fname)
        if key in seen:
            continue
        seen.add(key)

        extra = finfo.json_schema_extra
        tag = extra.get("tag", "other") if isinstance(extra, dict) else "other"

        annotation = finfo.annotation

        # Nested BaseModel — recurse but still render the parent field row.
        if _is_model(annotation):
            rows.append(
                (
                    tag,
                    fname,
                    _type_name(annotation),
                    _default_str(finfo),
                    finfo.description or "",
                    source_label,
                )
            )
            continue

        # Discriminated union (InferenceConfig = Annotated[A | B, Field(...)]).
        union_members = _iter_union_members(annotation)
        if union_members:
            for member in union_members:
                if _is_model(member):
                    member_label = member.__name__
                    for sub_fname, sub_finfo in member.model_fields.items():
                        sub_key = (member_label, sub_fname)
                        if sub_key in seen:
                            continue
                        seen.add(sub_key)
                        sub_extra = sub_finfo.json_schema_extra
                        sub_tag = sub_extra.get("tag", "other") if isinstance(sub_extra, dict) else "other"
                        rows.append(
                            (
                                sub_tag,
                                sub_fname,
                                _type_name(sub_finfo.annotation),
                                _default_str(sub_finfo),
                                sub_finfo.description or "",
                                member_label,
                            )
                        )
            continue

        rows.append(
            (
                tag,
                fname,
                _type_name(annotation),
                _default_str(finfo),
                finfo.description or "",
                source_label,
            )
        )

    return rows


def generate_schema_doc() -> str:
    """Return a Markdown config reference grouped by tag (DOC-02).

    Sections are ordered: primary → tuning → pipings → other.
    The discriminated InferenceConfig union is unpacked so both
    GroundedSAM2Config and GroundedSAM2HFConfig fields appear.
    Output begins with a title header and a dataset permanent-link
    placeholder per D-19.
    """
    all_rows: list[tuple[str, str, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for model in _SUB_MODELS:
        label = model.__name__
        all_rows.extend(_collect_fields(model, label, seen))

    # Group by tag.
    by_tag: dict[str, list[tuple[str, str, str, str, str, str]]] = {t: [] for t in TAG_ORDER}
    for row in all_rows:
        tag = row[0]
        if tag not in by_tag:
            by_tag.setdefault(tag, [])
        by_tag[tag].append(row)

    lines: list[str] = [
        "# tls2dseg Config Schema Reference",
        "",
        "Auto-generated by `tls2dseg schema` from the pydantic models.",
        "Fields are grouped by relevance tag: **primary** (user-facing), **tuning**",
        "(result quality), **pipings** (compute/infra), **other** (rarely changed).",
        "",
        "Reference datasets: <PLACEHOLDER_DATASET_URL>",
        "",
    ]

    for tag in TAG_ORDER:
        rows = by_tag.get(tag, [])
        if not rows:
            continue
        lines.append(f"## {tag}")
        lines.append("")
        lines.append("| Field | Type | Default | Description | Model |")
        lines.append("|-------|------|---------|-------------|-------|")
        for _tag, fname, type_str, default, desc, source in rows:
            # Escape pipe characters in values to avoid breaking the table.
            desc_safe = desc.replace("|", "\\|")
            type_safe = type_str.replace("|", "\\|")
            lines.append(f"| `{fname}` | `{type_safe}` | {default} | {desc_safe} | {source} |")
        lines.append("")

    return "\n".join(lines)
