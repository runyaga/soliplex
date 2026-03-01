#!/usr/bin/env python3
"""Generate typed Dart StateView wrappers from JSON Schema.

Reads the combined AG-UI feature schema (``schemas/schema.json``)
and emits one Dart file per feature into the output directory.

Usage::

    python scripts/generate_dart_views.py
    python scripts/generate_dart_views.py --schema schemas/schema.json \\
        --out schemas/dart
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "schema.json"
DEFAULT_OUTPUT = REPO_ROOT / "schemas" / "dart"

# JSON Schema type → Dart scalar type
SCALAR_MAP: dict[str, str] = {
    "string": "String",
    "integer": "int",
    "number": "double",
    "boolean": "bool",
}

# Fallback Dart literals when no default is given
SCALAR_DEFAULTS: dict[str, str] = {
    "String": "''",
    "int": "0",
    "double": "0.0",
    "bool": "false",
}


# ── Name helpers ─────────────────────────────────────────────


def feature_key_to_file_stem(key: str) -> str:
    """``'haiku.rag.chat'`` → ``'haiku_rag_chat'``."""
    return re.sub(r"[.\-]", "_", key)


def feature_key_to_class_prefix(key: str) -> str:
    """``'haiku.rag.chat'`` → ``'HaikuRagChat'``."""
    parts = re.split(r"[._\-]", key)
    return "".join(p.capitalize() for p in parts if p)


def snake_to_lower_camel(name: str) -> str:
    """``'document_filter'`` → ``'documentFilter'``."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# ── Schema helpers ───────────────────────────────────────────


def resolve_ref(ref_str: str, defs: dict) -> dict:
    """Resolve ``'#/$defs/Foo'`` to the definition dict."""
    name = ref_str.rsplit("/", 1)[-1]
    return defs[name]


def ref_title(ref_str: str) -> str:
    """``'#/$defs/Citation'`` → ``'Citation'``."""
    return ref_str.rsplit("/", 1)[-1]


def unwrap_nullable(
    schema: dict,
) -> tuple[dict, bool]:
    """If *schema* is ``anyOf[T, null]`` return ``(T, True)``.

    Otherwise return ``(schema, False)``.
    """
    any_of = schema.get("anyOf")
    if not any_of:
        return schema, False
    non_null = [s for s in any_of if s.get("type") != "null"]
    has_null = any(s.get("type") == "null" for s in any_of)
    if has_null and len(non_null) == 1:
        return non_null[0], True
    return schema, False


def dart_default_literal(value: object) -> str:
    """Turn a JSON-Schema default into a Dart literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "const []"
    if isinstance(value, dict):
        return "const {}"
    return repr(value)  # pragma: no cover


# ── Dart emitter ─────────────────────────────────────────────


def _emit_doc_comment(
    lines: list[str],
    description: str,
) -> None:
    for raw_line in description.split("\n"):
        stripped = raw_line.strip()
        if stripped:
            lines.append(f"/// {stripped}")
        else:
            lines.append("///")


def _schedule_nested(
    title: str,
    schema: dict,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    if title not in seen:
        seen.add(title)
        nested_queue.append((title, schema))


# ── Getter emitters ──────────────────────────────────────────


def _emit_scalar_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    schema_type: str,
    nullable: bool,
    default: object,
) -> None:
    dart_type = SCALAR_MAP.get(schema_type, "dynamic")
    is_number = schema_type == "number"

    if nullable:
        if is_number:
            lines.append(f"  double? get {dart_name} =>")
            lines.append(f"      (_data['{key}'] as num?)?.toDouble();")
        else:
            lines.append(f"  {dart_type}? get {dart_name} =>")
            lines.append(f"      _data['{key}'] as {dart_type}?;")
        return

    if default is not None:
        if is_number and isinstance(default, int):
            fallback = f"{default}.0"
        else:
            fallback = dart_default_literal(default)
    else:
        fallback = SCALAR_DEFAULTS.get(dart_type, "null")

    if is_number:
        lines.append(f"  double get {dart_name} =>")
        lines.append(
            f"      (_data['{key}'] as num?)?.toDouble() ?? {fallback};"
        )
    else:
        lines.append(f"  {dart_type} get {dart_name} =>")
        lines.append(f"      _data['{key}'] as {dart_type}? ?? {fallback};")


def _emit_ref_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    inner: dict,
    defs: dict,
    nullable: bool,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    title = ref_title(inner["$ref"])
    resolved = resolve_ref(inner["$ref"], defs)
    _schedule_nested(title, resolved, nested_queue, seen)
    view_cls = f"{title}View"

    if nullable:
        lines.append(f"  {view_cls}? get {dart_name} {{")
        lines.append(f"    final raw = _data['{key}'];")
        lines.append("    if (raw == null) return null;")
        lines.append(f"    return {view_cls}(")
        lines.append("      raw as Map<String, dynamic>,")
        lines.append("    );")
        lines.append("  }")
    else:
        lines.append(f"  {view_cls} get {dart_name} =>")
        lines.append(f"      {view_cls}(")
        lines.append(f"        _data['{key}'] as Map<String, dynamic>")
        lines.append("            ?? const {},")
        lines.append("      );")


def _emit_map_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    inner: dict,
    nullable: bool,
    default: object,
) -> None:
    val_schema = inner["additionalProperties"]
    val_type = SCALAR_MAP.get(val_schema.get("type", ""), "dynamic")
    dart_type = f"Map<String, {val_type}>"

    cast_expr = f"?.cast<String, {val_type}>()"

    if nullable:
        lines.append(f"  {dart_type}? get {dart_name} =>")
        lines.append(f"      (_data['{key}'] as Map<String, dynamic>?)")
        lines.append(f"          {cast_expr};")
    else:
        fallback = dart_default_literal(default if default is not None else {})
        lines.append(f"  {dart_type} get {dart_name} =>")
        lines.append(f"      (_data['{key}'] as Map<String, dynamic>?)")
        lines.append(f"          {cast_expr} ?? {fallback};")


def _emit_array_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    items: dict,
    defs: dict,
    nullable: bool,
    default: object,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    # ── Array of $ref objects ────────────────────────
    if "$ref" in items:
        _emit_object_array_getter(
            lines,
            dart_name,
            key,
            items,
            defs,
            nullable,
            nested_queue,
            seen,
        )
        return

    # ── Nested array (array of arrays) ──────────────
    if items.get("type") == "array":
        _emit_nested_array_getter(
            lines,
            dart_name,
            key,
            nullable,
        )
        return

    # ── Array of scalars ────────────────────────────
    _emit_scalar_array_getter(
        lines,
        dart_name,
        key,
        items,
        nullable,
    )


def _emit_object_array_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    items: dict,
    defs: dict,
    nullable: bool,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    title = ref_title(items["$ref"])
    resolved = resolve_ref(items["$ref"], defs)
    _schedule_nested(title, resolved, nested_queue, seen)
    view_cls = f"{title}View"
    raw_name = f"{dart_name}Raw"

    # Raw getter
    cast = "?.cast<Map<String, dynamic>>()"
    lines.append(f"  List<Map<String, dynamic>> get {raw_name} =>")
    lines.append(f"      (_data['{key}'] as List<dynamic>?)")
    if nullable:
        lines.append(f"          {cast};")
    else:
        lines.append(f"          {cast} ??")
        lines.append("          const [];")

    # Length getter
    lines.append("")
    if nullable:
        lines.append(f"  int? get {dart_name}Length =>")
    else:
        lines.append(f"  int get {dart_name}Length =>")
    length_line = f"      (_data['{key}'] as List<dynamic>?)?.length"
    if nullable:
        lines.append(f"{length_line};")
    else:
        lines.append(f"{length_line} ?? 0;")

    # Typed View list getter
    lines.append("")
    list_type = f"List<{view_cls}>"
    map_expr = f".map({view_cls}.new).toList()"
    if nullable:
        lines.append(f"  {list_type}? get {dart_name} =>")
        lines.append(f"      {raw_name}?{map_expr};")
    else:
        lines.append(f"  {list_type} get {dart_name} =>")
        lines.append(f"      {raw_name}{map_expr};")


def _emit_nested_array_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    nullable: bool,
) -> None:
    lines.append(f"  List<dynamic> get {dart_name} =>")
    lines.append(f"      (_data['{key}'] as List<dynamic>?) ?? const [];")
    lines.append("")
    lines.append(f"  int get {dart_name}Length =>")
    lines.append(f"      (_data['{key}'] as List<dynamic>?)?.length ?? 0;")


def _emit_scalar_array_getter(
    lines: list[str],
    dart_name: str,
    key: str,
    items: dict,
    nullable: bool,
) -> None:
    item_type = items.get("type", "dynamic")
    dart_item = SCALAR_MAP.get(item_type, "dynamic")
    is_number = item_type == "number"

    if nullable:
        lines.append(f"  List<{dart_item}>? get {dart_name} =>")
        if is_number:
            lines.append(f"      (_data['{key}'] as List<dynamic>?)")
            lines.append("          ?.map((e) => (e as num).toDouble())")
            lines.append("          .toList();")
        else:
            lines.append(f"      (_data['{key}'] as List<dynamic>?)")
            lines.append(f"          ?.cast<{dart_item}>();")
    else:
        lines.append(f"  List<{dart_item}> get {dart_name} =>")
        if is_number:
            lines.append(f"      (_data['{key}'] as List<dynamic>?)")
            lines.append("          ?.map((e) => (e as num).toDouble())")
            lines.append("          .toList() ?? const [];")
        else:
            lines.append(f"      (_data['{key}'] as List<dynamic>?)")
            lines.append(f"          ?.cast<{dart_item}>() ?? const [];")


# ── Main getter dispatcher ──────────────────────────────────


def emit_getter(
    lines: list[str],
    prop_name: str,
    raw_schema: dict,
    required: set[str],
    defs: dict,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    """Emit Dart getter(s) for a single property."""
    dart_name = snake_to_lower_camel(prop_name)
    key = prop_name
    default = raw_schema.get("default")

    inner, nullable = unwrap_nullable(raw_schema)

    # Propagate default from outer if inner lacks one
    if default is None and "default" in inner:
        default = inner.get("default")

    # ── $ref (nested object) ────────────────────────
    if "$ref" in inner:
        _emit_ref_getter(
            lines,
            dart_name,
            key,
            inner,
            defs,
            nullable,
            nested_queue,
            seen,
        )
        return

    schema_type = inner.get("type")

    # ── object with additionalProperties (Map) ──────
    if schema_type == "object" and "additionalProperties" in inner:
        _emit_map_getter(
            lines,
            dart_name,
            key,
            inner,
            nullable,
            default,
        )
        return

    # ── array ────────────────────────────────────────
    if schema_type == "array":
        items = inner.get("items", {})
        _emit_array_getter(
            lines,
            dart_name,
            key,
            items,
            defs,
            nullable,
            default,
            nested_queue,
            seen,
        )
        return

    # ── scalar ───────────────────────────────────────
    _emit_scalar_getter(
        lines,
        dart_name,
        key,
        schema_type or "dynamic",
        nullable,
        default,
    )


# ── View class emitter ───────────────────────────────────────


def emit_view_class(
    lines: list[str],
    class_name: str,
    schema: dict,
    defs: dict,
    nested_queue: list[tuple[str, dict]],
    seen: set[str],
) -> None:
    """Emit one ``@immutable class <Name>View``."""
    description = schema.get("description", "")
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    if description:
        _emit_doc_comment(lines, description)
    lines.append("@immutable")
    lines.append(f"class {class_name} {{")
    lines.append(f"  const {class_name}(this._data);")
    lines.append("  final Map<String, dynamic> _data;")

    for prop_name, prop_schema in properties.items():
        lines.append("")
        emit_getter(
            lines,
            prop_name,
            prop_schema,
            required,
            defs,
            nested_queue,
            seen,
        )

    lines.append("}")


# ── Per-feature orchestrator ─────────────────────────────────


def generate_feature_dart(
    feature_key: str,
    feature_schema: dict,
) -> str:
    """Return complete Dart source for one feature."""
    lines: list[str] = []
    defs = feature_schema.get("$defs", {})
    seen: set[str] = set()
    nested_queue: list[tuple[str, dict]] = []

    # File header
    lines.append(
        "// GENERATED by scripts/generate_dart_views.py \u2014 DO NOT EDIT."
    )
    lines.append("import 'package:meta/meta.dart';")
    lines.append("")

    # Root view class
    class_name = feature_key_to_class_prefix(feature_key) + "View"
    emit_view_class(
        lines,
        class_name,
        feature_schema,
        defs,
        nested_queue,
        seen,
    )

    # Drain the nested queue (BFS)
    while nested_queue:
        title, nested_schema = nested_queue.pop(0)
        lines.append("")
        emit_view_class(
            lines,
            f"{title}View",
            nested_schema,
            defs,
            nested_queue,
            seen,
        )

    return "\n".join(lines) + "\n"


# ── CLI entry point ──────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        default=DEFAULT_SCHEMA,
        help="Path to combined schema.json",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for .dart files",
    )
    args = parser.parse_args(argv)

    schema_path: pathlib.Path = args.schema
    out_dir: pathlib.Path = args.out

    if not schema_path.exists():
        print(
            f"Error: schema not found: {schema_path}",
            file=sys.stderr,
        )
        return 1

    with open(schema_path) as f:
        combined = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    features = combined.get("properties", {})
    for fkey, fschema in features.items():
        stem = feature_key_to_file_stem(fkey)
        dart = generate_feature_dart(fkey, fschema)
        out_path = out_dir / f"{stem}_view.dart"
        out_path.write_text(dart, encoding="utf-8")
        print(f"  {out_path.name}")

    count = len(features)
    print(f"\nGenerated {count} view(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
