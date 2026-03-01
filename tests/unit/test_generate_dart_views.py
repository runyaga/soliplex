"""Tests for scripts/generate_dart_views.py."""

import importlib
import importlib.util
import json
import pathlib

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "generate_dart_views.py"
)
_spec = importlib.util.spec_from_file_location("generate_dart_views", _SCRIPT)
gdv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gdv)


# ── Name helpers ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "key, expected",
    [
        ("haiku.rag.chat", "haiku_rag_chat"),
        ("filter_documents", "filter_documents"),
        ("monty", "monty"),
        ("my-feature", "my_feature"),
    ],
)
def test_feature_key_to_file_stem(key, expected):
    assert gdv.feature_key_to_file_stem(key) == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("haiku.rag.chat", "HaikuRagChat"),
        ("filter_documents", "FilterDocuments"),
        ("monty", "Monty"),
        ("my-feature", "MyFeature"),
    ],
)
def test_feature_key_to_class_prefix(key, expected):
    assert gdv.feature_key_to_class_prefix(key) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("document_filter", "documentFilter"),
        ("session_id", "sessionId"),
        ("id", "id"),
        ("qa_history", "qaHistory"),
    ],
)
def test_snake_to_lower_camel(name, expected):
    assert gdv.snake_to_lower_camel(name) == expected


# ── Schema helpers ───────────────────────────────────────────


def test_resolve_ref():
    defs = {"Foo": {"type": "object"}}
    result = gdv.resolve_ref("#/$defs/Foo", defs)
    assert result == {"type": "object"}


def test_ref_title():
    assert gdv.ref_title("#/$defs/Citation") == "Citation"


class TestUnwrapNullable:
    def test_non_nullable(self):
        schema = {"type": "string"}
        inner, nullable = gdv.unwrap_nullable(schema)
        assert inner is schema
        assert nullable is False

    def test_nullable_string(self):
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        }
        inner, nullable = gdv.unwrap_nullable(schema)
        assert inner == {"type": "string"}
        assert nullable is True

    def test_multiple_non_null(self):
        schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ]
        }
        inner, nullable = gdv.unwrap_nullable(schema)
        assert inner is schema
        assert nullable is False


# ── Default literals ─────────────────────────────────────────


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        ("hello", "'hello'"),
        ("it's", "'it\\'s'"),
        (42, "42"),
        (3.14, "3.14"),
        ([], "const []"),
        ({}, "const {}"),
    ],
)
def test_dart_default_literal(value, expected):
    assert gdv.dart_default_literal(value) == expected


# ── Getter emission ──────────────────────────────────────────


class TestEmitGetter:
    """Tests for individual getter emission."""

    def _emit(
        self,
        prop_name,
        prop_schema,
        required=None,
        defs=None,
    ):
        lines = []
        nested_queue = []
        seen = set()
        gdv.emit_getter(
            lines,
            prop_name,
            prop_schema,
            required or set(),
            defs or {},
            nested_queue,
            seen,
        )
        return "\n".join(lines), nested_queue

    def test_required_string(self):
        text, _ = self._emit(
            "name",
            {"type": "string"},
            required={"name"},
        )
        assert "String get name =>" in text
        assert "?? ''" in text

    def test_nullable_string(self):
        text, _ = self._emit(
            "title",
            {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
                "default": None,
            },
        )
        assert "String? get title =>" in text

    def test_number_with_default(self):
        text, _ = self._emit(
            "confidence",
            {"type": "number", "default": 0.9},
        )
        assert "double get confidence =>" in text
        assert "as num?)?.toDouble()" in text
        assert "?? 0.9" in text

    def test_number_int_default(self):
        text, _ = self._emit(
            "score",
            {"type": "number", "default": 0},
        )
        assert "?? 0.0" in text

    def test_nullable_number(self):
        text, _ = self._emit(
            "score",
            {
                "anyOf": [
                    {"type": "number"},
                    {"type": "null"},
                ],
                "default": None,
            },
        )
        assert "double? get score =>" in text
        assert "toDouble();" in text

    def test_boolean_required(self):
        text, _ = self._emit(
            "active",
            {"type": "boolean"},
            required={"active"},
        )
        assert "bool get active =>" in text
        assert "?? false" in text

    def test_string_array(self):
        text, _ = self._emit(
            "tags",
            {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
        )
        assert "List<String> get tags =>" in text
        assert "?.cast<String>() ?? const []" in text

    def test_nullable_string_array(self):
        text, _ = self._emit(
            "tags",
            {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    {"type": "null"},
                ],
                "default": None,
            },
        )
        assert "List<String>? get tags =>" in text
        assert "?.cast<String>();" in text

    def test_int_array(self):
        text, _ = self._emit(
            "page_numbers",
            {
                "type": "array",
                "items": {"type": "integer"},
                "default": [],
            },
        )
        assert "List<int> get pageNumbers =>" in text
        assert "?.cast<int>() ?? const []" in text

    def test_number_array_nullable(self):
        text, _ = self._emit(
            "scores",
            {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    {"type": "null"},
                ],
                "default": None,
            },
        )
        assert "List<double>? get scores =>" in text
        assert "(e as num).toDouble()" in text

    def test_number_array_non_nullable(self):
        text, _ = self._emit(
            "scores",
            {
                "type": "array",
                "items": {"type": "number"},
                "default": [],
            },
        )
        assert "List<double> get scores =>" in text
        assert "(e as num).toDouble()" in text
        assert "?? const []" in text

    def test_ref_array(self):
        defs = {
            "Item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
        }
        text, nested = self._emit(
            "items",
            {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
                "default": [],
            },
            defs=defs,
        )
        assert "List<Map<String, dynamic>> get itemsRaw" in text
        assert "int get itemsLength =>" in text
        assert "List<ItemView> get items =>" in text
        assert ".map(ItemView.new).toList()" in text
        assert len(nested) == 1
        assert nested[0][0] == "Item"

    def test_nullable_ref_object(self):
        defs = {
            "Ctx": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "default": "",
                    },
                },
            },
        }
        text, nested = self._emit(
            "context",
            {
                "anyOf": [
                    {"$ref": "#/$defs/Ctx"},
                    {"type": "null"},
                ],
                "default": None,
            },
            defs=defs,
        )
        assert "CtxView? get context {" in text
        assert "if (raw == null) return null;" in text
        assert "return CtxView(" in text
        assert len(nested) == 1

    def test_map_field(self):
        text, _ = self._emit(
            "registry",
            {
                "type": "object",
                "additionalProperties": {
                    "type": "integer",
                },
                "default": {},
            },
        )
        assert "Map<String, int> get registry =>" in text
        assert "?.cast<String, int>()" in text
        assert "?? const {}" in text

    def test_nullable_map_field(self):
        text, _ = self._emit(
            "registry",
            {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": {
                            "type": "integer",
                        },
                    },
                    {"type": "null"},
                ],
                "default": None,
            },
        )
        assert "Map<String, int>? get registry =>" in text
        assert "?.cast<String, int>();" in text

    def test_nested_array(self):
        text, _ = self._emit(
            "history",
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "default": [],
            },
        )
        assert "List<dynamic> get history =>" in text
        assert "int get historyLength =>" in text

    def test_ref_not_scheduled_twice(self):
        defs = {
            "Tag": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
        }
        lines = []
        nested_queue = []
        seen = set()
        # Emit twice with same $ref
        gdv.emit_getter(
            lines,
            "tags_a",
            {
                "type": "array",
                "items": {"$ref": "#/$defs/Tag"},
                "default": [],
            },
            set(),
            defs,
            nested_queue,
            seen,
        )
        gdv.emit_getter(
            lines,
            "tags_b",
            {
                "type": "array",
                "items": {"$ref": "#/$defs/Tag"},
                "default": [],
            },
            set(),
            defs,
            nested_queue,
            seen,
        )
        assert len(nested_queue) == 1

    def test_nullable_ref_array(self):
        defs = {
            "Item": {
                "type": "object",
                "properties": {
                    "val": {"type": "string"},
                },
            },
        }
        text, nested = self._emit(
            "items",
            {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {
                            "$ref": "#/$defs/Item",
                        },
                    },
                    {"type": "null"},
                ],
                "default": None,
            },
            defs=defs,
        )
        assert "List<Map<String, dynamic>>?" not in text
        assert "get itemsRaw =>" in text
        assert "int? get itemsLength =>" in text
        assert "List<ItemView>? get items =>" in text
        assert len(nested) == 1


# ── Full feature generation ──────────────────────────────────


class TestGenerateFeatureDart:
    def test_simple_feature(self):
        schema = {
            "description": "Test feature.",
            "properties": {
                "name": {"type": "string", "default": ""},
            },
            "title": "TestFeature",
            "type": "object",
        }
        dart = gdv.generate_feature_dart("test", schema)
        assert "class TestView {" in dart
        assert "const TestView(this._data);" in dart
        assert "final Map<String, dynamic> _data;" in dart
        assert "String get name =>" in dart
        assert "// GENERATED" in dart
        assert "import 'package:meta/meta.dart';" in dart

    def test_feature_with_nested(self):
        schema = {
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "integer",
                            "default": 0,
                        },
                    },
                    "title": "Inner",
                },
            },
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Inner"},
                    "default": [],
                },
            },
            "title": "Outer",
            "type": "object",
        }
        dart = gdv.generate_feature_dart("outer", schema)
        assert "class OuterView {" in dart
        assert "class InnerView {" in dart
        assert "InnerView.new" in dart

    def test_no_lines_exceed_79_chars(self):
        """Verify all generated Dart lines are ≤79 chars."""
        schema = json.loads(
            (_SCRIPT.parent.parent / "schemas" / "schema.json").read_text()
        )
        for fkey, fschema in schema["properties"].items():
            dart = gdv.generate_feature_dart(fkey, fschema)
            for i, line in enumerate(dart.split("\n"), start=1):
                assert len(line) <= 79, (
                    f"{fkey}:{i}: {len(line)} chars > 79: {line!r}"
                )

    def test_monty_view(self):
        schema = {
            "description": "Monty state.",
            "properties": {
                "ace_skillbook_context": {
                    "default": "",
                    "type": "string",
                },
            },
            "title": "MontyState",
            "type": "object",
        }
        dart = gdv.generate_feature_dart("monty", schema)
        assert "class MontyView {" in dart
        assert "String get aceSkillbookContext =>" in dart

    def test_doc_comment_emitted(self):
        schema = {
            "description": "Line one.\n\nLine three.",
            "properties": {},
            "type": "object",
        }
        dart = gdv.generate_feature_dart("doc", schema)
        assert "/// Line one." in dart
        assert "///" in dart
        assert "/// Line three." in dart


# ── CLI main ─────────────────────────────────────────────────


class TestMain:
    def test_missing_schema(self, tmp_path):
        rc = gdv.main(
            [
                "--schema",
                str(tmp_path / "missing.json"),
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1

    def test_generate_to_dir(self, tmp_path):
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "my_feature": {
                    "properties": {
                        "foo": {
                            "type": "string",
                            "default": "",
                        },
                    },
                    "title": "MyFeature",
                    "type": "object",
                },
            },
        }
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(schema))
        out_dir = tmp_path / "dart"

        rc = gdv.main(
            [
                "--schema",
                str(schema_file),
                "--out",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert (out_dir / "my_feature_view.dart").exists()

        content = (out_dir / "my_feature_view.dart").read_text()
        assert "class MyFeatureView {" in content

    def test_creates_output_dir(self, tmp_path):
        schema = {
            "properties": {
                "x": {
                    "properties": {},
                    "type": "object",
                },
            },
        }
        schema_file = tmp_path / "s.json"
        schema_file.write_text(json.dumps(schema))
        out_dir = tmp_path / "nested" / "dir"

        rc = gdv.main(
            [
                "--schema",
                str(schema_file),
                "--out",
                str(out_dir),
            ]
        )
        assert rc == 0
        assert out_dir.is_dir()


# ── Integration: real schema.json ────────────────────────────


class TestRealSchema:
    """Run the generator against the actual schema.json."""

    @pytest.fixture
    def real_schema(self):
        path = _SCRIPT.parent.parent / "schemas" / "schema.json"
        with open(path) as f:
            return json.load(f)

    def test_haiku_rag_chat_view(self, real_schema):
        schema = real_schema["properties"]["haiku.rag.chat"]
        dart = gdv.generate_feature_dart(
            "haiku.rag.chat",
            schema,
        )
        assert "class HaikuRagChatView {" in dart
        assert "class CitationView {" in dart
        assert "class QAHistoryEntryView {" in dart
        assert "class SessionContextView {" in dart
        # Key getters
        assert "get initialContext" in dart
        assert "get citations" in dart
        assert "get citationsRaw" in dart
        assert "get citationsLength" in dart
        assert "get qaHistory" in dart
        assert "get documentFilter" in dart
        assert "get citationRegistry" in dart
        assert "get citationsHistory" in dart

    def test_filter_documents_view(self, real_schema):
        schema = real_schema["properties"]["filter_documents"]
        dart = gdv.generate_feature_dart(
            "filter_documents",
            schema,
        )
        assert "class FilterDocumentsView {" in dart
        assert "get documentIds" in dart
        assert "List<String>?" in dart

    def test_ask_history_view(self, real_schema):
        schema = real_schema["properties"]["ask_history"]
        dart = gdv.generate_feature_dart(
            "ask_history",
            schema,
        )
        assert "class AskHistoryView {" in dart
        assert "class QuestionResponseCitationsView {" in dart
        assert "class CitationView {" in dart
        assert "get questions" in dart
        assert "get questionsRaw" in dart
        assert "get questionsLength" in dart

    def test_monty_view(self, real_schema):
        schema = real_schema["properties"]["monty"]
        dart = gdv.generate_feature_dart(
            "monty",
            schema,
        )
        assert "class MontyView {" in dart
        assert "get aceSkillbookContext" in dart

    def test_all_features_present(self, real_schema):
        features = real_schema["properties"]
        assert "haiku.rag.chat" in features
        assert "filter_documents" in features
        assert "ask_history" in features
        assert "monty" in features
