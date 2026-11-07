"""Exercise filesystem Monty tools with real sandbox execution."""

import dataclasses

import pydantic_core
import pytest
from pydantic_ai import Agent
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from soliplex.capabilities import filesystem
from soliplex.capabilities import monty


def _skill(tmp_path, source, *, allowed=(), name="skill", deferred=False):
    path = tmp_path / name
    (path / "monty").mkdir(parents=True)
    (path / "monty" / "task.mpy").write_text(source, encoding="utf-8")
    return filesystem.FilesystemCapability(
        id=name,
        instructions="Use the bundled script.",
        path=path,
        allowed_tools=allowed,
        defer_loading=deferred,
    )


async def _run(capability, *, arguments=None, toolsets=(), capabilities=()):
    calls = iter(
        [ToolCallPart("load_capability", {"id": capability.id})]
        * capability.defer_loading
        + [ToolCallPart(f"{capability.id}_task", arguments or {})]
    )

    def respond(messages, info):
        call = next(calls, None)
        return ModelResponse(parts=[call or TextPart("done")])

    result = await Agent(
        FunctionModel(respond),
        capabilities=[capability, *capabilities],
        toolsets=toolsets,
    ).run("Run the script")
    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
        and part.tool_name == f"{capability.id}_task"
    ]
    (output,) = returns
    return output


@pytest.mark.anyio
async def test_schema_and_validated_arguments(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(text: str, count: int, ratio: float, flag: bool, "
        "items: list[dict[str, list[int | None]]], *, "
        "optional: str | None = None, size: int = 2):\n"
        '    """Summarize the arguments."""\n'
        "    print(text)\n"
        "    return [count, ratio, flag, items, optional, size]\n",
    )
    toolset = cap.get_toolset()
    tools = await toolset.get_tools(
        RunContext(deps=None, model=TestModel(), usage=RunUsage())
    )
    assert list(tools) == ["skill_task"]
    tool = tools["skill_task"]
    assert tool.tool_def.description == "Summarize the arguments."
    schema = tool.tool_def.parameters_json_schema
    assert schema["required"] == ["text", "count", "ratio", "flag", "items"]
    assert schema["properties"]["optional"]["default"] is None
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["additionalProperties"] is False
    with pytest.raises(pydantic_core.ValidationError):
        tool.args_validator.validate_python({"text": 123})
    output = await _run(
        cap,
        arguments={
            "text": "hello",
            "count": 3,
            "ratio": 1.5,
            "flag": True,
            "items": [{"a": [1, None]}],
        },
    )
    assert output == 'hello\n[3, 1.5, true, [{"a": [1, null]}], null, 2]'


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.anyio
async def test_host_function(tmp_path, deferred):
    cap = _skill(
        tmp_path,
        "def main(value: int):\n    return double(value=value)\n",
        allowed=("double", "missing"),
        deferred=deferred,
    )
    tools = FunctionToolset()

    @tools.tool_plain
    def double(value: int) -> int:
        return value * 2

    assert await _run(cap, arguments={"value": 4}, toolsets=[tools]) == "8"


@pytest.mark.parametrize(
    "source, message",
    [
        ("def main(:", "invalid syntax"),
        ("def main(x: set[int]): pass", "Unsupported parameter annotation"),
        ("def main(x): pass", "Unsupported parameter annotation"),
        ("def main(x: int | None): pass", "require a default"),
        ("def other(): pass", "exactly one main"),
        ("def main(): pass\ndef main(): pass", "exactly one main"),
        ("async def main(): pass", "exactly one main"),
        ("def main(*args: int): pass", "Unsupported main signature"),
        ("def main(**kw: int): pass", "Unsupported main signature"),
        ("@decorate\ndef main(): pass", "Unsupported main signature"),
        ("def main(x: int = object()): pass", "malformed node"),
        ("def main(x: int = 'no'): pass", "validation error"),
    ],
)
def test_bad_scripts_are_logged_and_skipped(tmp_path, caplog, source, message):
    cap = _skill(tmp_path, source)
    assert cap.get_toolset() is None
    assert str(cap.path / "monty" / "task.mpy") in caplog.text
    assert message in caplog.text


@pytest.mark.anyio
async def test_only_direct_regular_named_scripts_are_discovered(tmp_path):
    cap = _skill(tmp_path, "def main(): return 1")
    root = cap.path / "monty"
    (root / "directory.mpy").mkdir()
    (root / "directory.mpy" / "nested.mpy").write_text("def main(): pass")
    (root / "Invalid.mpy").write_text("def main(): pass")
    (root / "other.py").write_text("def main(): pass")
    tools = await cap.get_toolset().get_tools(
        RunContext(deps=None, model=TestModel(), usage=RunUsage())
    )
    assert list(tools) == ["skill_task"]
    assert monty.MontyToolset(root, ()).id is None


@pytest.mark.parametrize(
    "source, expected",
    [
        ("def main(): return secret()", "secret"),
        ("def main():\n    x: int = 'bad'\n    return x", "type"),
        ("def main(): return 1 / 0", "ZeroDivisionError"),
        ("def main():\n    while True:\n        pass", "time"),
    ],
)
@pytest.mark.anyio
async def test_errors_return_text(tmp_path, monkeypatch, source, expected):
    cap = _skill(tmp_path, source, allowed=("missing",))
    monkeypatch.setitem(monty._LIMITS, "max_feed_duration_secs", 0.01)
    tools = FunctionToolset()

    @tools.tool_plain
    def secret() -> str:
        return "private"

    assert secret() == "private"
    output = await _run(cap, toolsets=[tools])
    assert output.startswith("SCRIPT ERROR:")
    assert expected in output


@pytest.mark.anyio
async def test_file_edits_are_read_each_call(tmp_path):
    cap = _skill(tmp_path, "def main(): return 1")
    path = cap.path / "monty" / "task.mpy"
    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_text("def main(): return 2", encoding="utf-8")
        if calls <= 2:
            return ModelResponse(parts=[ToolCallPart("skill_task", {})])
        return ModelResponse(parts=[TextPart("done")])

    result = await Agent(FunctionModel(respond), capabilities=[cap]).run("run")
    assert [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ] == ["1", "2"]


@pytest.mark.parametrize(
    "source",
    [
        "def main(): return fail()",
        "def main():\n"
        "    try:\n        fail()\n"
        "    except Exception:\n        return 'caught'\n",
    ],
)
@pytest.mark.anyio
async def test_host_failure_even_when_caught_returns_error(tmp_path, source):
    cap = _skill(tmp_path, source, allowed=("fail",))
    tools = FunctionToolset()

    @tools.tool_plain
    def fail() -> str:
        message = "host broke"
        raise ValueError(message)

    output = await _run(cap, toolsets=[tools])
    assert output.startswith("SCRIPT ERROR:")
    assert "host broke" in output


@pytest.mark.anyio
async def test_host_call_limit(tmp_path):
    cap = _skill(
        tmp_path,
        "def main():\n    for i in range(101):\n        ping()\n",
        allowed=("ping",),
    )
    tools = FunctionToolset()
    calls = []

    @tools.tool_plain
    def ping() -> int:
        calls.append(1)
        return 1

    output = await _run(cap, toolsets=[tools])
    assert output.startswith("SCRIPT ERROR:")
    assert "Host call limit of 100 exceeded" in output
    assert len(calls) == 100


@pytest.mark.anyio
async def test_per_run_capability_state_is_shared(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(): return search(query='from script')",
        allowed=("search",),
    )
    runs = []

    @dataclasses.dataclass
    class Search(AbstractCapability):
        results: list = dataclasses.field(default_factory=list)

        async def for_run(self, ctx):
            instance = dataclasses.replace(self, results=[])
            runs.append(instance)
            return instance

        def get_toolset(self):
            tools = FunctionToolset()

            @tools.tool_plain
            def search(query: str) -> list[str]:
                self.results.append(query)
                return self.results

            return tools

        async def after_tool_execute(self, ctx, *, result, **kwargs):
            self.results.append(ctx.tool_name)
            return result

    search = Search()
    assert (
        await _run(cap, capabilities=[search]) == '["from script", "search"]'
    )
    assert search.results == []
    assert runs[0].results == ["from script", "search", "skill_task"]


@pytest.mark.anyio
async def test_native_deferred_visibility(tmp_path):
    cap = _skill(tmp_path, "def main(): return 'loaded'", deferred=True)
    steps = []

    def respond(messages, info):
        steps.append({tool.name for tool in info.function_tools})
        if len(steps) == 1:
            assert "skill_task" not in steps[-1]
            return ModelResponse(
                parts=[ToolCallPart("load_capability", {"id": "skill"})]
            )
        if len(steps) == 2:
            assert "skill_task" in steps[-1]
            return ModelResponse(parts=[ToolCallPart("skill_task", {})])
        assert any(
            isinstance(part, ToolReturnPart) and part.content == '"loaded"'
            for message in messages
            for part in message.parts
        )
        return ModelResponse(parts=[TextPart("done")])

    await Agent(FunctionModel(respond), capabilities=[cap]).run("load")
    assert len(steps) == 3


@pytest.mark.parametrize(
    "declaration, expected",
    [
        ("", ()),
        (
            "allowed-tools: 'search  weather\tread'\n",
            ("search", "weather", "read"),
        ),
        ("allowed-tools: search weather\n", ("search", "weather")),
        ("allowed-tools: [search, weather]\n", ("search", "weather")),
    ],
)
def test_frontmatter_allowed_tools(tmp_path, declaration, expected):
    path = tmp_path / "skill"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: skill\ndescription: Test\n"
        + declaration
        + "---\nUse it.\n",
        encoding="utf-8",
    )
    capabilities, errors = filesystem.discover_filesystem_capabilities([path])
    assert errors == []
    assert capabilities[0].allowed_tools == expected


def test_frontmatter_rejects_non_string_allowed_tools(tmp_path):
    path = tmp_path / "skill"
    path.mkdir()
    (path / "SKILL.md").write_text(
        "---\nname: skill\ndescription: Test\n"
        "allowed-tools: [12]\n---\nUse it.",
        encoding="utf-8",
    )
    capabilities, errors = filesystem.discover_filesystem_capabilities([path])
    assert capabilities == []
    assert "allowed-tools" in str(errors[0])


def test_discovery_logs_unreadable_script(tmp_path, monkeypatch, caplog):
    cap = _skill(tmp_path, "def main(): pass")

    def unreadable(*args, **kwargs):
        raise OSError("unreadable")

    monkeypatch.setattr(type(cap.path), "read_text", unreadable)
    assert cap.get_toolset() is None
    assert str(cap.path / "monty" / "task.mpy") in caplog.text
    assert "unreadable" in caplog.text


@pytest.mark.parametrize(
    "replacement, expected",
    [("def main(:", "syntax"), (None, "task.mpy")],
)
@pytest.mark.anyio
async def test_changed_or_removed_script_returns_error(
    tmp_path, replacement, expected
):
    cap = _skill(tmp_path, "def main(): return 1")

    def respond(messages, info):
        if len(messages) == 1:
            path = cap.path / "monty" / "task.mpy"
            if replacement is None:
                path.unlink()
            else:
                path.write_text(replacement, encoding="utf-8")
            return ModelResponse(parts=[ToolCallPart("skill_task", {})])
        return ModelResponse(parts=[TextPart("done")])

    result = await Agent(FunctionModel(respond), capabilities=[cap]).run("run")
    output = result.all_messages()[2].parts[0].content
    assert output.startswith("SCRIPT ERROR:")
    assert expected in output


@pytest.mark.anyio
async def test_host_call_type_checks_against_stub(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(): return len(text=12)",
        allowed=("len",),
    )
    tools = FunctionToolset()

    @tools.tool_plain(name="len")
    def length(text: str) -> int:
        return len(text)

    assert length("abc") == 3
    output = await _run(cap, toolsets=[tools])
    assert output.startswith("SCRIPT ERROR:")
    assert "int" in output
    assert "str" in output


@pytest.mark.anyio
async def test_other_skill_script_is_a_host_tool(tmp_path):
    first = _skill(
        tmp_path,
        "def main(): return second_task()",
        allowed=("second_task",),
        name="first",
    )
    second = _skill(tmp_path, "def main(): return 42", name="second")
    assert await _run(first, capabilities=[second]) == '"42"'


@pytest.mark.anyio
async def test_missing_and_hidden_hosts_are_undefined(tmp_path):
    first = _skill(
        tmp_path,
        "def main(): return hidden_task()",
        allowed=("hidden_task", "missing"),
        name="first",
    )
    hidden = _skill(
        tmp_path, "def main(): return 42", name="hidden", deferred=True
    )
    output = await _run(first, capabilities=[hidden])
    assert output.startswith("SCRIPT ERROR:")
    assert "hidden_task" in output


@pytest.mark.anyio
async def test_python_script_tool_rejects_monty_directory(tmp_path):
    cap = _skill(tmp_path, "def main(): return 1")
    with pytest.raises(filesystem.MissingCapabilityFile):
        await cap.run_script("monty/task.mpy")


@pytest.mark.anyio
async def test_memory_limit_returns_error(tmp_path, monkeypatch):
    cap = _skill(tmp_path, "def main(): return [0] * 1000000")
    monkeypatch.setitem(monty._LIMITS, "max_memory", 1024 * 1024)
    output = await _run(cap)
    assert output.startswith("SCRIPT ERROR:")
    assert "memory" in output.lower()


@pytest.mark.anyio
async def test_positional_only_parameters(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(a: int, b: str = 'b', /, *, c: bool = True):\n"
        "    return [a, b, c]",
    )
    assert await _run(cap, arguments={"a": 3}) == '[3, "b", true]'


@pytest.mark.anyio
async def test_host_names_match_the_models_prefixed_names(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(): return remote_client_search(query='test')",
        allowed=("remote-client_search",),
    )
    tools = FunctionToolset()

    @tools.tool_plain
    def search(query: str) -> dict[str, str]:
        return {"found": query}

    output = await _run(cap, toolsets=[tools.prefixed("remote-client")])
    assert output == '{"found": "test"}'


@pytest.mark.parametrize(
    "source",
    [
        "def main(): return second_task()",
        "def main():\n"
        "    try:\n        second_task()\n"
        "    except Exception:\n        return 'caught'\n",
    ],
)
@pytest.mark.anyio
async def test_nested_script_failure_returns_callers_error(tmp_path, source):
    first = _skill(tmp_path, source, allowed=("second_task",), name="first")
    second = _skill(tmp_path, "def main(): return 1 / 0", name="second")
    output = await _run(first, capabilities=[second])
    assert output.startswith("SCRIPT ERROR:")
    assert "SCRIPT ERROR:" in output.removeprefix("SCRIPT ERROR:")
    assert "ZeroDivisionError" in output
    assert "caught" not in output
    direct = await _run(second)
    assert direct.startswith("SCRIPT ERROR:")
    assert "ZeroDivisionError" in direct


@pytest.mark.anyio
async def test_regular_host_error_text_remains_a_value(tmp_path):
    cap = _skill(
        tmp_path,
        "def main(): return host()",
        allowed=("host",),
    )
    tools = FunctionToolset()

    @tools.tool_plain
    def host() -> str:
        return "SCRIPT ERROR: ordinary host text"

    assert await _run(cap, toolsets=[tools]) == (
        '"SCRIPT ERROR: ordinary host text"'
    )
