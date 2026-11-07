"""Expose bundled Monty programs and dispatch their permitted host tools."""

import ast
import contextvars
import dataclasses
import functools
import json
import logging
import pathlib
import re
import typing

import pydantic
import pydantic_core
from pydantic_ai import RunContext
from pydantic_ai import ToolDefinition
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets import WrapperToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai_harness._code_tools import partition_callable_tools
from pydantic_ai_harness._code_tools import run_nested_call
from pydantic_ai_harness._monty_exec import MontyExecutor
from pydantic_ai_harness._monty_exec import MontyRunState
from pydantic_ai_harness._monty_exec import PrintCapture
from pydantic_ai_harness.code_mode._toolset import CodeModeToolset
from pydantic_monty import MontyError

_LOGGER = logging.getLogger(__name__)
_LIMITS = {"max_feed_duration_secs": 30, "max_memory": 256 * 1024 * 1024}
_MAX_HOST_CALLS = 100
_HOST_TOOLSET = contextvars.ContextVar[AbstractToolset]("monty_host_toolset")
_SCALARS = {"str": str, "int": int, "float": float, "bool": bool}


def _annotation(node: ast.expr) -> typing.Any:
    """Resolve the supported annotation grammar without evaluating source."""
    match node:
        case ast.Name(id=name) if name in _SCALARS:
            return _SCALARS[name]
        case ast.Subscript(value=ast.Name(id="list"), slice=item):
            return list[_annotation(item)]
        case ast.Subscript(
            value=ast.Name(id="dict"),
            slice=ast.Tuple(elts=[ast.Name(id="str"), item]),
        ):
            return dict[str, _annotation(item)]
        case ast.BinOp(
            left=item, op=ast.BitOr(), right=ast.Constant(value=None)
        ):
            return _annotation(item) | None
        case _:
            message = "Unsupported parameter annotation"
            raise ValueError(message)


def _signature(source: str) -> tuple[str, dict, typing.Any, tuple[str, ...]]:
    """Build a validated argument schema from the main function."""
    mains = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if len(mains) != 1:
        message = "Define exactly one main function"
        raise ValueError(message)
    main = mains[0]
    args = main.args
    if args.vararg or args.kwarg or main.decorator_list:
        message = "Unsupported main signature"
        raise ValueError(message)
    positional = args.posonlyargs + args.args
    parameters = positional + args.kwonlyargs
    defaults = (
        [None] * (len(positional) - len(args.defaults))
        + args.defaults
        + args.kw_defaults
    )
    fields = {}
    for parameter, default in zip(parameters, defaults, strict=True):
        annotation = _annotation(parameter.annotation)
        schema = pydantic.TypeAdapter(annotation).core_schema
        if type(None) in typing.get_args(annotation) and default is None:
            message = "Nullable parameters require a default"
            raise ValueError(message)
        if default is not None:
            value = ast.literal_eval(default)
            value = pydantic.TypeAdapter(annotation).validate_python(value)
            schema = pydantic_core.core_schema.with_default_schema(
                schema, default=value
            )
        fields[parameter.arg] = pydantic_core.core_schema.typed_dict_field(
            schema, required=default is None
        )
    schema = pydantic_core.core_schema.typed_dict_schema(
        fields, extra_behavior="forbid"
    )
    return (
        ast.get_docstring(main) or "",
        pydantic.json_schema.GenerateJsonSchema().generate(schema),
        pydantic_core.SchemaValidator(schema),
        tuple(parameter.arg for parameter in args.posonlyargs),
    )


@dataclasses.dataclass(kw_only=True)
class ScriptTool(ToolsetTool):
    """Carry a script path and its permitted host names through wrappers."""

    path: pathlib.Path
    allowed_tools: tuple[str, ...]
    positional: tuple[str, ...]


@dataclasses.dataclass
class MontyToolset(AbstractToolset):
    """Discover direct Monty files and execute their validated calls."""

    path: pathlib.Path
    allowed_tools: tuple[str, ...]
    tools: dict[str, ScriptTool] = dataclasses.field(init=False)

    @property
    def id(self):
        return None

    def __post_init__(self):
        self.tools = {}
        for path in sorted(self.path.glob("*.mpy")):
            if not path.is_file() or not re.fullmatch(
                r"[a-z][a-z0-9_]*", path.stem
            ):
                continue
            try:
                description, schema, validator, positional = _signature(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, SyntaxError) as exc:
                _LOGGER.warning("Skipping Monty script %s: %s", path, exc)
                continue
            self.tools[path.stem] = ScriptTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name=path.stem,
                    description=description,
                    parameters_json_schema=schema,
                ),
                max_retries=1,
                args_validator=validator,
                path=path,
                allowed_tools=self.allowed_tools,
                positional=positional,
            )

    async def get_tools(self, ctx):
        return self.tools

    async def call_tool(self, name, tool_args, ctx, tool):
        return await _execute(tool, tool_args, ctx)


@dataclasses.dataclass
class MontyHostToolset(WrapperToolset):
    """Attach host tools and preserve native skill visibility."""

    async def call_tool(self, name, tool_args, ctx, tool):
        token = _HOST_TOOLSET.set(self)
        try:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        finally:
            _HOST_TOOLSET.reset(token)


def _raise_failure(error: Exception):
    """Propagate a recorded sandbox or host failure."""
    raise error


async def _execute(tool: ScriptTool, arguments: dict, ctx: RunContext) -> str:
    """Run a fresh checked sandbox and return output or an error as text."""
    state = MontyRunState()
    capture = PrintCapture()
    try:
        source = tool.path.read_text(encoding="utf-8")
        host_toolset = _HOST_TOOLSET.get()
        hosts = {
            name: host
            for name, host in (await host_toolset.get_tools(ctx)).items()
            if name in tool.allowed_tools
            and ctx.is_tool_available(host.tool_def)
        }
        definitions, original_names = partition_callable_tools(
            hosts, owner=str(tool.path), reserved_name="main"
        )
        definitions = {
            name: dataclasses.replace(definition, sequential=True)
            for name, definition in definitions.items()
        }
        manager = ToolManager(
            toolset=host_toolset,
            root_capability=ctx.tool_manager.root_capability,
            ctx=ctx,
            tools=hosts,
        )
        calls = 0
        failure = None

        async def dispatch(name, kwargs):
            nonlocal calls, failure
            calls += 1
            if calls > _MAX_HOST_CALLS:
                failure = RuntimeError("Host call limit of 100 exceeded")
                _raise_failure(failure)
            outcome = await run_nested_call(
                manager,
                ToolCallPart(
                    original_names.get(name, name),
                    kwargs,
                    tool_call_id=f"{ctx.tool_call_id}_monty_{calls}",
                ),
            )
            if outcome.error is not None:
                failure = outcome.error
                _raise_failure(failure)
            return outcome.content

        session = await state.get_session(
            type_check=True,
            type_check_stubs=CodeModeToolset._build_type_check_stubs(
                definitions
            ),
            limits=_LIMITS,
        )
        call = ", ".join(
            repr(value) if key in tool.positional else f"{key}={value!r}"
            for key, value in arguments.items()
        )
        completed = await MontyExecutor(
            dispatch=dispatch,
            valid_names=definitions,
            sequential_names=set(definitions),
            max_sleep_secs=_LIMITS["max_feed_duration_secs"],
        ).run(
            functools.partial(
                session.feed_start,
                source + f"\nmain({call})\n",
                print_callback=capture.callback,
            )
        )
        if failure is not None:
            _raise_failure(failure)
        return capture.joined + json.dumps(completed.output)
    except MontyError as exc:
        return f"SCRIPT ERROR: {exc.display()}"
    except Exception as exc:
        return f"SCRIPT ERROR: {exc}"
    finally:
        await state.close()
