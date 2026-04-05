"""Client-side tool definitions for the Soliplex TUI.

Tools defined here are advertised to the backend via AG-UI and
executed locally when the LLM issues a tool call matching a
registered name.
"""

from __future__ import annotations

import json
import typing

from ag_ui import core as agui_core

# -- Tool executor type ------------------------------------------------------

ToolExecutor = typing.Callable[[str], str]
"""Takes the JSON-encoded arguments string, returns a result string."""

# -- Registry ----------------------------------------------------------------


class ClientToolRegistry:
    """Immutable registry of client-side tools."""

    def __init__(
        self,
        tools: dict[str, tuple[agui_core.Tool, ToolExecutor]] | None = None,
    ):
        self._tools: dict[str, tuple[agui_core.Tool, ToolExecutor]] = (
            dict(tools) if tools else {}
        )

    def register(
        self,
        definition: agui_core.Tool,
        executor: ToolExecutor,
    ) -> None:
        self._tools[definition.name] = (definition, executor)

    @property
    def tool_definitions(self) -> list[agui_core.Tool]:
        return [defn for defn, _ in self._tools.values()]

    def contains(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, arguments: str) -> str:
        _, executor = self._tools[name]
        return executor(arguments)


# -- Built-in tools ----------------------------------------------------------


def _echo_executor(arguments: str) -> str:
    args = json.loads(arguments) if arguments else {}
    return args.get("text", "")


ECHO_TOOL = agui_core.Tool(
    name="echo",
    description="Echoes back the text argument.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to echo back.",
            },
        },
        "required": ["text"],
    },
)


def build_default_registry() -> ClientToolRegistry:
    registry = ClientToolRegistry()
    registry.register(ECHO_TOOL, _echo_executor)
    return registry
