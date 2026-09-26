"""A room agent which calls the 'shell' client tool, scripted

Used by 'test_client_tools.py' through a 'kind: factory' room config.
No LLM:  a pydantic-ai 'FunctionModel' decides from the history.

- No 'shell' result in the history yet:  call 'shell' (and, when the
  prompt says "mixed", the server-side 'server_tool' alongside it).
- Otherwise:  answer with the last 'shell' result's content.

Every request's history is appended to 'HISTORIES', so a test can check
what the server handed the model.
"""

from __future__ import annotations

import pydantic_ai
from pydantic_ai import messages as ai_messages
from pydantic_ai.models import function as ai_function

SHELL_COMMAND = "echo client-tool-ran"

HISTORIES: list[list[ai_messages.ModelMessage]] = []


def server_tool() -> str:
    """A tool the server runs itself"""
    return "server-tool-ran"


def _prompt(messages) -> str:
    return next(
        part.content
        for message in messages
        if isinstance(message, ai_messages.ModelRequest)
        for part in message.parts
        if isinstance(part, ai_messages.UserPromptPart)
    )


def _shell_returns(messages) -> list[ai_messages.ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ai_messages.ModelRequest)
        for part in message.parts
        if isinstance(part, ai_messages.ToolReturnPart)
        and part.tool_name == "shell"
    ]


async def _stream(messages, info: ai_function.AgentInfo):
    HISTORIES.append(list(messages))

    returns = _shell_returns(messages)

    if returns:
        yield f"shell said: {returns[-1].content}"
        return

    calls = {
        0: ai_function.DeltaToolCall(
            name="shell",
            json_args=f'{{"command": "{SHELL_COMMAND}"}}',
            tool_call_id="shell-call-1",
        ),
    }

    if "mixed" in _prompt(messages):
        calls[1] = ai_function.DeltaToolCall(
            name="server_tool",
            json_args="{}",
            tool_call_id="server-call-1",
        )

    yield calls


def agent_factory(**_kwargs) -> pydantic_ai.Agent:
    return pydantic_ai.Agent(
        model=ai_function.FunctionModel(stream_function=_stream),
        tools=[server_tool],
        name="scripted-shell",
    )
