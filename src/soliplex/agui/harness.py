"""A context harness for AG-UI clients which own their history

A client of the AG-UI endpoints ('soliplex-cli ask --url', the TUI)
sends the whole thread -- every message, and the state -- with each run,
and every run's model request carries it.  This module holds what such
a client does to that history before each POST, so that clients can be
thin layers over it:

- 'validate_pairing':  refuse, locally, a history the server could not
  load (a tool result without its call, a call answered twice, ...),
  rather than send it.

'Harness' ties it together:  a client makes one per thread, and hands
its 'before_post' to 'client_tools.run_loop'.
"""

from __future__ import annotations

import dataclasses
from collections import abc

from ag_ui import core as agui_core

from soliplex.agui import client_tools


class InconsistentHistory(client_tools.ClientToolsError):
    """A history 'validate_pairing' refused to send

    'problems' says what is wrong with it, one line each.
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            "Refusing to send an inconsistent history: " + "; ".join(problems),
        )


#
#   Pairing
#
def pairing_problems(
    messages: abc.Sequence[agui_core.Message],
    *,
    allow_pending: bool = False,
) -> list[str]:
    """What makes 'messages' a history the server cannot load, if anything

    - tool call ids must be unique (results are matched to calls by id);
    - every tool result must answer a call made by an *earlier* assistant
      message (else pydantic-ai fails the run:  'Tool call with ID ...
      not found in the history');
    - no call may have more than one result;
    - unless 'allow_pending', every call must have its result (the server
      would silently drop a trailing unanswered call, and fail on any
      other).

    Message ids are not checked:  the server does not rely on them, and
    older TUI threads can hold duplicate user-message ids, which must not
    stop the thread going on.
    """
    problems: list[str] = []
    calls: dict[str, str] = {}  # call id -> tool name
    answered: set[str] = set()

    for message in messages:
        if isinstance(message, agui_core.AssistantMessage):
            for call in message.tool_calls or ():
                if call.id in calls:
                    problems.append(f"duplicate tool call id {call.id!r}")

                calls[call.id] = call.function.name

        elif isinstance(message, agui_core.ToolMessage):
            call_id = message.tool_call_id

            if call_id not in calls:
                problems.append(
                    f"tool result {message.id!r} answers {call_id!r}, "
                    f"which no earlier assistant message calls",
                )
            elif call_id in answered:
                problems.append(
                    f"tool call {call_id!r} has more than one result",
                )

            answered.add(call_id)

    if not allow_pending:
        problems.extend(
            f"tool call {call_id!r} ({name}) has no result"
            for call_id, name in calls.items()
            if call_id not in answered
        )

    return problems


def validate_pairing(
    messages: abc.Sequence[agui_core.Message],
    *,
    allow_pending: bool = False,
) -> None:
    """Raise 'InconsistentHistory' if 'pairing_problems' finds any"""
    problems = pairing_problems(messages, allow_pending=allow_pending)

    if problems:
        raise InconsistentHistory(problems)


#
#   The harness
#
@dataclasses.dataclass
class Harness:
    """What a client does to one thread's history before each POST

    Pass 'before_post' to 'client_tools.run_loop'.  With 'pairing_check',
    a history 'validate_pairing' refuses is never sent:  'run_loop'
    raises 'InconsistentHistory' instead.
    """

    pairing_check: bool = True

    def before_post(
        self,
        client: client_tools.SoliplexClient,
        run_input: agui_core.RunAgentInput,
        *,
        first: bool,
    ) -> agui_core.RunAgentInput:
        """The history to send, in place of 'run_input'

        'first' is true for the first run of a prompt, false for the
        runs 'run_loop' makes to send client tool results back.
        """
        if self.pairing_check:
            validate_pairing(run_input.messages)

        return run_input
