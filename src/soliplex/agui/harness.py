"""A context harness for AG-UI clients which own their history

A client of the AG-UI endpoints ('soliplex-cli ask --url', the TUI)
sends the whole thread -- every message, and the state -- with each run,
and every run's model request carries it.  This module holds what such
a client does to that history before each POST, so that clients can be
thin layers over it:

- 'validate_pairing':  refuse, locally, a history the server could not
  load (a tool result without its call, a call answered twice, ...),
  rather than send it;
- compaction ('compact_history'):  rewrite the content of old, large
  tool results in place -- a haiku.rag search keeps its chunk ids and
  headers -- so that each later request carries less.  No message is
  removed, merged or moved, and no call loses its result:  the server's
  history (and haiku.rag's per-question ledger, which counts messages)
  is unchanged in shape.  A compacted result starts with
  'COMPACTED_MARKER', so that any client reading the thread back can
  tell, and is byte-identical on every later resend.

'Harness' ties it together:  a client makes one per thread, and hands
its 'before_post' to 'client_tools.run_loop'.
"""

from __future__ import annotations

import dataclasses
import json
import re
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
#   Compaction:  rewriting old tool results
#
COMPACTION_OFF = "off"
COMPACTION_ALWAYS = "always"
COMPACTION_MODES = (COMPACTION_OFF, COMPACTION_ALWAYS)

DEFAULT_KEEP_RECENT = 4
DEFAULT_MIN_ELIDE_CHARS = 1024

#   Every compacted tool result starts with this, then 'key=value' words
#   ('tool', 'format', 'original_bytes') and ']':  see 'compacted_info'.
COMPACTED_MARKER = "[compacted by soliplex-tui harness:"
_COMPACTED_HEADER = re.compile(
    re.escape(COMPACTED_MARKER) + r"((?: [a-z_]+=\S+)+)\]",
)

#   Results never compacted, whatever their size:  haiku.rag's citation
#   acknowledgements, and the tools whose returns pydantic-ai re-reads to
#   know which deferred capabilities and tools are loaded.
PROTECTED_TOOLS = frozenset(["cite", "load_capability", "search_tools"])

#   How much of the start and end of a result is kept, in characters.
EXECUTE_CODE_KEEP_CHARS = 600
GENERIC_KEEP_CHARS = 300
STDERR_TAIL_CHARS = 400

#   A haiku.rag search result:  hits joined by '---', each a header block
#   (its first line '[chunk-id] [rank i of n]') then 'Content:' and the
#   (context-expanded) text.
_SEARCH_HIT_SEPARATOR = "\n\n---\n\n"
_SEARCH_HIT_FIRST_LINE = re.compile(
    r"\[[^\]\n]+\] (?:\[rank \d+(?: of \d+)?\]|\(score: [-\d.]+\))",
)
_SEARCH_ALSO_MATCHED = "Also matched, shown above: "
_SEARCH_HEADER_PREFIXES = (
    "[",
    "Document ID: ",
    "Collection: ",
    "Source: ",
    "Type: ",
)
SEARCH_NOTE = (
    "Bodies omitted to save context; cite these chunk ids directly, or "
    "search again for the text."
)
SHELL_NOTE = "Output compacted by the client; re-run the command if needed."


class InvalidCompactionMode(ValueError):
    def __init__(self, mode):
        super().__init__(
            f"Unknown compaction mode {mode!r} "
            f"(one of: {', '.join(COMPACTION_MODES)})",
        )


@dataclasses.dataclass(frozen=True)
class CompactionPolicy:
    """Which old tool results to compact, before each POST

    'mode':

    - 'off':  none;
    - 'always':  every eligible result but the newest 'keep_recent'.

    A result is eligible if it is at least 'min_elide_chars' long, not
    compacted yet, answers a call of a tool not in 'PROTECTED_TOOLS', and
    neither it nor its call carries an 'encrypted_value'.
    """

    mode: str = COMPACTION_OFF
    keep_recent: int = DEFAULT_KEEP_RECENT
    min_elide_chars: int = DEFAULT_MIN_ELIDE_CHARS

    def __post_init__(self):
        if self.mode not in COMPACTION_MODES:
            raise InvalidCompactionMode(self.mode)


def is_compacted(content: str) -> bool:
    return content.startswith(COMPACTED_MARKER)


def compacted_info(content: str) -> dict[str, str] | None:
    """The 'key=value' words of a compacted result's first line, or None

    E.g. '{"tool": "search", "format": "headers", "original_bytes":
    "10612"}':  what a client reading the thread back needs to show "this
    result was compacted", and how large it was.
    """
    match = _COMPACTED_HEADER.match(content)

    if match is None:
        return None

    return dict(word.split("=", 1) for word in match.group(1).split())


def _header(tool: str, fmt: str, original: str) -> str:
    size = len(original.encode())
    return (
        f"{COMPACTED_MARKER} tool={tool} format={fmt} original_bytes={size}]"
    )


def _head_tail(text: str, keep: int) -> str:
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n...[{omitted} chars omitted]...\n{text[-keep:]}"


def _search_skeleton(content: str) -> str | None:
    """A haiku.rag search result's headers, or None if it is not one"""
    kept = []

    for hit in content.split(_SEARCH_HIT_SEPARATOR):
        first = hit.partition("\n")[0]

        if first.startswith(_SEARCH_ALSO_MATCHED):
            kept.append(first)
            continue

        block, content, _ = hit.partition("\nContent:")

        if not content or not _SEARCH_HIT_FIRST_LINE.match(first):
            return None

        header = [first] + [
            line
            for line in block.split("\n")[1:]
            if line.startswith(_SEARCH_HEADER_PREFIXES)
        ]
        kept.append("\n".join(header))

    return "\n---\n".join(kept)


def _shell_summary(content: str) -> str | None:
    """A 'shell' result's outcome, without its output;  None if not one"""
    try:
        result = json.loads(content)
    except ValueError:
        return None

    if not isinstance(result, dict) or "exit_code" not in result:
        return None

    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    summary = {
        "compacted": True,
        "exit_code": result["exit_code"],
        "timed_out": result.get("timed_out", False),
        "stdout_chars": len(stdout),
        "stderr_chars": len(stderr),
    }

    # Never let a failure read as a success:  keep why.
    if result["exit_code"] != 0 and stderr:
        summary["stderr_tail"] = stderr[-STDERR_TAIL_CHARS:]

    if result.get("error"):
        summary["error"] = result["error"]

    summary["note"] = SHELL_NOTE
    return json.dumps(summary)


def compact_content(tool: str, content: str) -> str:
    """The compacted form of a result of 'tool' holding 'content'

    Depends on nothing else, so that the same result always compacts to
    the same bytes (a history resent after compaction keeps the model
    server's cached prefix).  The first line is the marker (see
    'compacted_info'), then:

    - a haiku.rag search ('format=headers'):  each hit's header block
      (chunk id and rank, collection, source, type), without its text;
    - a client 'shell' result ('format=shell'):  exit code, timeout and
      output sizes as JSON, with the end of stderr if it failed;
    - 'execute_code' ('format=head_tail'):  its first and last
      'EXECUTE_CODE_KEEP_CHARS' characters;
    - anything else ('format=head_tail'):  its first and last
      'GENERIC_KEEP_CHARS'.
    """
    skeleton = _search_skeleton(content)

    if skeleton is not None:
        body, fmt = f"{SEARCH_NOTE}\n{skeleton}", "headers"
    elif (summary := _shell_summary(content)) is not None:
        body, fmt = summary, "shell"
    else:
        keep = (
            EXECUTE_CODE_KEEP_CHARS
            if tool == "execute_code"
            else GENERIC_KEEP_CHARS
        )
        body, fmt = _head_tail(content, keep), "head_tail"

    return f"{_header(tool, fmt, content)}\n{body}"


def _calls_by_id(
    messages: abc.Sequence[agui_core.Message],
) -> dict[str, agui_core.ToolCall]:
    return {
        call.id: call
        for message in messages
        if isinstance(message, agui_core.AssistantMessage)
        for call in message.tool_calls or ()
    }


def compaction_candidates(
    messages: abc.Sequence[agui_core.Message],
    policy: CompactionPolicy,
) -> list[int]:
    """Indexes of the results 'policy' allows compacting, oldest first

    Leaves out the newest 'policy.keep_recent' of the eligible results
    (see 'CompactionPolicy'), and any result whose compacted form would
    not be shorter.
    """
    calls = _calls_by_id(messages)
    eligible = []

    for index, message in enumerate(messages):
        if not isinstance(message, agui_core.ToolMessage):
            continue

        call = calls.get(message.tool_call_id)

        if (
            call is None
            or call.function.name in PROTECTED_TOOLS
            or call.encrypted_value is not None
            or message.encrypted_value is not None
            or len(message.content) < policy.min_elide_chars
            or is_compacted(message.content)
        ):
            continue

        eligible.append(index)

    if policy.keep_recent:
        eligible = eligible[: -policy.keep_recent]

    def shorter(index: int) -> bool:
        message = messages[index]
        tool = calls[message.tool_call_id].function.name
        return len(compact_content(tool, message.content)) < len(
            message.content,
        )

    return [index for index in eligible if shorter(index)]


def compact_history(
    messages: abc.Sequence[agui_core.Message],
    indexes: abc.Iterable[int],
) -> list[agui_core.Message]:
    """A copy of 'messages', the results at 'indexes' compacted

    Only those results' content changes:  every message keeps its id and
    place, and every other message is the same object.
    """
    calls = _calls_by_id(messages)
    compacted = list(messages)

    for index in indexes:
        message = compacted[index]
        tool = calls[message.tool_call_id].function.name
        compacted[index] = message.model_copy(
            update={"content": compact_content(tool, message.content)},
        )

    return compacted


def wire_chars(messages: abc.Sequence[agui_core.Message]) -> int:
    """The size of 'messages' as JSON, in characters"""
    return len(
        json.dumps(
            [
                message.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
                for message in messages
            ],
            ensure_ascii=False,
        ),
    )


@dataclasses.dataclass(frozen=True)
class ResendReport:
    """What 'Harness.before_post' did to one history, and its size

    'resend_chars' ('wire_chars') and 'state_chars' are the sizes of the
    history's messages and state as sent;  'compacted' how many results
    this POST compacted (saving 'compacted_chars'), and
    'compacted_total' how many of the history's results are compacted.
    """

    thread_id: str
    run_id: str
    parent_run_id: str | None
    first: bool
    mode: str
    messages: int
    resend_chars: int
    state_chars: int
    compacted: int
    compacted_chars: int
    compacted_total: int

    def as_json(self) -> dict:
        return dataclasses.asdict(self)


#
#   The harness
#
@dataclasses.dataclass
class Harness:
    """What a client does to one thread's history before each POST

    Pass 'before_post' to 'client_tools.run_loop'.  It compacts the
    history as 'compaction' says;  then, with 'pairing_check', a history
    'validate_pairing' refuses is never sent:  'run_loop' raises
    'InconsistentHistory' instead.  Each POST's 'ResendReport' is appended
    to 'reports', and passed to 'on_report' (e.g., for a UI to show).
    """

    pairing_check: bool = True
    compaction: CompactionPolicy = CompactionPolicy()
    on_report: abc.Callable[[ResendReport], None] | None = None
    reports: list[ResendReport] = dataclasses.field(default_factory=list)

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
        messages = run_input.messages
        saved = 0
        indexes = []

        if self.compaction.mode != COMPACTION_OFF:
            indexes = compaction_candidates(messages, self.compaction)

        if indexes:
            before = wire_chars(messages)
            messages = compact_history(messages, indexes)
            run_input = run_input.model_copy(update={"messages": messages})
            saved = before - wire_chars(messages)

        if self.pairing_check:
            validate_pairing(messages)

        report = ResendReport(
            thread_id=run_input.thread_id,
            run_id=run_input.run_id,
            parent_run_id=run_input.parent_run_id,
            first=first,
            mode=self.compaction.mode,
            messages=len(messages),
            resend_chars=wire_chars(messages),
            state_chars=len(json.dumps(run_input.state, ensure_ascii=False)),
            compacted=len(indexes),
            compacted_chars=saved,
            compacted_total=sum(
                1
                for message in messages
                if isinstance(message, agui_core.ToolMessage)
                and is_compacted(message.content)
            ),
        )
        self.reports.append(report)

        if self.on_report is not None:
            self.on_report(report)

        return run_input
