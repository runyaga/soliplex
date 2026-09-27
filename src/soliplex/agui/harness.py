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
  tell, and is byte-identical on every later resend;
- RAG-state trimming ('trim_rag_state'):  once a question is answered,
  the working evidence haiku.rag keeps in the AG-UI state (every
  expanded search result) is dropped before the next prompt -- as the
  server itself drops it when the next question starts -- so it is not
  uploaded, and stored, with every run;
- the context budget ('ContextBudget'):  the model's window (from an
  option, the room, or the model server), and an estimate of how much
  of it the next request takes, anchored on the tokens the server
  measured for the last run.  In 'auto' mode, compaction waits until the
  estimate crosses a high-water mark, then compacts down to a low-water
  mark in one batch:  a model server caching prompt prefixes (vLLM)
  keeps its cache between batches, since the history only grows.

'Harness' ties it together:  a client makes one per thread, and hands
its 'before_post' to 'client_tools.run_loop'.
"""

from __future__ import annotations

import dataclasses
import json
import re
import typing
import uuid
from collections import abc
from urllib import parse as urllib_parse

import httpx
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


#   The result given a call left without one (see 'answer_unanswered').
UNANSWERED_ERROR = (
    "No result was recorded for this call:  the run ended before its "
    "result was sent (it was cancelled, interrupted, or stopped at the "
    "turn limit).  Whether it ran, and what it did, is unknown:  check "
    "its effects before running it again."
)


def answer_unanswered(
    messages: abc.Sequence[agui_core.Message],
) -> tuple[list[agui_core.Message], int]:
    """'messages', each call without a result given a "not run" one

    A thread reloaded after its client stopped mid-way -- a run
    cancelled, or interrupted while a client tool ran, or stopped at
    '--max-turns' -- holds calls the server stored, but whose results
    were never sent.  Each gets a result saying so ('UNANSWERED_ERROR'),
    placed after its call's message and the results already following
    it:  nothing is run again.  The result says the outcome is unknown,
    not that the call did not run:  a command may have run (and changed
    things) before its client stopped.  Returns the history, and how many
    results it added.
    """
    answered = {
        message.tool_call_id
        for message in messages
        if isinstance(message, agui_core.ToolMessage)
    }
    content = client_tools.tool_result_content(
        client_tools._not_run(UNANSWERED_ERROR),
    )
    found: list[agui_core.Message] = []
    waiting: dict[str, agui_core.ToolCall] = {}
    added = 0

    def flush() -> None:
        nonlocal added

        for call_id in waiting:
            found.append(
                agui_core.ToolMessage(
                    id=uuid.uuid4().hex,
                    tool_call_id=call_id,
                    content=content,
                ),
            )
            answered.add(call_id)
            added += 1

        waiting.clear()

    for message in messages:
        if not isinstance(message, agui_core.ToolMessage):
            flush()

        found.append(message)

        if isinstance(message, agui_core.AssistantMessage):
            for call in message.tool_calls or ():
                if call.id not in answered:
                    waiting.setdefault(call.id, call)

    flush()
    return found, added


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
COMPACTION_AUTO = "auto"
COMPACTION_ALWAYS = "always"
COMPACTION_MODES = (COMPACTION_OFF, COMPACTION_AUTO, COMPACTION_ALWAYS)

DEFAULT_KEEP_RECENT = 4
DEFAULT_MIN_ELIDE_CHARS = 1024
#   'auto':  compact once the estimate is over this fraction of the usable
#   window, down to under that one, in one batch.
DEFAULT_TRIGGER_FRACTION = 0.70
DEFAULT_TARGET_FRACTION = 0.40

#   Every compacted tool result starts with this, then 'key=value' words
#   ('tool', 'format', 'original_bytes') and ']':  see 'compacted_info'.
#   The tool's name is percent-encoded ('urllib.parse.quote', nothing
#   safe), so it holds no space, ']' or '='.
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
_SEARCH_HIT_FIRST_LINE = re.compile(
    r"\[[^\]\n]+\] (?:\[rank \d+(?: of \d+)?\]|\(score: [-\d.]+\))$",
)
_SEARCH_ALSO_MATCHED = "Also matched, shown above: "
#   The separator between hits:  only where a hit's first line follows it,
#   not a horizontal rule within a hit's text.
_SEARCH_HIT_SEPARATOR = re.compile(
    r"\n\n---\n\n(?=\[[^\]\n]+\] (?:\[rank |\(score: )"
    + "|"
    + re.escape(_SEARCH_ALSO_MATCHED)
    + ")",
)
SEARCH_TOOL = "search"
SHELL_TOOL = "shell"
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


class InvalidCompactionCounts(ValueError):
    def __init__(self, keep_recent, min_elide_chars):
        super().__init__(
            f"Compaction needs keep_recent ({keep_recent}) >= 0 and "
            f"min_elide_chars ({min_elide_chars}) >= 1",
        )


class InvalidCompactionFractions(ValueError):
    def __init__(self, trigger, target):
        super().__init__(
            f"Compaction needs 0 < target ({target}) < trigger ({trigger}) "
            f"<= 1",
        )


@dataclasses.dataclass(frozen=True)
class CompactionPolicy:
    """Which old tool results to compact, before each POST

    'mode':

    - 'off':  none;
    - 'always':  every eligible result but the newest 'keep_recent';
    - 'auto':  nothing, until the next request's estimate is over
      'trigger_fraction' of the usable window;  then, oldest first, as
      many eligible results as bring it under 'target_fraction', in one
      batch.  (Nothing, with no window known:  see 'ContextBudget'.)

    A result is eligible if it is at least 'min_elide_chars' long, not
    compacted yet, answers a call of a tool not in 'PROTECTED_TOOLS', and
    neither it nor its call carries an 'encrypted_value'.
    """

    mode: str = COMPACTION_AUTO
    keep_recent: int = DEFAULT_KEEP_RECENT
    min_elide_chars: int = DEFAULT_MIN_ELIDE_CHARS
    trigger_fraction: float = DEFAULT_TRIGGER_FRACTION
    target_fraction: float = DEFAULT_TARGET_FRACTION

    def __post_init__(self):
        if self.mode not in COMPACTION_MODES:
            raise InvalidCompactionMode(self.mode)

        if self.keep_recent < 0 or self.min_elide_chars < 1:
            raise InvalidCompactionCounts(
                self.keep_recent,
                self.min_elide_chars,
            )

        if not 0 < self.target_fraction < self.trigger_fraction <= 1:
            raise InvalidCompactionFractions(
                self.trigger_fraction,
                self.target_fraction,
            )


def is_compacted(content: str) -> bool:
    """Does 'content' start with a marker line 'compacted_info' can read?"""
    return _COMPACTED_HEADER.match(content) is not None


def compacted_info(content: str) -> dict[str, str] | None:
    """The 'key=value' words of a compacted result's first line, or None

    E.g. '{"tool": "search", "format": "headers", "original_bytes":
    "10612"}':  what a client reading the thread back needs to show "this
    result was compacted", and how large it was.
    """
    match = _COMPACTED_HEADER.match(content)

    if match is None:
        return None

    info = dict(word.split("=", 1) for word in match.group(1).split())

    if "tool" in info:
        info["tool"] = urllib_parse.unquote(info["tool"])

    return info


def _header(tool: str, fmt: str, original: str) -> str:
    size = len(original.encode())
    name = urllib_parse.quote(tool, safe="")
    return (
        f"{COMPACTED_MARKER} tool={name} format={fmt} original_bytes={size}]"
    )


def _head_tail(text: str, keep: int) -> str:
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n...[{omitted} chars omitted]...\n{text[-keep:]}"


def _search_skeleton(content: str) -> str | None:
    """A haiku.rag search result's headers, or None if it is not one

    Each hit keeps its whole header block:  every line before its
    'Content:' line (chunk id and rank, then e.g. collection, source, type
    and figure captions).  A hit already shown ('Also matched, ...')
    keeps its one line.  ('Content:' ends a line of its own, so a header
    which merely holds the word -- a title with a newline in it -- is not
    taken for the end of the block.)
    """
    kept = []

    for hit in _SEARCH_HIT_SEPARATOR.split(content):
        first = hit.partition("\n")[0]

        if first.startswith(_SEARCH_ALSO_MATCHED) and first == hit:
            kept.append(first)
            continue

        # More than one 'Content:' line (e.g. a title with one in it):
        # where the header ends is ambiguous.
        if hit.count("\nContent:\n") != 1:
            return None

        block, _, _ = hit.partition("\nContent:\n")

        if not _SEARCH_HIT_FIRST_LINE.match(first):
            return None

        kept.append(block)

    return "\n---\n".join(kept)


def _is_text(value) -> bool:
    return value is None or isinstance(value, str)


def _shell_summary(content: str) -> str | None:
    """A 'shell' result's outcome, without its output;  None if not one

    Only for a result of the shape 'client_tools.tool_result_content'
    makes:  anything else is left alone, lest a failure be lost.
    """
    try:
        result = json.loads(content)
    except (ValueError, RecursionError):  # e.g., '[[[...' nested too deep
        return None

    if (
        not isinstance(result, dict)
        or "exit_code" not in result
        or not (
            result["exit_code"] is None or type(result["exit_code"]) is int
        )
        or not all(
            _is_text(result.get(key)) for key in ("stdout", "stderr", "error")
        )
        or not isinstance(result.get("timed_out", False), bool)
        or not isinstance(result.get("truncated", False), bool)
    ):
        return None

    stdout = result.get("stdout") or ""
    stderr = result.get("stderr") or ""
    summary = {
        "compacted": True,
        "exit_code": result["exit_code"],
        "timed_out": result.get("timed_out", False),
        "truncated": result.get("truncated", False),
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
    server's cached prefix);  content already compacted is returned as
    it is.  The first line is the marker (see 'compacted_info'), then:

    - a haiku.rag search ('format=headers'):  each hit's header block
      (chunk id and rank, collection, source, type, ...), without its
      text;
    - a client 'shell' result ('format=shell'):  exit code, timeout and
      output sizes as JSON, with the end of stderr if it failed;
    - 'execute_code' ('format=head_tail'):  its first and last
      'EXECUTE_CODE_KEEP_CHARS' characters;
    - anything else ('format=head_tail'):  its first and last
      'GENERIC_KEEP_CHARS'.

    A 'search' or 'shell' result which is not of the expected shape is
    returned as it is:  cutting it could lose its chunk ids, or a
    failure.
    """
    if is_compacted(content):
        return content

    # A 'shell' result is only ever summarized as one:  whatever else it
    # looks like, its exit status must survive.
    if tool == SHELL_TOOL:
        skeleton, summary = None, _shell_summary(content)
    else:
        skeleton, summary = _search_skeleton(content), None

    if skeleton is not None:
        body, fmt = f"{SEARCH_NOTE}\n{skeleton}", "headers"
    elif summary is not None:
        body, fmt = summary, "shell"
    elif tool in (SEARCH_TOOL, SHELL_TOOL):
        return content
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

    def shorter(index: int) -> bool:  # as sent:  UTF-8 JSON
        message = messages[index]
        tool = calls[message.tool_call_id].function.name
        compacted = compact_content(tool, message.content)
        return _wire_bytes(compacted) < _wire_bytes(message.content)

    return [index for index in eligible if shorter(index)]


def _wire_bytes(content: str) -> int:
    """'content''s size in a POST's body:  UTF-8 JSON, not ASCII-escaped"""
    return len(json.dumps(content, ensure_ascii=False).encode())


def compact_history(
    messages: abc.Sequence[agui_core.Message],
    indexes: abc.Iterable[int],
) -> list[agui_core.Message]:
    """A copy of 'messages', the results at 'indexes' compacted

    'indexes' should come from 'compaction_candidates', which is what
    leaves out protected and encrypted results.  Only those results'
    content changes:  every message keeps its id and
    place, and every other message (including a result 'compact_content'
    leaves as it is) is the same object.
    """
    calls = _calls_by_id(messages)
    compacted = list(messages)

    for index in sorted(set(indexes)):
        message = compacted[index]
        tool = calls[message.tool_call_id].function.name
        content = compact_content(tool, message.content)

        if content != message.content:
            compacted[index] = message.model_copy(update={"content": content})

    return compacted


def wire_chars(messages: abc.Sequence[agui_core.Message]) -> int:
    """The size of 'messages' as JSON, in characters

    A stable measure of what the history costs to send, and of what the
    model reads:  JSON without null fields, so close to (not exactly)
    the size of the POST's 'messages'.
    """
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


#
#   RAG state
#
TRIM_OFF = "off"
TRIM_BOUNDARY = "boundary"
TRIM_AGGRESSIVE = "aggressive"
TRIM_MODES = (TRIM_OFF, TRIM_BOUNDARY, TRIM_AGGRESSIVE)

#   A haiku.rag 'RAGState''s working evidence, and its empty value.  At a
#   finished question all of it goes (as 'RAGState.begin_invocation'
#   drops it when the next starts);  mid-question ('aggressive'), only the
#   search results and executions.
_BOUNDARY_FIELDS = {"searches": {}, "executions": [], "citations": []}
_AGGRESSIVE_FIELDS = {"searches": {}, "executions": []}


class InvalidTrimMode(ValueError):
    def __init__(self, mode):
        super().__init__(
            f"Unknown RAG-state trim mode {mode!r} "
            f"(one of: {', '.join(TRIM_MODES)})",
        )


def _rag_state_in_progress(value) -> bool | None:
    """For a haiku.rag 'RAGState' dump, whether its question is open

    None if 'value' is not one:  a dict with an 'evidence' record saying
    'in_progress', 'searches' and a 'citation_index'.  (Keyed by any
    name:  a room may have several RAG capabilities, each its own state
    namespace.)
    """
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("searches"), dict)
        or not isinstance(value.get("citation_index"), dict)
    ):
        return None

    evidence = value.get("evidence")

    if not isinstance(evidence, dict) or not isinstance(
        evidence.get("in_progress"),
        bool,
    ):
        return None

    return evidence["in_progress"]


def trim_rag_state(state, mode: str) -> tuple[typing.Any, list[str]]:
    """'state', its RAG namespaces' working evidence dropped as 'mode' says

    - 'off':  nothing;
    - 'boundary':  in each namespace whose question is finished
      ('evidence.in_progress' false), 'searches', 'executions' and
      'citations' are emptied.  Lossless:  the server drops them itself
      when the next question starts.  'citation_index', 'evidence',
      'document_filter' and 'sources' are kept;
    - 'aggressive':  as 'boundary', and mid-question too, 'searches' and
      'executions'.  Lossy:  'cite' can then no longer correct a mangled
      chunk id against the question's results, and falls back to the
      database (whose citations carry no expanded text).

    Returns the state (the same object if nothing was trimmed:  trimming
    is idempotent), and the namespaces trimmed.
    """
    if mode not in TRIM_MODES:
        raise InvalidTrimMode(mode)

    if mode == TRIM_OFF or not isinstance(state, dict):
        return state, []

    trimmed = dict(state)
    names = []

    for name, value in state.items():
        in_progress = _rag_state_in_progress(value)

        if in_progress is None or (in_progress and mode == TRIM_BOUNDARY):
            continue

        fields = _AGGRESSIVE_FIELDS if in_progress else _BOUNDARY_FIELDS
        emptied = {
            field: type(empty)()
            for field, empty in fields.items()
            if field in value and value[field] != empty
        }

        if emptied:
            trimmed[name] = value | emptied
            names.append(name)

    return (trimmed if names else state), names


#
#   The context budget
#
DEFAULT_OUTPUT_RESERVE = 4096
#   Without a tokenizer:  about this many characters of JSON per token.
DEFAULT_CHARS_PER_TOKEN = 3.5

WINDOW_FROM_OPTION = "option"
WINDOW_FROM_ROOM = "room"
WINDOW_FROM_MODEL_SERVER = "model server"


def room_context_window(room_info: dict) -> int | None:
    """The room's declared context window ('agent.context_window'), if any"""
    agent = room_info.get("agent") or {}
    return agent.get("context_window")


def probe_model_window(
    room_info: dict,
    http: httpx.Client,
) -> int | None:
    """The room's model's window, as its server's '/v1/models' says

    For an OpenAI-compatible server which reports 'max_model_len' (vLLM):
    the entry whose 'id' is the room's 'agent.model_name', at the room's
    'agent.provider_base_url' (with or without its '/v1').  None if the
    room names no such server, the server cannot be reached, or it does
    not say.  Contacts the model server directly, so only on request.
    """
    agent = room_info.get("agent") or {}
    base_url = agent.get("provider_base_url")
    model_name = agent.get("model_name")

    if not base_url or not model_name:
        return None

    url = f"{base_url.rstrip('/').removesuffix('/v1')}/v1/models"

    try:
        response = http.get(url)
        response.raise_for_status()
        models = response.json()["data"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None

    if not isinstance(models, list):
        return None

    for model in models:
        if isinstance(model, dict) and model.get("id") == model_name:
            window = _tokens(model.get("max_model_len"))
            return window or None  # a window of 0 tokens is no window

    return None


def resolve_window(
    room_info: dict,
    *,
    context_window: int | None = None,
    probe: abc.Callable[[dict], int | None] | None = None,
) -> tuple[int | None, str | None]:
    """The model's context window in tokens, and where it came from

    In order:  'context_window' (an option);  the room's declared window;
    'probe(room_info)' (e.g. 'probe_model_window'), if given;  else
    unknown, '(None, None)'.  Never a default:  a guess (such as
    pydantic-ai-harness's 200k) is badly wrong for small local models.
    """
    if context_window is not None:
        return context_window, WINDOW_FROM_OPTION

    window = room_context_window(room_info)

    if window is not None:
        return window, WINDOW_FROM_ROOM

    if probe is not None:
        window = probe(room_info)

        if window is not None:
            return window, WINDOW_FROM_MODEL_SERVER

    return None, None


def _tokens(value) -> int | None:
    """'value' if it is a whole number of tokens (not a bool), else None"""
    return value if type(value) is int and value >= 0 else None


def measured_tokens(usage: dict | None) -> int | None:
    """The tokens a run left in the thread, from its usage record

    'final_input_tokens + final_output_tokens':  the last request's input,
    plus the reply it made, which the next request carries.  (Never the
    cumulative 'input_tokens', which counts every request of the run.)
    None if the run recorded no usage, or not those, or not as numbers:
    a record which cannot be read reports nothing.  (A missing, or null,
    'final_output_tokens' counts as none.)
    """
    if not isinstance(usage, dict):
        return None

    final_input = _tokens(usage.get("final_input_tokens"))
    final_output = usage.get("final_output_tokens")

    if final_output is not None:
        final_output = _tokens(final_output)

        if final_output is None:
            return None

    if final_input is None:
        return None

    return final_input + (final_output or 0)


class InvalidContextBudget(ValueError):
    def __init__(self, window, reserve):
        super().__init__(
            f"The context window ({window} tokens) must be larger than the "
            f"output reserve ({reserve} tokens)",
        )


@dataclasses.dataclass
class ContextBudget:
    """How much of the model's context window the next request takes

    'window_tokens' (and 'window_source':  see 'resolve_window') is the
    window, None if unknown;  'output_reserve' tokens of it are kept for
    the reply.

    The estimate is anchored on what the server measured:  after each
    run, 'anchor' records the run's 'measured_tokens' against the size of
    the history the run left.  The next request is estimated as that,
    plus the characters added since (or less those compaction saved) at
    'chars_per_token'.  Before any measurement it is the history's size
    at 'chars_per_token', which leaves out the system prompt and the tool
    definitions.  A run with no measurement keeps the last anchor, and
    marks it 'stale'.
    """

    window_tokens: int | None = None
    window_source: str | None = None
    output_reserve: int = DEFAULT_OUTPUT_RESERVE
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    anchor_tokens: int | None = None
    anchor_chars: int = 0
    stale: bool = False

    def __post_init__(self):
        if self.window_tokens is not None and not (
            0 <= self.output_reserve < self.window_tokens
        ):
            raise InvalidContextBudget(self.window_tokens, self.output_reserve)

    @property
    def usable_tokens(self) -> int | None:
        """The window, less the output reserve;  None if unknown"""
        if self.window_tokens is None:
            return None

        return self.window_tokens - self.output_reserve

    def estimate(self, chars: int) -> int:
        """Tokens a request carrying a history of 'chars' characters takes"""
        if self.anchor_tokens is None:
            return round(chars / self.chars_per_token)

        added = (chars - self.anchor_chars) / self.chars_per_token
        return max(round(self.anchor_tokens + added), 0)

    def anchor(self, usage: dict | None, chars: int) -> None:
        """Anchor on a run's usage, its history 'chars' characters long"""
        tokens = measured_tokens(usage)

        if tokens is None:
            self.stale = self.anchor_tokens is not None
            return

        self.anchor_tokens = tokens
        self.anchor_chars = chars
        self.stale = False

    def fraction(self, tokens: int) -> float | None:
        """'tokens' as a fraction of the whole window;  None if unknown

        (What a context meter shows.  Compaction's marks are fractions of
        'usable_tokens' instead.)
        """
        if self.window_tokens is None:
            return None

        return tokens / self.window_tokens


@dataclasses.dataclass(frozen=True)
class ResendReport:
    """What 'Harness.before_post' did to one history, and its size

    'resend_chars' ('wire_chars') and 'state_chars' are the sizes of the
    history's messages and state as sent;  'compacted' how many results
    this POST compacted (saving 'compacted_chars'), and
    'compacted_total' how many of the history's results are compacted.
    'est_tokens' is the request's estimated size, of 'window_tokens'
    (from 'window_source'), 'stale' if the last run measured nothing.
    'answered' is how many calls left without a result were given one
    (see 'answer_unanswered');  'trimmed' the state namespaces whose RAG
    working evidence was dropped (see 'trim_rag_state').
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
    est_tokens: int
    window_tokens: int | None
    window_source: str | None
    stale: bool
    answered: int = 0
    trimmed: list[str] = dataclasses.field(default_factory=list)

    def as_json(self) -> dict:
        return dataclasses.asdict(self)


#
#   The harness
#
@dataclasses.dataclass
class Harness:
    """What a client does to one thread's history before each POST

    Pass 'before_post' and 'after_run' to 'client_tools.run_loop'.
    'before_post' compacts the history as 'compaction' says, measured
    against 'budget' (which 'after_run' anchors on each run's usage);
    then, with 'pairing_check', a prompt's first POST answers any call
    left without a result ('answer_unanswered'), and a history
    'validate_pairing' refuses is never sent:  'run_loop' raises
    'InconsistentHistory' instead.  The state's RAG working evidence is
    trimmed as 'trim_rag_state' says:  'boundary' at a prompt's first
    POST, 'aggressive' at every POST.  Each
    POST's 'ResendReport' is appended to 'reports', and passed to
    'on_report' (e.g., for a UI to show).
    """

    pairing_check: bool = True
    compaction: CompactionPolicy = CompactionPolicy()
    trim_rag_state: str = TRIM_BOUNDARY
    budget: ContextBudget = dataclasses.field(default_factory=ContextBudget)
    on_report: abc.Callable[[ResendReport], None] | None = None
    reports: list[ResendReport] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.trim_rag_state not in TRIM_MODES:
            raise InvalidTrimMode(self.trim_rag_state)

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
        answered = 0

        if self.pairing_check and first:
            messages, answered = answer_unanswered(messages)

            if answered:
                run_input = run_input.model_copy(update={"messages": messages})

        trimmed = []

        if first or self.trim_rag_state == TRIM_AGGRESSIVE:
            state, trimmed = trim_rag_state(
                run_input.state, self.trim_rag_state
            )

            if trimmed:
                run_input = run_input.model_copy(update={"state": state})

        chars = wire_chars(messages)
        indexes = self._to_compact(messages, chars)
        saved = 0

        if indexes:
            messages = compact_history(messages, indexes)
            run_input = run_input.model_copy(update={"messages": messages})
            saved = chars - wire_chars(messages)
            chars -= saved

        if self.pairing_check:
            validate_pairing(messages)

        report = ResendReport(
            thread_id=run_input.thread_id,
            run_id=run_input.run_id,
            parent_run_id=run_input.parent_run_id,
            first=first,
            mode=self.compaction.mode,
            messages=len(messages),
            resend_chars=chars,
            state_chars=len(json.dumps(run_input.state, ensure_ascii=False)),
            compacted=len(indexes),
            compacted_chars=saved,
            compacted_total=sum(
                1
                for message in messages
                if isinstance(message, agui_core.ToolMessage)
                and is_compacted(message.content)
            ),
            est_tokens=self.budget.estimate(chars),
            window_tokens=self.budget.window_tokens,
            window_source=self.budget.window_source,
            stale=self.budget.stale,
            answered=answered,
            trimmed=trimmed,
        )
        self.reports.append(report)

        if self.on_report is not None:
            self.on_report(report)

        return run_input

    def _to_compact(
        self,
        messages: abc.Sequence[agui_core.Message],
        chars: int,
    ) -> list[int]:
        """The results to compact now, per 'compaction' and 'budget'"""
        policy = self.compaction

        if policy.mode == COMPACTION_OFF:
            return []

        candidates = compaction_candidates(messages, policy)

        if policy.mode == COMPACTION_ALWAYS:
            return candidates

        usable = self.budget.usable_tokens
        estimate = self.budget.estimate(chars)

        if not usable or estimate <= policy.trigger_fraction * usable:
            return []

        # One batch, oldest first, down to the low-water mark:  the
        # history then only grows again until the next one.
        target = policy.target_fraction * usable
        calls = _calls_by_id(messages)
        chosen = []

        for index in candidates:
            if estimate <= target:
                break

            message = messages[index]
            tool = calls[message.tool_call_id].function.name
            compacted = compact_content(tool, message.content)
            # As 'wire_chars' counts it:  as JSON.
            saved = len(json.dumps(message.content, ensure_ascii=False)) - len(
                json.dumps(compacted, ensure_ascii=False),
            )
            estimate -= saved / self.budget.chars_per_token
            chosen.append(index)

        return chosen

    def after_run(
        self,
        client: client_tools.SoliplexClient,
        run_input: agui_core.RunAgentInput,
    ) -> None:
        """Anchor the budget on the run's usage, as the server measured it

        A usage which cannot be had (an HTTP error, a body which is not
        JSON, or not a usage record) leaves the anchor as it was, 'stale':
        the run itself is not failed for it.
        """
        try:
            usage = client.run_usage(run_input.thread_id, run_input.run_id)
        except (client_tools.ClientToolsError, ValueError):
            usage = None

        self.budget.anchor(usage, wire_chars(run_input.messages))


PROBE_TIMEOUT = httpx.Timeout(5.0)


def make_harness(
    room_info: dict,
    *,
    context_window: int | None = None,
    probe_window: bool = False,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    **options,
) -> Harness:
    """A 'Harness' for a thread in the room 'room_info' describes

    Its budget's window is 'resolve_window''s:  'context_window' if given,
    else the room's, else (only if 'probe_window') what the model's server
    says ('probe_model_window').  'options' are the other 'Harness' fields.
    """

    def probe(info: dict) -> int | None:
        with httpx.Client(timeout=PROBE_TIMEOUT) as http:
            return probe_model_window(info, http)

    window, source = resolve_window(
        room_info,
        context_window=context_window,
        probe=probe if probe_window else None,
    )
    budget = ContextBudget(
        window_tokens=window,
        window_source=source,
        output_reserve=output_reserve,
    )
    return Harness(budget=budget, **options)
