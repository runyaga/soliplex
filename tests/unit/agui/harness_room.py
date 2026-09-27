"""Scripted rooms for the context-harness end-to-end test

Used by 'test_harness_e2e.py' through 'kind: factory' room configs.  No
LLM, and no network:  a pydantic-ai 'FunctionModel' decides from the
history it is handed, as the server rebuilt it from what the client sent.

'chain_agent_factory' -- one long question, answered through a chain of
runs.  With 'r' results of 'big_search' in the history:

- 'r < TURNS':  call 'big_search' (which the server runs:  a result in
  haiku.rag's exact shape, about 10.6 KB) AND the client's 'shell' (so
  the run ends with that call pending, and the client starts the next);
- else answer 'DONE r=<r> pairs=<p> sha=<hash of every call id>'.

'questions_agent_factory' -- many short questions, each in its own
prompt:  'rag_search' (a server tool keeping haiku.rag-shaped working
evidence in the AG-UI state, which it sends back as a 'STATE_SNAPSHOT'),
then 'close_question' (marking the question finished, as haiku.rag does
when a run ends with an answer), then an answer.

On every request, either model:

- checks that every tool call has exactly one result after it, and
  answers 'PAIRING_FAIL <id>' if not;
- appends a record of what it was handed to 'REQUESTS';
- simulates its context window:  a history longer than 'WINDOW_CHARS'
  fails the request as vLLM does ('Model token limit exceeded ...');
- reports the input tokens a real model would ('MeteredFunctionModel'),
  so that the server's run usage -- which the harness anchors its
  budget on -- means something.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import random

import pydantic_ai
from ag_ui import core as agui_core
from pydantic_ai import exceptions as ai_exceptions
from pydantic_ai import messages as ai_messages
from pydantic_ai import usage as ai_usage
from pydantic_ai.models import function as ai_function

TURNS = 20
HITS = 5
HIT_BODY_CHARS = 2000
WINDOW_CHARS = 10**9
OVERFLOW_MESSAGE = (
    "Model token limit exceeded before any response was generated"
)

#   What a token of history is, in characters, for 'MeteredFunctionModel'.
CHARS_PER_TOKEN = 3.5

REQUESTS: list[dict] = []


def reset(window_chars: int = 10**9) -> None:
    """Forget what the models saw, and set the simulated window"""
    global WINDOW_CHARS
    WINDOW_CHARS = window_chars
    REQUESTS.clear()


def _filler(seed: str, size: int) -> str:
    """Deterministic text of 'size' characters:  about 3.5 a word"""
    words = random.Random(seed).choices(["ab", "cde", "fghij"], k=size)
    return " ".join(words)[:size]


def search_result(turn: int) -> str:
    """A haiku.rag search result:  'HITS' context-expanded hits"""
    return "\n\n---\n\n".join(
        f"[chunk-{turn}-{k}] [rank {k} of {HITS}]\n"
        f'Source: "Doc {turn}" > Sec {k}\n'
        "Type: paragraph\n"
        "Content:\n"
        f"{_filler(f'{turn}-{k}', HIT_BODY_CHARS)}"
        for k in range(1, HITS + 1)
    )


def _parts(messages, part_type):
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, part_type)
    ]


def _unpaired(messages) -> str | None:
    """The first call without exactly one later result, if any"""
    returns = [
        (index, part.tool_call_id)
        for index, message in enumerate(messages)
        for part in message.parts
        if isinstance(part, ai_messages.ToolReturnPart)
    ]

    for index, message in enumerate(messages):
        if not isinstance(message, ai_messages.ModelResponse):
            continue

        for part in message.parts:
            if isinstance(part, ai_messages.ToolCallPart):
                later = [
                    call_id
                    for at, call_id in returns
                    if at > index and call_id == part.tool_call_id
                ]
                if len(later) != 1:
                    return part.tool_call_id

    return None


def _history_chars(messages) -> int:
    return len(ai_messages.ModelMessagesTypeAdapter.dump_json(messages))


def _record(messages, model: str) -> int:
    chars = _history_chars(messages)
    REQUESTS.append(
        {
            "model": model,
            "n_messages": len(messages),
            "history_chars": chars,
            "returns": len(_parts(messages, ai_messages.ToolReturnPart)),
        },
    )
    return chars


class MeteredFunctionModel(ai_function.FunctionModel):
    """A 'FunctionModel' reporting input tokens as a real model would

    A streamed 'FunctionModel' reports a fixed guess (50 input tokens),
    whatever it is handed;  this one reports a token for every
    'CHARS_PER_TOKEN' characters of the history.
    """

    @contextlib.asynccontextmanager
    async def request_stream(self, messages, *args, **kwargs):
        async with super().request_stream(messages, *args, **kwargs) as got:
            got._usage = ai_usage.RequestUsage(
                input_tokens=round(_history_chars(messages) / CHARS_PER_TOKEN),
            )
            yield got


def _check_window(chars: int) -> None:
    if chars > WINDOW_CHARS:
        raise ai_exceptions.ModelHTTPError(
            400,
            "harness-room",
            body={"message": OVERFLOW_MESSAGE},
        )


#
#   One long question:  a chain of client tool calls
#
def big_search(
    ctx: pydantic_ai.RunContext,
    query: str,
) -> ai_messages.ToolReturn:
    """Search the knowledge base (scripted)

    Keeps haiku.rag-shaped working evidence for the question, which stays
    open through the whole chain.
    """
    turn = int(query.rsplit("-", 1)[-1])
    rag = _rag_state(ctx)
    rag["evidence"] = rag["evidence"] | {"question": 1, "in_progress": True}
    rag["searches"][query] = [
        {"chunk_id": f"chunk-{turn}-{k}"} for k in (1, 2)
    ]
    rag["citation_index"][f"chunk-{turn}-1"] = {"turn": turn}
    return _snapshot(ctx, search_result(turn))


async def _chain_stream(messages, info: ai_function.AgentInfo):
    _check_window(_record(messages, "chain"))

    unpaired = _unpaired(messages)

    if unpaired is not None:
        yield f"PAIRING_FAIL {unpaired}"
        return

    searched = [
        part
        for part in _parts(messages, ai_messages.ToolReturnPart)
        if part.tool_name == "big_search"
    ]
    turn = len(searched)

    if turn < TURNS:
        yield {
            0: ai_function.DeltaToolCall(
                name="big_search",
                json_args=f'{{"query": "q-{turn}"}}',
                tool_call_id=f"s-{turn}",
            ),
            1: ai_function.DeltaToolCall(
                name="shell",
                json_args=f'{{"command": "echo {turn}"}}',
                tool_call_id=f"c-{turn}",
            ),
        }
        return

    call_ids = [
        part.tool_call_id
        for part in _parts(messages, ai_messages.ToolCallPart)
    ]
    digest = hashlib.sha256(" ".join(call_ids).encode()).hexdigest()[:16]
    yield f"DONE r={turn} pairs={len(call_ids)} sha={digest}"


def chain_agent_factory(**_kwargs) -> pydantic_ai.Agent:
    return pydantic_ai.Agent(
        model=MeteredFunctionModel(stream_function=_chain_stream),
        tools=[big_search],
        name="harness-chain",
    )


#
#   Many short questions:  RAG working evidence in the state
#
RAG_NAMESPACE = "rag"


def _rag_state(ctx) -> dict:
    state = ctx.deps.state
    rag = state.setdefault(RAG_NAMESPACE, {})
    rag.setdefault("citation_index", {})
    rag.setdefault("citations", [])
    rag.setdefault("evidence", {"question": 0, "in_progress": False})
    rag.setdefault("searches", {})
    rag.setdefault("executions", [])
    return rag


def _snapshot(ctx, text: str) -> ai_messages.ToolReturn:
    return ai_messages.ToolReturn(
        return_value=text,
        metadata=[
            agui_core.StateSnapshotEvent(
                type=agui_core.EventType.STATE_SNAPSHOT,
                snapshot=copy.deepcopy(ctx.deps.state),
            ),
        ],
    )


def rag_search(
    ctx: pydantic_ai.RunContext, query: str
) -> ai_messages.ToolReturn:
    """Search, keeping the expanded results as working evidence"""
    rag = _rag_state(ctx)

    if not rag["evidence"]["in_progress"]:
        # A new question:  what haiku.rag's 'begin_invocation' drops.
        rag["citations"] = []
        rag["searches"] = {}
        rag["executions"] = []
        rag["evidence"] = {
            "question": rag["evidence"]["question"] + 1,
            "in_progress": True,
        }

    rag["searches"][query] = [{"content": _filler(query, 5000)}]
    rag["citation_index"][query] = {"chunk_id": query}
    rag["citations"].append(query)
    return _snapshot(ctx, f"found {query}")


def close_question(ctx: pydantic_ai.RunContext) -> ai_messages.ToolReturn:
    """Mark the question answered"""
    rag = _rag_state(ctx)
    rag["evidence"] = rag["evidence"] | {"in_progress": False}
    return _snapshot(ctx, "closed")


async def _questions_stream(messages, info: ai_function.AgentInfo):
    _check_window(_record(messages, "questions"))

    unpaired = _unpaired(messages)

    if unpaired is not None:
        yield f"PAIRING_FAIL {unpaired}"
        return

    prompts = _parts(messages, ai_messages.UserPromptPart)
    question = len(prompts)
    last = messages[-1].parts[-1]

    if isinstance(last, ai_messages.UserPromptPart):
        yield {
            0: ai_function.DeltaToolCall(
                name="rag_search",
                json_args=f'{{"query": "question-{question}"}}',
                tool_call_id=f"rs-{question}",
            ),
        }
    elif last.tool_name == "rag_search":
        yield {
            0: ai_function.DeltaToolCall(
                name="close_question",
                json_args="{}",
                tool_call_id=f"cq-{question}",
            ),
        }
    else:
        yield f"ANSWER {question}"


def questions_agent_factory(**_kwargs) -> pydantic_ai.Agent:
    return pydantic_ai.Agent(
        model=MeteredFunctionModel(stream_function=_questions_stream),
        tools=[rag_search, close_question],
        name="harness-questions",
    )
