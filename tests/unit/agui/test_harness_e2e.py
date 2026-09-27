"""The context harness, end to end:  scripted rooms over real HTTP

A real Soliplex app (a 'kind: factory' installation) is served by uvicorn
on a loopback socket, and 'client_tools.run_loop' drives it through
'SoliplexClient', as 'soliplex-cli ask --url' and the TUI do.  No LLM and
no network:  the rooms' models are 'FunctionModel's
('tests/unit/agui/harness_room.py'), and nothing but 127.0.0.1 is used.

The chain room answers one question through 20 runs, each adding a 10.6
KB server-side search result and a client 'shell' result to the history.
Three arms:

- 'off_unlimited':  compaction off, an unlimited model window:  the
  history grows past twice the budget, and the answer arrives;
- 'off_limited':  compaction off, a 150,000-character window:  the run
  fails as vLLM does ('Model token limit exceeded'), promptly;
- 'on_limited':  compaction 'auto', the same window:  the history stays
  under the budget, rarely compacted, and the same answer arrives.

The questions room answers 20 prompts, each keeping haiku.rag-shaped
working evidence in the state, to check what boundary trimming resends.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import json
import socket
import threading
import time
from unittest import mock

import httpx
import pydantic_ai.models
import pytest
import uvicorn
from ag_ui import core as agui_core
from pydantic_ai import messages as ai_messages
from pydantic_ai.ui import ag_ui as ai_ag_ui

from soliplex import main
from soliplex.agui import client_tools
from soliplex.agui import harness
from soliplex.config import routing as config_routing
from tests._dburi import sqlite_dburi
from tests.unit.agui import harness_room

CHAIN_ROOM = "harness-chain"
QUESTIONS_ROOM = "harness-questions"

#   The client's budget for what it resends, in 'wire_chars'.
BUDGET_CHARS = 100_000
#   The simulated model window, in characters of pydantic-ai's history.
WINDOW_CHARS = 150_000
#   The window the 'on' arm's harness is told, in tokens:  its trigger
#   (0.70 of it, less the reserve) comes well under the budget.
CONTEXT_WINDOW = 40_000
#   Compaction batches over the 21 POSTs of a chain, at most.
MAX_COMPACTION_EVENTS = 6

#   A read timeout;  but the server sends a keepalive every 15 seconds,
#   so a stream which never ends would never time out:  every client also
#   has a deadline ('Deadline'), 'WATCHDOG_SECS' after it is made.
HTTP_TIMEOUT = httpx.Timeout(10.0, read=20.0)
WATCHDOG_SECS = 180.0


class DeadlinePassed(httpx.ReadTimeout):
    """A response still being read when its client's deadline passed"""


class Deadline(httpx.HTTPTransport):
    """A transport whose responses fail once 'secs' have passed

    Checked on every chunk read, keepalives included:  a watchdog which a
    stream of keepalives cannot outrun.  Each time it stops a response,
    it records when in 'fired'.
    """

    def __init__(self, secs: float = WATCHDOG_SECS):
        super().__init__(trust_env=False)
        self.started = time.monotonic()
        self.deadline = self.started + secs
        self.fired: list[float] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        response.stream = _Guarded(self, request, response.stream)
        return response


class _Guarded(httpx.SyncByteStream):
    def __init__(self, deadline: Deadline, request, stream):
        self._deadline = deadline
        self._request = request
        self._stream = stream

    def __iter__(self):
        for chunk in self._stream:
            if time.monotonic() > self._deadline.deadline:
                self._deadline.fired.append(
                    time.monotonic() - self._deadline.started,
                )
                raise DeadlinePassed("watchdog", request=self._request)

            yield chunk

    def close(self) -> None:
        self._stream.close()


def _http(secs: float = WATCHDOG_SECS, **kwargs) -> httpx.Client:
    """A client for the fixture's server only:  no proxy from the env,
    and a deadline ('Deadline';  its transport is the client's
    '_transport')
    """
    return httpx.Client(
        timeout=HTTP_TIMEOUT,
        trust_env=False,
        transport=Deadline(secs),
        **kwargs,
    )


def _fired(http: httpx.Client) -> list[float]:
    return http._transport.fired


INSTALLATION_YAML = """\
id: "harness-e2e"
secrets:
  - secret_name: "URL_SAFE_TOKEN_SECRET"
    sources:
      - kind: "random_chars"
  - secret_name: "SESSION_MIDDLEWARE_TOKEN"
    sources:
      - kind: "random_chars"
environment:
  - name: "INSTALLATION_PATH"
    value: "file:."
thread_persistence_db:
  sync_dburi: "{agui_sync}"
  async_dburi: "{agui_async}"
authorization_db:
  sync_dburi: "{authz_sync}"
  async_dburi: "{authz_async}"
filesystem_skills_paths: []
room_paths:
  - "./rooms/{chain}"
  - "./rooms/{questions}"
oidc_paths: []
completion_paths: []
quizzes_paths: []
"""

ROOM_YAML = """\
id: "{room_id}"
name: "{room_id}"
description: "Scripted, for the context-harness end-to-end test"
agent:
  kind: "factory"
  factory_name: "tests.unit.agui.harness_room.{factory}"
allow_mcp: false
"""


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    """A real server for the scripted rooms, on a loopback socket

    Module-scoped:  the arms run once, and every test checks them.
    """
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        mock.patch.dict(config_routing.__dict__) as patched_routing,
    ):
        patched_routing["APP_ROUTERS_BY_GROUP_NAME"] = {}

        for name in ("OPENAI_API_KEY", "OLLAMA_BASE_URL", "LOGFIRE_TOKEN"):
            monkeypatch.delenv(name, raising=False)

        # No Logfire credentials from disk either:  nothing is sent.
        monkeypatch.setenv(
            "LOGFIRE_CREDENTIALS_DIR",
            str(tmp_path_factory.mktemp("no-logfire")),
        )

        monkeypatch.setattr(
            pydantic_ai.models,
            "ALLOW_MODEL_REQUESTS",
            False,
        )
        yield from _serve(tmp_path_factory.mktemp("harness-e2e"))


def _serve(tmp_path):
    install = tmp_path / "install"

    for room_id, factory in [
        (CHAIN_ROOM, "chain_agent_factory"),
        (QUESTIONS_ROOM, "questions_agent_factory"),
    ]:
        room_dir = install / "rooms" / room_id
        room_dir.mkdir(parents=True)
        (room_dir / "room_config.yaml").write_text(
            ROOM_YAML.format(room_id=room_id, factory=factory),
            encoding="utf-8",
        )

    agui_db = tmp_path / "agui.sqlite"
    authz_db = tmp_path / "authz.sqlite"
    (install / "installation.yaml").write_text(
        INSTALLATION_YAML.format(
            agui_sync=sqlite_dburi(agui_db),
            agui_async=sqlite_dburi(agui_db, "+aiosqlite"),
            authz_sync=sqlite_dburi(authz_db),
            authz_async=sqlite_dburi(authz_db, "+aiosqlite"),
            chain=CHAIN_ROOM,
            questions=QUESTIONS_ROOM,
        ),
        encoding="utf-8",
    )

    config_routing.register_default_routers()
    app = main.create_app(install, no_auth_mode=True)
    config_routing.add_registered_routers(app)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="warning",
            ws="none",
            timeout_graceful_shutdown=2,
        ),
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [sock]},
        daemon=True,
    )
    thread.start()

    try:
        deadline = time.monotonic() + 60.0

        while not server.started:
            assert thread.is_alive(), "the server failed to start"
            assert time.monotonic() < deadline, "the server did not start"
            time.sleep(0.05)

        yield f"http://{host}:{port}"

    finally:
        server.should_exit = True
        thread.join(timeout=15.0)
        sock.close()

    assert not thread.is_alive(), "the server did not stop"


#
#   A client which records what it sends, and what it is answered
#
@dataclasses.dataclass
class Recorder:
    """Every POST's input, as sent, and every HTTP status"""

    posts: list[agui_core.RunAgentInput] = dataclasses.field(
        default_factory=list,
    )
    statuses: list[tuple[str, str, int]] = dataclasses.field(
        default_factory=list,
    )
    http: httpx.Client | None = None

    def client(self, url: str, room_id: str) -> client_tools.SoliplexClient:
        def on_response(response: httpx.Response) -> None:
            request = response.request
            self.statuses.append(
                (request.method, request.url.path, response.status_code),
            )

        http = _http(event_hooks={"response": [on_response]})
        self.http = http
        client = client_tools.SoliplexClient(url, room_id, http=http)
        stream_run = client.stream_run

        def capturing(run_input):
            # After 'before_post':  exactly what goes over the wire.
            self.posts.append(run_input.model_copy(deep=True))
            return stream_run(run_input)

        client.stream_run = capturing
        return client


def fake_shell(args: dict, context: client_tools.ToolContext) -> dict:
    """The client's 'shell', without a subprocess:  a small result"""
    return {
        "stdout": f"ok-{args['command']}",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
    }


FAKE_SHELL = dataclasses.replace(client_tools.SHELL_TOOL, execute=fake_shell)
TOOLS = {FAKE_SHELL.name: FAKE_SHELL}


@dataclasses.dataclass
class Arm:
    url: str
    room_id: str
    thread_id: str
    result: client_tools.LoopResult
    error: client_tools.ClientToolsError | None
    recorder: Recorder
    harness: harness.Harness
    model_requests: list[dict]
    elapsed: float
    anchors: list[int | None] = dataclasses.field(default_factory=list)

    @property
    def posts(self) -> list[agui_core.RunAgentInput]:
        return self.recorder.posts

    @property
    def agui_url(self) -> str:
        return f"{self.url}/api/v1/rooms/{self.room_id}/agui"

    def get(self, path: str = "") -> dict:
        """'GET .../agui/{thread_id}{path}', from the server's REST API"""
        with _http() as http:
            response = http.get(f"{self.agui_url}/{self.thread_id}{path}")

        assert response.status_code == 200
        return response.json()

    def finished_runs(self, deadline_secs: float = 10.0) -> dict:
        """The thread's runs, once all of this arm's are marked finished

        The stream ends a moment before the server stores that its run
        finished:  poll, within a deadline.
        """
        deadline = time.monotonic() + deadline_secs

        while True:
            runs = self.get()["runs"]

            if all(runs[run_id]["finished"] for run_id in self.result.run_ids):
                return runs

            assert time.monotonic() < deadline, "runs never finished"
            time.sleep(0.05)


def run_chain(
    url: str,
    root,
    the_harness: harness.Harness,
    *,
    window_chars: int,
) -> Arm:
    harness_room.reset(window_chars)
    recorder = Recorder()
    context = client_tools.ToolContext(root=root)
    anchors = []

    def after_run(client, run_input):
        the_harness.after_run(client, run_input)
        anchors.append(the_harness.budget.anchor_tokens)

    with recorder.client(url, CHAIN_ROOM) as client:
        thread = client.new_thread()
        run_input = client_tools.initial_run_input(thread, "chain", TOOLS)
        started = time.monotonic()
        error = None

        try:
            result = client_tools.run_loop(
                client,
                run_input,
                context,
                tools=TOOLS,
                max_turns=harness_room.TURNS + 5,
                before_post=the_harness.before_post,
                after_run=after_run,
            )
        except client_tools.ClientToolsError as exc:
            error, result = exc, exc.result

        elapsed = time.monotonic() - started

    assert _fired(recorder.http) == [], "the chain hung:  its deadline passed"

    return Arm(
        url=url,
        room_id=CHAIN_ROOM,
        thread_id=thread["thread_id"],
        result=result,
        error=error,
        recorder=recorder,
        harness=the_harness,
        model_requests=list(harness_room.REQUESTS),
        elapsed=elapsed,
        anchors=anchors,
    )


def _off() -> harness.Harness:
    return harness.Harness(
        compaction=harness.CompactionPolicy(mode=harness.COMPACTION_OFF),
    )


def _always() -> harness.Harness:
    return harness.Harness(
        compaction=harness.CompactionPolicy(
            mode=harness.COMPACTION_ALWAYS,
            keep_recent=4,
        ),
    )


def _on() -> harness.Harness:
    return harness.make_harness(
        {},
        context_window=CONTEXT_WINDOW,
        compaction=harness.CompactionPolicy(
            mode=harness.COMPACTION_AUTO,
            keep_recent=4,
        ),
    )


@pytest.fixture(scope="module")
def arms(server_url, tmp_path_factory):
    root = tmp_path_factory.mktemp("root")

    return {
        "off_unlimited": run_chain(
            server_url,
            root,
            _off(),
            window_chars=10**9,
        ),
        "off_limited": run_chain(
            server_url,
            root,
            _off(),
            window_chars=WINDOW_CHARS,
        ),
        "on_limited": run_chain(
            server_url,
            root,
            _on(),
            window_chars=WINDOW_CHARS,
        ),
        "always_limited": run_chain(
            server_url,
            root,
            _always(),
            window_chars=WINDOW_CHARS,
        ),
    }


#
#   Oracles
#
def _calls(messages):
    return [
        call.id
        for message in messages
        if isinstance(message, agui_core.AssistantMessage)
        for call in message.tool_calls or ()
    ]


def _results_follow_calls(messages) -> bool:
    seen = set()

    for message in messages:
        if isinstance(message, agui_core.AssistantMessage):
            seen.update(call.id for call in message.tool_calls or ())
        elif isinstance(message, agui_core.ToolMessage):
            if message.tool_call_id not in seen:
                return False

    return True


def _provider_valid(model_messages) -> bool:
    """Every call is answered once, later;  no result answers nothing

    (After pydantic-ai-harness's 'is_provider_valid':  what a provider
    requires of a history.)
    """
    calls = []
    returns = collections.Counter()

    for message in model_messages:
        for part in message.parts:
            if isinstance(part, ai_messages.ToolCallPart):
                if returns[part.tool_call_id]:
                    return False  # answered before it was made
                calls.append(part.tool_call_id)
            elif isinstance(part, ai_messages.ToolReturnPart):
                returns[part.tool_call_id] += 1

    return collections.Counter(calls) == returns and all(
        count == 1 for count in returns.values()
    )


def _assert_pairing(sent: agui_core.RunAgentInput) -> None:
    messages = sent.messages
    call_ids = _calls(messages)
    result_ids = [
        message.tool_call_id
        for message in messages
        if isinstance(message, agui_core.ToolMessage)
    ]

    assert len(call_ids) == len(set(call_ids))
    assert collections.Counter(result_ids) == collections.Counter(call_ids)
    assert _results_follow_calls(messages)
    assert client_tools.pending_tool_calls(messages) == []
    assert harness.pairing_problems(messages) == []
    # What the server does with it:  raises on an orphan result.
    model_messages = ai_ag_ui.AGUIAdapter.load_messages(messages)
    assert _provider_valid(model_messages)


def _search_results(sent: agui_core.RunAgentInput):
    """The 'big_search' results, oldest first"""
    calls = {
        call.id: call.function.name
        for message in sent.messages
        if isinstance(message, agui_core.AssistantMessage)
        for call in message.tool_calls or ()
    }
    return [
        message
        for message in sent.messages
        if isinstance(message, agui_core.ToolMessage)
        and calls[message.tool_call_id] == "big_search"
    ]


def _message_bytes(sent: agui_core.RunAgentInput) -> list[str]:
    return [message.model_dump_json() for message in sent.messages]


#
#   The chain:  budget, answer, overflow
#
def test_chain_answers_under_budget_with_compaction(arms):
    on, off = arms["on_limited"], arms["off_unlimited"]

    for arm in (on, off):
        assert arm.error is None
        assert arm.result.response.startswith(
            f"DONE r={harness_room.TURNS} pairs={2 * harness_room.TURNS}",
        )
        assert len(arm.result.run_ids) == harness_room.TURNS + 1
        assert len(arm.posts) == harness_room.TURNS + 1

    # The same calls, in the same order:  compaction changed no answer.
    assert (
        on.result.response.split("sha=")[1]
        == off.result.response.split("sha=")[1]
    )

    # Under the budget with compaction;  over twice it without.
    on_sizes = [harness.wire_chars(sent.messages) for sent in on.posts]
    off_sizes = [harness.wire_chars(sent.messages) for sent in off.posts]
    assert max(on_sizes) <= BUDGET_CHARS
    assert off_sizes[-1] > 2 * BUDGET_CHARS
    # ... and what the model was handed, too.
    assert max(r["history_chars"] for r in on.model_requests) <= BUDGET_CHARS
    assert max(r["history_chars"] for r in off.model_requests) > WINDOW_CHARS

    for arm in (on, off):
        assert "PAIRING_FAIL" not in arm.result.response


def test_chain_overflows_without_compaction(arms):
    # The real failure, reproduced:  an error, not a hang.
    limited = arms["off_limited"]

    assert isinstance(limited.error, client_tools.RunErrored)
    assert "token limit exceeded" in str(limited.error)
    assert limited.elapsed < 60
    assert len(limited.result.run_ids) < harness_room.TURNS + 1


def test_chain_pairing_on_every_post(arms):
    for arm in arms.values():
        for sent in arm.posts:
            _assert_pairing(sent)


def test_chain_every_post_accepted(arms):
    # Through the real routes:  every POST validated and answered, none
    # refused (422, 400), and every run finished.
    for arm in arms.values():
        assert arm.recorder.statuses
        assert {status for _, _, status in arm.recorder.statuses} == {200}

    for arm in (arms["on_limited"], arms["off_unlimited"]):
        runs = arm.finished_runs()

        assert set(runs) >= set(arm.result.run_ids)


def _shape(sent: agui_core.RunAgentInput) -> list[tuple]:
    """Each message's role, and the call it answers or makes"""
    return [
        (
            message.role,
            getattr(message, "tool_call_id", None),
            tuple(_calls([message])),
        )
        for message in sent.messages
    ]


def test_chain_compaction_keeps_the_structure(arms):
    on, off = arms["on_limited"], arms["off_unlimited"]

    for a, b in zip(on.posts, off.posts, strict=True):
        # The same messages in the same places, the same calls answered,
        # the same state and tools:  only content differs.
        assert _shape(a) == _shape(b)
        assert len(ai_ag_ui.AGUIAdapter.load_messages(a.messages)) == len(
            ai_ag_ui.AGUIAdapter.load_messages(b.messages),
        )
        assert a.state == b.state
        assert a.tools == b.tools

    # No message is ever removed, moved, or given a new id.
    for before, after in zip(on.posts, on.posts[1:], strict=False):
        ids = [message.id for message in before.messages]
        assert [message.id for message in after.messages][: len(ids)] == ids


def test_chain_compacted_results(arms):
    on = arms["on_limited"]
    last = on.posts[-1]
    searches = _search_results(last)

    compacted = [m for m in searches if harness.is_compacted(m.content)]
    intact = [m for m in searches if not harness.is_compacted(m.content)]

    assert compacted
    # Oldest first:  the compacted are all older than the intact, and the
    # newest 4 are kept.
    assert searches[: len(compacted)] == compacted
    assert len(intact) >= 4

    for message in compacted:
        info = harness.compacted_info(message.content)
        assert info["tool"] == "big_search"
        assert info["format"] == "headers"
        assert "[chunk-" in message.content
        assert "Content:" not in message.content
        assert len(message.content) <= 1_000

    # Only large results:  no small 'shell' result was touched.
    tool_messages = [m for m in last.messages if m.role == "tool"]
    assert all(
        not harness.is_compacted(m.content)
        for m in tool_messages
        if m not in searches
    )


def test_chain_compaction_is_rare_and_byte_stable(arms):
    # KV-cache friendly:  a few batches, and between them the history
    # only grows -- what was sent is resent byte for byte.
    on = arms["on_limited"]
    reports = on.harness.reports

    assert len(reports) == len(on.posts)
    events = [
        index for index, report in enumerate(reports) if report.compacted
    ]
    assert 1 <= len(events) <= MAX_COMPACTION_EVENTS

    for index in range(1, len(on.posts)):
        before = _message_bytes(on.posts[index - 1])
        after = _message_bytes(on.posts[index])

        if index not in events:
            assert after[: len(before)] == before

    # A compacted result, once compacted, never changes again.
    first_seen = {}

    for sent in on.posts:
        for message in sent.messages:
            if message.role == "tool" and harness.is_compacted(
                message.content
            ):
                first_seen.setdefault(message.id, message.content)
                assert message.content == first_seen[message.id]


def test_chain_parent_run_ids(arms):
    # Compaction never changes 'parent_run_id', or the order of the runs.
    for arm in arms.values():
        posts = arm.posts
        assert posts[0].parent_run_id is None

        for previous, sent in zip(posts, posts[1:], strict=False):
            assert sent.parent_run_id == previous.run_id

        assert [sent.run_id for sent in posts] == arm.result.run_ids[
            : len(posts)
        ]


def test_chain_budget_follows_each_hop(arms):
    # The server's usage is read after every run of the chain:  the
    # anchor moves on with each hop, and each POST is estimated from it.
    on = arms["on_limited"]

    assert len(on.anchors) == len(on.posts)
    assert all(anchor is not None for anchor in on.anchors)
    assert len(set(on.anchors)) > len(on.anchors) // 2

    for report in on.harness.reports[1:]:
        assert report.stale is False
        assert report.window_tokens == CONTEXT_WINDOW
        assert report.window_source == harness.WINDOW_FROM_OPTION

    # The failed arm read its failed run's usage too.
    limited = arms["off_limited"]
    assert len(limited.anchors) == len(limited.posts)


#
#   What other clients see:  the server's copy
#
def test_chain_marker_persisted(arms):
    # The server stores the compacted input:  a client reading the thread
    # back sees each compacted result's marker, with its original size.
    on, off = arms["on_limited"], arms["off_unlimited"]
    last_run = on.get(f"/{on.posts[-1].run_id}")
    stored = agui_core.RunAgentInput.model_validate(last_run["run_input"])
    originals = {
        message.tool_call_id: message.content
        for message in _search_results(off.posts[-1])
    }

    compacted = [
        message
        for message in _search_results(stored)
        if harness.is_compacted(message.content)
    ]

    assert compacted
    assert _message_bytes(stored) == _message_bytes(on.posts[-1])

    for message in compacted:
        info = harness.compacted_info(message.content)
        original = originals[message.tool_call_id]
        assert int(info["original_bytes"]) == len(original.encode())

    # ... and the thread's runs, as listed, name the run each came from.
    runs = on.get()["runs"]
    assert [runs[run_id]["parent_run_id"] for run_id in on.result.run_ids] == [
        None,
        *on.result.run_ids[:-1],
    ]


def _set_fields(event: dict) -> dict:
    """An event's fields which are set:  the store keeps the null ones"""
    return {key: value for key, value in event.items() if value is not None}


def _sse_events(response: httpx.Response) -> list[dict]:
    return list(client_tools.iter_sse_json(response.iter_lines()))


def test_chain_reconnect_to_a_compacted_run(arms):
    # Reconnecting ('Last-Event-ID') to a run whose input was compacted
    # replays that run's events from the store:  no error, and no result
    # of an older (compacted) call replayed in full.
    on = arms["on_limited"]
    (index, *_) = [
        index
        for index, report in enumerate(on.harness.reports)
        if report.compacted
    ]
    sent = on.posts[index]
    compacted_ids = {
        message.tool_call_id
        for message in sent.messages
        if message.role == "tool" and harness.is_compacted(message.content)
    }
    assert compacted_ids

    stored_events = on.get(f"/{sent.run_id}")["events"]
    harness_room.reset()

    with (
        _http() as http,
        http.stream(
            "POST",
            f"{on.agui_url}/{on.thread_id}/{sent.run_id}",
            json=sent.model_dump(mode="json", by_alias=True),
            headers={
                "Accept": "text/event-stream",
                "Last-Event-ID": f"{sent.run_id}:0",
            },
        ) as response,
    ):
        assert response.status_code == 200
        events = _sse_events(response)

    assert _fired(http) == []
    # Exactly the stored events after the cursor, and the model not run
    # again.
    assert [_set_fields(event) for event in events] == [
        _set_fields(event) for event in stored_events[1:]
    ]
    assert harness_room.REQUESTS == []
    types = [event["type"] for event in events]
    assert "RUN_ERROR" not in types
    assert types[-1] == "RUN_FINISHED"
    replayed = {
        event["toolCallId"]
        for event in events
        if event["type"] == "TOOL_CALL_RESULT"
    }
    assert replayed == {f"s-{index}"}
    assert not replayed & compacted_ids

    # The stored input is still the compacted one.
    stored = on.get(f"/{sent.run_id}")["run_input"]
    assert _message_bytes(
        agui_core.RunAgentInput.model_validate(stored),
    ) == _message_bytes(sent)


def test_chain_orphan_refused_before_sending(arms, server_url, tmp_path):
    # A history missing the call a result answers is refused locally,
    # before any stream is opened;  sent raw, the server (with the
    # stream-init fix) answers 'RUN_ERROR', promptly.
    on = arms["on_limited"]
    sent = on.posts[5]
    orphaned = sent.model_copy(
        update={
            "messages": [
                message
                for message in sent.messages
                if "s-2" not in _calls([message])
            ],
        },
    )
    recorder = Recorder()

    with recorder.client(server_url, CHAIN_ROOM) as client:
        new_run = client.new_run(on.thread_id, parent_run_id=sent.run_id)
        orphaned = orphaned.model_copy(
            update={
                "run_id": new_run["run_id"],
                "parent_run_id": sent.run_id,
            },
        )

        with pytest.raises(harness.InconsistentHistory, match="'s-2'"):
            client_tools.run_loop(
                client,
                orphaned,
                client_tools.ToolContext(root=tmp_path),
                tools=TOOLS,
                before_post=harness.Harness().before_post,
            )

    assert recorder.posts == []  # no stream was opened
    assert [path for _, path, _ in recorder.statuses] == [
        f"/api/v1/rooms/{CHAIN_ROOM}/agui/{on.thread_id}",
    ]

    harness_room.reset()
    started = time.monotonic()

    with (
        _http() as http,
        http.stream(
            "POST",
            f"{on.agui_url}/{on.thread_id}/{orphaned.run_id}",
            json=orphaned.model_dump(mode="json", by_alias=True),
            headers={"Accept": "text/event-stream"},
        ) as response,
    ):
        assert response.status_code == 200
        types = [event["type"] for event in _sse_events(response)]

    assert _fired(http) == []
    assert types == ["RUN_STARTED", "RUN_ERROR"]
    assert time.monotonic() - started < 20
    assert harness_room.REQUESTS == []  # it never reached the model


#
#   Many questions:  the RAG state resent
#
QUESTIONS = 20
ONE_QUESTION_OF_EVIDENCE = 5_000


def run_questions(url: str, root, the_harness: harness.Harness) -> Arm:
    harness_room.reset()
    recorder = Recorder()
    context = client_tools.ToolContext(root=root)
    responses = []

    with recorder.client(url, QUESTIONS_ROOM) as client:
        thread = client.new_thread()
        run_input = client_tools.initial_run_input(thread, "question", {})

        for question in range(QUESTIONS):
            if question:
                new_run = client.new_run(
                    run_input.thread_id,
                    parent_run_id=run_input.run_id,
                )
                run_input = run_input.model_copy(
                    update={
                        "run_id": new_run["run_id"],
                        "parent_run_id": run_input.run_id,
                        "messages": [
                            *run_input.messages,
                            agui_core.UserMessage(
                                id=f"prompt-{question}",
                                content="question",
                            ),
                        ],
                    },
                )

            result = client_tools.run_loop(
                client,
                run_input,
                context,
                tools={},
                before_post=the_harness.before_post,
                after_run=the_harness.after_run,
            )
            responses.append(result.response)
            run_input = result.run_input

    assert _fired(recorder.http) == [], "the questions hung:  deadline passed"

    return Arm(
        url=url,
        room_id=QUESTIONS_ROOM,
        thread_id=thread["thread_id"],
        result=dataclasses.replace(result, response="\n".join(responses)),
        error=None,
        recorder=recorder,
        harness=the_harness,
        model_requests=list(harness_room.REQUESTS),
        elapsed=0.0,
    )


@pytest.fixture(scope="module")
def question_arms(server_url, tmp_path_factory):
    root = tmp_path_factory.mktemp("questions")

    return {
        trim: run_questions(
            server_url,
            root,
            harness.Harness(trim_rag_state=trim),
        )
        for trim in (harness.TRIM_BOUNDARY, harness.TRIM_OFF)
    }


def _state_chars(sent: agui_core.RunAgentInput) -> int:
    return len(json.dumps(sent.state))


def test_questions_answered(question_arms):
    for arm in question_arms.values():
        assert arm.result.response.split("\n") == [
            f"ANSWER {question}" for question in range(1, QUESTIONS + 1)
        ]
        assert len(arm.posts) == QUESTIONS
        assert {status for _, _, status in arm.recorder.statuses} == {200}

        for sent in arm.posts:
            _assert_pairing(sent)

        for previous, sent in zip(arm.posts, arm.posts[1:], strict=False):
            assert sent.parent_run_id == previous.run_id


def test_questions_state_trimmed_at_each_boundary(question_arms):
    trimmed = question_arms[harness.TRIM_BOUNDARY]
    untrimmed = question_arms[harness.TRIM_OFF]

    # Each prompt after the first:  the answered question's working
    # evidence is not resent...
    for sent, report in zip(
        trimmed.posts[1:],
        trimmed.harness.reports[1:],
        strict=True,
    ):
        rag = sent.state[harness_room.RAG_NAMESPACE]
        assert rag["searches"] == {}
        assert rag["executions"] == []
        assert rag["citations"] == []
        assert rag["evidence"]["in_progress"] is False
        assert report.trimmed == [harness_room.RAG_NAMESPACE]

    # ... where, untrimmed, it is.
    for sent in untrimmed.posts[1:]:
        rag = sent.state[harness_room.RAG_NAMESPACE]
        assert len(rag["searches"]) == 1
        assert _state_chars(sent) > ONE_QUESTION_OF_EVIDENCE

    # Bounded over 20 questions:  less than one question's evidence, and
    # growing only by the kept citation index.
    sizes = [_state_chars(sent) for sent in trimmed.posts]
    assert max(sizes) < ONE_QUESTION_OF_EVIDENCE
    assert sizes[-1] - sizes[1] < 100 * QUESTIONS

    # The server echoes the whole state back (with the current question's
    # evidence), and trimming it again finds only that to trim:  stable.
    for sent in trimmed.posts[1:]:
        again, names = harness.trim_rag_state(
            sent.state,
            harness.TRIM_BOUNDARY,
        )
        assert again is sent.state
        assert names == []

    # What was kept is what the server needs, exactly as untrimmed:  the
    # citation index and the evidence ledger (each question began anew).
    for kept, whole in zip(trimmed.posts, untrimmed.posts, strict=True):
        kept_rag = kept.state.get(harness_room.RAG_NAMESPACE, {})
        whole_rag = whole.state.get(harness_room.RAG_NAMESPACE, {})

        for field in ("citation_index", "evidence"):
            assert kept_rag.get(field) == whole_rag.get(field)

    assert [
        sent.state[harness_room.RAG_NAMESPACE]["evidence"]["question"]
        for sent in trimmed.posts[1:]
    ] == list(range(1, QUESTIONS))


#
#   The oracles themselves:  each catches what it is meant to
#
def _call_response(call_id):
    return ai_messages.ModelResponse(
        parts=[ai_messages.ToolCallPart(tool_name="t", tool_call_id=call_id)],
    )


def _return_request(call_id):
    return ai_messages.ModelRequest(
        parts=[
            ai_messages.ToolReturnPart(
                tool_name="t",
                tool_call_id=call_id,
                content="x",
            ),
        ],
    )


def test_oracle_results_follow_calls():
    call = agui_core.AssistantMessage(
        id="a",
        tool_calls=[
            agui_core.ToolCall(
                id="c",
                function=agui_core.FunctionCall(name="t", arguments="{}"),
            ),
        ],
    )
    result = agui_core.ToolMessage(id="r", tool_call_id="c", content="x")

    assert _results_follow_calls([call, result])
    assert not _results_follow_calls([result, call])


@pytest.mark.parametrize(
    "messages, expected",
    [
        ([_call_response("c"), _return_request("c")], True),
        ([_return_request("c"), _call_response("c")], False),  # too soon
        ([_call_response("c")], False),  # never answered
        (
            [_call_response("c"), _return_request("c"), _return_request("c")],
            False,  # answered twice
        ),
    ],
)
def test_oracle_provider_valid(messages, expected):
    assert _provider_valid(messages) is expected


async def _collect(stream) -> list:
    return [item async for item in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream",
    [harness_room._chain_stream, harness_room._questions_stream],
)
async def test_room_models_report_pairing_failures(stream):
    harness_room.reset()
    history = [
        ai_messages.ModelRequest(
            parts=[ai_messages.UserPromptPart(content="q")],
        ),
        _call_response("lost"),
    ]

    assert await _collect(stream(history, None)) == ["PAIRING_FAIL lost"]
    assert harness_room.REQUESTS[-1]["n_messages"] == 2


def test_room_rag_search_twice_in_a_question():
    # A second search in an open question keeps the first's evidence.
    ctx = mock.Mock(spec=["deps"])
    ctx.deps.state = {}

    harness_room.rag_search(ctx, "one")
    found = harness_room.rag_search(ctx, "two")

    rag = ctx.deps.state[harness_room.RAG_NAMESPACE]
    assert sorted(rag["searches"]) == ["one", "two"]
    assert rag["evidence"] == {"question": 1, "in_progress": True}
    (snapshot,) = found.metadata
    assert snapshot.snapshot == ctx.deps.state
    assert snapshot.snapshot is not ctx.deps.state


def test_chain_always_compacts_all_but_the_newest(arms):
    # Section E's 'always' arm:  deterministic, under budget -- but it
    # rewrites a result on nearly every POST, which is why 'auto' batches.
    always = arms["always_limited"]
    last = always.posts[-1]
    searches = _search_results(last)

    assert always.error is None
    assert always.result.response.startswith(
        f"DONE r={harness_room.TURNS} pairs={2 * harness_room.TURNS}",
    )
    compacted = [m for m in searches if harness.is_compacted(m.content)]
    assert len(compacted) == harness_room.TURNS - 4
    assert all(not harness.is_compacted(m.content) for m in searches[-4:])
    assert (
        max(harness.wire_chars(sent.messages) for sent in always.posts)
        <= BUDGET_CHARS
    )
    events = [report for report in always.harness.reports if report.compacted]
    assert len(events) > MAX_COMPACTION_EVENTS


def test_chain_open_question_state_never_trimmed(arms):
    # Through the whole chain the RAG question stays open:  boundary
    # trimming (the default) leaves its evidence alone on every POST,
    # and the server's snapshot carries it forward.
    on = arms["on_limited"]

    for turn, sent in enumerate(on.posts[1:], start=1):
        rag = sent.state[harness_room.RAG_NAMESPACE]
        assert rag["evidence"]["in_progress"] is True
        assert sorted(rag["searches"]) == sorted(f"q-{k}" for k in range(turn))
        assert len(rag["citation_index"]) == turn

    assert all(report.trimmed == [] for report in on.harness.reports)


def test_arm_finished_runs_polls(monkeypatch):
    arm = Arm(
        url="u",
        room_id="r",
        thread_id="t",
        result=client_tools.LoopResult(thread_id="t", run_ids=["a"]),
        error=None,
        recorder=Recorder(),
        harness=harness.Harness(),
        model_requests=[],
        elapsed=0.0,
    )
    answers = iter(
        [
            {"runs": {"a": {"finished": None}}},
            {"runs": {"a": {"finished": "2026-09-26T00:00:00"}}},
        ],
    )
    monkeypatch.setattr(arm, "get", lambda: next(answers))

    assert arm.finished_runs()["a"]["finished"]


def test_deadline_stops_an_endless_stream():
    # A stream which only ever sends keepalives -- which a read timeout
    # never stops -- is stopped by its client's deadline.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()
    stop = threading.Event()

    def serve():
        conn, _ = listener.accept()

        # Until told to stop, or the client goes.
        with conn, contextlib.suppress(OSError):
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n",
            )

            while not stop.wait(0.05):
                conn.sendall(b"d\r\n: keepalive\n\n\r\n")

    server = threading.Thread(target=serve, daemon=True)
    server.start()

    try:
        with (
            _http(secs=0.5) as http,
            pytest.raises(DeadlinePassed),
            http.stream("GET", f"http://{host}:{port}/") as response,
        ):
            collections.deque(response.iter_lines(), maxlen=0)  # read it all
    finally:
        stop.set()
        listener.close()
        server.join(timeout=5)

    (fired,) = _fired(http)
    assert 0.5 <= fired < 5
