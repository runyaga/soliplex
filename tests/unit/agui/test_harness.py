from __future__ import annotations

import json
from unittest import mock

import httpx
import pytest
from ag_ui import core as agui_core

from soliplex.agui import client_tools
from soliplex.agui import harness


def _call(call_id, name="search"):
    return agui_core.ToolCall(
        id=call_id,
        function=agui_core.FunctionCall(name=name, arguments="{}"),
    )


def _assistant(message_id, *call_ids, content=None):
    return agui_core.AssistantMessage(
        id=message_id,
        content=content,
        tool_calls=[_call(call_id) for call_id in call_ids] or None,
    )


def _result(message_id, call_id, content="result"):
    return agui_core.ToolMessage(
        id=message_id,
        tool_call_id=call_id,
        content=content,
    )


def _user(message_id="u1", content="hi"):
    return agui_core.UserMessage(id=message_id, content=content)


def _run_input(messages, state=None):
    return agui_core.RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state=state if state is not None else {},
        messages=messages,
        tools=[],
        context=[],
        forwarded_props={},
    )


# -- pairing -----------------------------------------------------------------


def test_pairing_problems_w_consistent_history():
    messages = [
        _user(),
        _assistant("a1", "c1", "c2"),
        _result("t1", "c1"),
        _result("t2", "c2"),
        _assistant("a2", content="Done."),
    ]

    assert harness.pairing_problems(messages) == []
    harness.validate_pairing(messages)  # does not raise


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [_user(), _assistant("a1", "c1"), _assistant("a2", "c1")],
            [
                "duplicate tool call id 'c1'",
                "tool call 'c1' (search) has no result",
            ],
        ),
        (
            [_user(), _result("t1", "c9")],
            [
                "tool result 't1' answers 'c9', which no earlier assistant "
                "message calls",
            ],
        ),
        (
            # The call comes after its result:  just as bad.
            [_user(), _result("t1", "c1"), _assistant("a1", "c1")],
            [
                "tool result 't1' answers 'c1', which no earlier assistant "
                "message calls",
            ],
        ),
        (
            [
                _user(),
                _assistant("a1", "c1"),
                _result("t1", "c1"),
                _result("t2", "c1"),
            ],
            ["tool call 'c1' has more than one result"],
        ),
        (
            [_user(), _assistant("a1", "c1")],
            ["tool call 'c1' (search) has no result"],
        ),
    ],
)
def test_pairing_problems(messages, expected):
    assert harness.pairing_problems(messages) == expected

    with pytest.raises(harness.InconsistentHistory) as exc_info:
        harness.validate_pairing(messages)

    assert exc_info.value.problems == expected
    assert str(exc_info.value) == (
        "Refusing to send an inconsistent history: " + "; ".join(expected)
    )
    assert isinstance(exc_info.value, client_tools.ClientToolsError)


def test_pairing_problems_w_duplicate_message_ids():
    # Older TUI threads can hold these;  the server does not mind.
    messages = [_user(), _assistant("a1", content="x"), _user()]

    assert harness.pairing_problems(messages) == []


def test_pairing_problems_w_allow_pending():
    messages = [_user(), _assistant("a1", "c1")]

    assert harness.pairing_problems(messages, allow_pending=True) == []
    harness.validate_pairing(messages, allow_pending=True)


# -- Harness -----------------------------------------------------------------


@pytest.mark.parametrize("first", [False, True])
def test_harness_before_post_w_consistent_history(first):
    run_input = _run_input(
        [_user(), _assistant("a1", "c1"), _result("t1", "c1")]
    )

    found = harness.Harness().before_post(mock.Mock(), run_input, first=first)

    assert found is run_input


@pytest.mark.parametrize(
    "pairing_check, raises",
    [(True, True), (False, False)],
)
def test_harness_before_post_w_orphan_result(pairing_check, raises):
    run_input = _run_input([_user(), _result("t1", "c9")])
    the_harness = harness.Harness(pairing_check=pairing_check)

    if raises:
        with pytest.raises(harness.InconsistentHistory, match="'c9'"):
            the_harness.before_post(mock.Mock(), run_input, first=True)
    else:
        found = the_harness.before_post(mock.Mock(), run_input, first=True)
        assert found is run_input


# -- compaction ---------------------------------------------------------------


def _hit(chunk, rank, total=5, body_chars=2000, extra=()):
    lines = [
        f"[{chunk}] [rank {rank} of {total}]",
        "Collection: afman",
        f'Source: "AFMAN 10-3500 Vol 2" > Chapter {rank}',
        "Type: paragraph",
        *extra,
        "Content:",
        ("lorem ipsum " * (body_chars // 12 + 1))[:body_chars],
    ]
    return "\n".join(lines)


def _search_result(prefix="c", hits=5, **kwargs):
    return "\n\n---\n\n".join(
        _hit(f"{prefix}-{k}", k, hits, **kwargs) for k in range(1, hits + 1)
    )


def _shell_json(exit_code=0, stdout="x" * 3000, stderr="", **extra):
    return json.dumps(
        {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": False,
            "truncated": False,
            **extra,
        },
    )


def test_compaction_policy_defaults():
    policy = harness.CompactionPolicy()

    assert policy.mode == harness.COMPACTION_AUTO
    assert policy.keep_recent == harness.DEFAULT_KEEP_RECENT
    assert policy.min_elide_chars == harness.DEFAULT_MIN_ELIDE_CHARS
    assert policy.trigger_fraction == harness.DEFAULT_TRIGGER_FRACTION
    assert policy.target_fraction == harness.DEFAULT_TARGET_FRACTION


@pytest.mark.parametrize(
    "trigger, target",
    [(0.5, 0.5), (0.4, 0.5), (1.1, 0.5), (0.7, 0.0), (0.7, -0.1)],
)
def test_compaction_policy_w_bad_fractions(trigger, target):
    with pytest.raises(harness.InvalidCompactionFractions, match="target"):
        harness.CompactionPolicy(
            trigger_fraction=trigger,
            target_fraction=target,
        )


def test_compaction_policy_w_bad_mode():
    with pytest.raises(harness.InvalidCompactionMode, match="'sometimes'"):
        harness.CompactionPolicy(mode="sometimes")


def test_compact_content_search():
    content = _search_result(extra=["Figure caption (#/pictures/0): a map"])
    content += "\n\n---\n\nAlso matched, shown above: [c-1] [rank 6 of 6]"

    found = harness.compact_content("search", content)

    first, note, *rest = found.split("\n")
    assert first == (
        "[compacted by soliplex-tui harness: tool=search format=headers "
        f"original_bytes={len(content.encode())}]"
    )
    assert note == harness.SEARCH_NOTE
    skeleton = "\n".join(rest)
    for k in range(1, 6):
        assert f"[c-{k}] [rank {k} of 5]" in skeleton
        assert f'Source: "AFMAN 10-3500 Vol 2" > Chapter {k}' in skeleton
    assert "Collection: afman" in skeleton
    assert "Type: paragraph" in skeleton
    assert "Also matched, shown above: [c-1] [rank 6 of 6]" in skeleton
    assert "Content:" not in found
    assert "lorem" not in found
    # The whole header block is kept.
    assert found.count("Figure caption (#/pictures/0): a map") == 5
    assert len(found) < 1500
    # Byte-stable:  the same result always compacts to the same text.
    assert harness.compact_content("search", content) == found


def test_compact_content_search_w_score_header():
    content = "[c-1] (score: 0.87)\nContent:\n" + "text " * 400

    found = harness.compact_content("search", content)

    assert harness.compacted_info(found)["format"] == "headers"
    assert found.endswith("[c-1] (score: 0.87)")


def test_compact_content_search_w_rules_in_the_text():
    # A horizontal rule within a hit's text is not a separator:  every
    # hit's id survives.
    body = "intro\n\n---\n\nmore text " * 100
    content = "\n\n---\n\n".join(
        f"[c-{k}] [rank {k} of 3]\nSource: doc\nContent:\n{body}"
        for k in range(1, 4)
    )

    found = harness.compact_content("search", content)

    assert harness.compacted_info(found)["format"] == "headers"
    assert found.split("\n", 2)[2] == (
        "[c-1] [rank 1 of 3]\nSource: doc\n---\n"
        "[c-2] [rank 2 of 3]\nSource: doc\n---\n"
        "[c-3] [rank 3 of 3]\nSource: doc"
    )


NOT_A_SEARCH = [
    # A hit's first line is not a search header.
    "Not a search\nContent:\n" + "text " * 400,
    # A header, but no 'Content:'.
    "[c-1] [rank 1 of 1]\n" + "text " * 400,
    # A header line with more after it.
    "[c-1] [rank 1 of 1] and more\nContent:\n" + "text " * 400,
]


@pytest.mark.parametrize("content", NOT_A_SEARCH)
def test_compact_content_not_a_search(content):
    # From another tool:  cut like any other text.
    found = harness.compact_content("other", content)

    assert harness.compacted_info(found)["format"] == "head_tail"


@pytest.mark.parametrize("content", NOT_A_SEARCH)
def test_compact_content_search_not_recognized(content):
    # From 'search':  left alone, rather than lose its chunk ids.
    assert harness.compact_content("search", content) == content


def test_compact_content_shell_success():
    content = _shell_json(stdout="ok\n" * 1000, stderr="warning")

    found = harness.compact_content("shell", content)

    header, body = found.split("\n", 1)
    assert harness.compacted_info(found) == {
        "tool": "shell",
        "format": "shell",
        "original_bytes": str(len(content.encode())),
    }
    assert json.loads(body) == {
        "compacted": True,
        "exit_code": 0,
        "timed_out": False,
        "truncated": False,
        "stdout_chars": 3000,
        "stderr_chars": 7,
        "note": harness.SHELL_NOTE,
    }


def test_compact_content_shell_failure_keeps_why():
    stderr = "e" * 1000 + "THE ERROR"
    content = _shell_json(exit_code=2, stderr=stderr)

    body = json.loads(
        harness.compact_content("shell", content).split("\n", 1)[1]
    )

    assert body["exit_code"] == 2
    assert body["stderr_tail"] == stderr[-harness.STDERR_TAIL_CHARS :]
    assert body["stderr_tail"].endswith("THE ERROR")


def test_compact_content_shell_not_run():
    content = _shell_json(exit_code=None, stdout="", error="Refused: x" * 200)

    body = json.loads(
        harness.compact_content("shell", content).split("\n", 1)[1]
    )

    assert body["exit_code"] is None
    assert body["error"] == "Refused: x" * 200
    assert "stderr_tail" not in body


@pytest.mark.parametrize(
    "fields",
    [
        {"stdout": 42},
        {"stderr": ["a", "b"]},
        {"error": {"why": "x"}},
        {"exit_code": "1"},
        {"exit_code": True},
        {"exit_code": 1.5},
        {"timed_out": "no"},
    ],
)
def test_compact_content_shell_w_unexpected_shape(fields):
    # Not what the client makes:  left alone, lest a failure be lost.
    content = json.dumps(
        {"stdout": "x" * 3000, "stderr": "", "exit_code": 1} | fields,
    )

    assert harness.compact_content("shell", content) == content


def test_compact_content_shell_w_null_streams():
    content = json.dumps(
        {
            "stdout": None,
            "stderr": None,
            "exit_code": None,
            "error": "e" * 2000,
        },
    )

    body = json.loads(
        harness.compact_content("shell", content).split("\n", 1)[1],
    )

    assert body["stdout_chars"] == body["stderr_chars"] == 0
    assert body["error"] == "e" * 2000


def test_compact_content_other_tool_w_exit_code():
    # Only the client's 'shell' results are summarized so.
    content = _shell_json(exit_code=1)

    found = harness.compact_content("my_tool", content)

    assert harness.compacted_info(found)["format"] == "head_tail"


@pytest.mark.parametrize(
    "tool",
    ["other", "my tool", "a]b=c", "\u00fcn\u00efcode", "100%"],
)
def test_compact_content_marker_names_any_tool(tool):
    content = "x" * 2000

    found = harness.compact_content(tool, content)

    first_line = found.split("\n", 1)[0]
    assert first_line.count(" ") == 6  # the marker's, and its three words
    assert harness.compacted_info(found) == {
        "tool": tool,
        "format": "head_tail",
        "original_bytes": "2000",
    }


@pytest.mark.parametrize(
    "tool, content",
    [
        ("search", _search_result()),
        ("shell", _shell_json(exit_code=3, stderr="bad" * 500)),
        ("execute_code", "z" * 5000),
        ("other", "y" * 5000),
    ],
)
def test_compact_content_is_idempotent(tool, content):
    once = harness.compact_content(tool, content)

    assert harness.compact_content(tool, once) == once
    assert harness.compact_content("other", once) == once


@pytest.mark.parametrize(
    "tool, content, keep",
    [
        ("execute_code", "a" * 600 + "b" * 1000 + "c" * 600, 600),
        ("other", "a" * 300 + "b" * 1000 + "c" * 300, 300),
        # JSON, but not a shell result.
        ("other", json.dumps(["a" * 300 + "b" * 1000 + "c" * 300]), 300),
        ("other", json.dumps({"a" * 300 + "b" * 1000 + "c" * 300: 1}), 300),
    ],
)
def test_compact_content_head_tail(tool, content, keep):
    found = harness.compact_content(tool, content)

    header, body = found.split("\n", 1)
    assert harness.compacted_info(found)["format"] == "head_tail"
    omitted = len(content) - 2 * keep
    assert body == (
        f"{content[:keep]}\n...[{omitted} chars omitted]...\n{content[-keep:]}"
    )


@pytest.mark.parametrize(
    "content",
    ["plain text", "[compacted by soliplex-tui harness:] nothing"],
)
def test_compacted_info_w_uncompacted(content):
    assert harness.compacted_info(content) is None
    # Only a marker line which parses counts as compacted.
    assert harness.is_compacted(content) is False


def _history(*results, calls=None, start=0):
    """A user prompt, then one assistant call and result per 'results'

    'results' are '(tool, content)';  'calls' may override a call.  Ids
    are numbered from 'start'.
    """
    messages = [_user()]

    for index, (tool, content) in enumerate(results, start):
        call_id = f"call-{index}"
        call = (calls or {}).get(index) or _call(call_id, tool)
        messages.append(
            agui_core.AssistantMessage(id=f"a-{index}", tool_calls=[call]),
        )
        messages.append(_result(f"t-{index}", call_id, content))

    return messages


def _always(**kwargs):
    return harness.CompactionPolicy(mode=harness.COMPACTION_ALWAYS, **kwargs)


def test_compaction_candidates_keeps_the_newest():
    messages = _history(
        *[("search", _search_result(f"s{k}")) for k in range(6)]
    )

    found = harness.compaction_candidates(messages, _always(keep_recent=4))

    assert found == [2, 4]  # the two oldest results


def test_compaction_candidates_w_keep_recent_zero():
    messages = _history(
        *[("search", _search_result(f"s{k}")) for k in range(3)]
    )

    found = harness.compaction_candidates(messages, _always(keep_recent=0))

    assert found == [2, 4, 6]


def test_compaction_candidates_skips_the_ineligible():
    big = _search_result()
    encrypted_call = agui_core.ToolCall(
        id="call-2",
        function=agui_core.FunctionCall(name="search", arguments="{}"),
        encrypted_value="opaque",
    )
    messages = _history(
        ("cite", "x" * 5000),  # protected
        ("load_capability", "x" * 5000),  # protected
        ("search", big),  # its call is encrypted
        ("search", big),  # it is encrypted
        ("search", "short"),  # too short
        ("search", harness.compact_content("search", big)),  # compacted
        ("search", big),  # eligible
        calls={2: encrypted_call},
    )
    messages[8] = messages[8].model_copy(update={"encrypted_value": "opaque"})
    # A result whose call is unknown (never sent, but never compacted).
    messages.append(_result("t-orphan", "no-such-call", big))

    found = harness.compaction_candidates(messages, _always(keep_recent=0))

    assert found == [14]


def test_compaction_candidates_w_nothing_to_gain():
    # Compacted, it would be no shorter:  left alone.
    messages = _history(("other", "x" * 100))

    found = harness.compaction_candidates(
        messages,
        _always(keep_recent=0, min_elide_chars=10),
    )

    assert found == []


def test_compact_history():
    messages = _history(
        ("search", _search_result("s0")),
        ("search", _search_result("s1")),
    )

    found = harness.compact_history(messages, [2])

    assert [message.id for message in found] == [m.id for m in messages]
    assert harness.is_compacted(found[2].content)
    assert found[2].tool_call_id == messages[2].tool_call_id
    # Only the compacted message is new;  the rest are the same objects.
    assert all(
        a is b
        for a, b in zip(found, messages, strict=True)
        if a is not found[2]
    )
    assert messages[2].content == _search_result("s0")  # not changed


def test_compact_history_twice_and_w_duplicate_indexes():
    messages = _history(
        ("search", _search_result("s0")),
        ("search", _search_result("s1")),
    )

    once = harness.compact_history(messages, [2, 2, 4])
    twice = harness.compact_history(once, [4, 2])

    assert [m.content for m in once] == [m.content for m in twice]
    # Already compacted:  left the same object.
    assert all(a is b for a, b in zip(once, twice, strict=True))


def test_compacted_history_loads_on_the_server():
    # The server's adapter keeps a compacted result as the string it is
    # (it is not JSON), with its call, in the same number of messages.
    from pydantic_ai import messages as ai_messages
    from pydantic_ai.ui import ag_ui as ai_ag_ui

    messages = _history(
        ("search", _search_result("s0")),
        ("shell", _shell_json(exit_code=1, stderr="oops" * 300)),
        ("search", _search_result("s2")),
    )
    compacted = harness.compact_history(messages, [2, 4])

    before = ai_ag_ui.AGUIAdapter.load_messages(messages)
    after = ai_ag_ui.AGUIAdapter.load_messages(compacted)

    assert len(after) == len(before)
    returns = [
        part
        for message in after
        for part in message.parts
        if isinstance(part, ai_messages.ToolReturnPart)
    ]
    assert [part.tool_call_id for part in returns] == [
        "call-0",
        "call-1",
        "call-2",
    ]
    assert returns[0].content == compacted[2].content
    assert returns[1].content == compacted[4].content
    assert harness.compacted_info(returns[1].content)["format"] == "shell"


def test_wire_chars():
    messages = [_user(content="hé")]

    assert harness.wire_chars(messages) == len(
        '[{"id": "u1", "role": "user", "content": "hé"}]',
    )


def test_harness_before_post_compacts_always():
    messages = _history(
        *[("search", _search_result(f"s{k}")) for k in range(6)]
    )
    run_input = _run_input(messages, state={"rag": {"x": 1}})
    on_report = mock.Mock()
    the_harness = harness.Harness(
        compaction=_always(keep_recent=4),
        on_report=on_report,
    )

    found = the_harness.before_post(mock.Mock(), run_input, first=True)

    assert found is not run_input
    assert [
        harness.is_compacted(m.content)
        for m in found.messages
        if m.role == "tool"
    ] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert run_input.messages == messages  # the input is not changed
    (report,) = the_harness.reports
    on_report.assert_called_once_with(report)
    assert report.as_json() == {
        "thread_id": "thread-1",
        "run_id": "run-1",
        "parent_run_id": None,
        "first": True,
        "mode": "always",
        "messages": 13,
        "resend_chars": harness.wire_chars(found.messages),
        "state_chars": len('{"rag": {"x": 1}}'),
        "compacted": 2,
        "compacted_chars": (
            harness.wire_chars(messages) - harness.wire_chars(found.messages)
        ),
        "compacted_total": 2,
        "est_tokens": round(harness.wire_chars(found.messages) / 3.5),
        "window_tokens": None,
        "window_source": None,
        "stale": False,
        "answered": 0,
        "trimmed": [],
    }

    # Again:  nothing more to do, and the same bytes.
    again = the_harness.before_post(mock.Mock(), found, first=False)

    assert again is found
    assert the_harness.reports[-1].compacted == 0
    assert the_harness.reports[-1].compacted_chars == 0
    assert the_harness.reports[-1].compacted_total == 2


def test_harness_before_post_w_compaction_off():
    messages = _history(
        *[("search", _search_result(f"s{k}")) for k in range(6)]
    )
    run_input = _run_input(messages)
    the_harness = harness.Harness(
        compaction=harness.CompactionPolicy(mode=harness.COMPACTION_OFF),
        budget=harness.ContextBudget(window_tokens=5000),
    )

    found = the_harness.before_post(mock.Mock(), run_input, first=True)

    assert found is run_input
    (report,) = the_harness.reports
    assert report.mode == "off"
    assert report.compacted == 0


def test_compacted_info_w_other_words():
    content = f"{harness.COMPACTED_MARKER} format=x size=3]\nbody"

    assert harness.compacted_info(content) == {"format": "x", "size": "3"}


@pytest.mark.parametrize("content", ["not json " * 300, json.dumps([1] * 900)])
def test_compact_content_shell_not_a_result(content):
    assert harness.compact_content("shell", content) == content


# -- context budget ----------------------------------------------------------


@pytest.mark.parametrize(
    "room_info, expected",
    [
        ({}, None),
        ({"agent": None}, None),
        ({"agent": {"kind": "factory"}}, None),
        ({"agent": {"context_window": 98304}}, 98304),
    ],
)
def test_room_context_window(room_info, expected):
    assert harness.room_context_window(room_info) == expected


def _vllm_room(base_url="http://vllm:8000/v1", model="glimmer"):
    return {"agent": {"provider_base_url": base_url, "model_name": model}}


def _models_http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "base_url",
    ["http://vllm:8000/v1", "http://vllm:8000/v1/", "http://vllm:8000"],
)
def test_probe_model_window(base_url):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": [
                    "junk",
                    {"id": "other", "max_model_len": 1},
                    {"id": "glimmer", "max_model_len": 98304},
                ],
            },
        )

    found = harness.probe_model_window(
        _vllm_room(base_url),
        _models_http(handler),
    )

    assert found == 98304
    assert urls == ["http://vllm:8000/v1/models"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": [{"id": "other"}]}),
        httpx.Response(200, json={"data": [{"id": "glimmer"}]}),
        httpx.Response(
            200,
            json={"data": [{"id": "glimmer", "max_model_len": "big"}]},
        ),
        httpx.Response(200, json={"models": []}),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, text="not json"),
        httpx.Response(500, text="down"),
    ],
)
def test_probe_model_window_w_nothing_useful(response):
    found = harness.probe_model_window(
        _vllm_room(),
        _models_http(lambda request: response),
    )

    assert found is None


def test_probe_model_window_w_transport_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    assert (
        harness.probe_model_window(_vllm_room(), _models_http(handler)) is None
    )


@pytest.mark.parametrize(
    "room_info",
    [{}, _vllm_room(base_url=None), _vllm_room(model=None)],
)
def test_probe_model_window_w_no_server(room_info):
    http = _models_http(mock.Mock(side_effect=AssertionError("no request")))

    assert harness.probe_model_window(room_info, http) is None


@pytest.mark.parametrize(
    "kwargs, room_info, probed, expected",
    [
        (
            {"context_window": 1000},
            {"agent": {"context_window": 2}},
            3,
            (1000, "option"),
        ),
        ({}, {"agent": {"context_window": 2}}, 3, (2, "room")),
        ({}, {}, 3, (3, "model server")),
        ({}, {}, None, (None, None)),
    ],
)
def test_resolve_window(kwargs, room_info, probed, expected):
    probe = mock.Mock(return_value=probed)

    found = harness.resolve_window(room_info, probe=probe, **kwargs)

    assert found == expected


def test_resolve_window_without_probe():
    assert harness.resolve_window({}) == (None, None)


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, None),
        ({}, None),
        ({"input_tokens": 500, "final_input_tokens": None}, None),
        ({"final_input_tokens": 100}, 100),
        ({"final_input_tokens": 100, "final_output_tokens": None}, 100),
        # Never the cumulative 'input_tokens'.
        (
            {
                "input_tokens": 900,
                "final_input_tokens": 100,
                "final_output_tokens": 20,
            },
            120,
        ),
    ],
)
def test_measured_tokens(usage, expected):
    assert harness.measured_tokens(usage) == expected


def test_context_budget_unknown_window():
    budget = harness.ContextBudget()

    assert budget.usable_tokens is None
    assert budget.fraction(100) is None
    assert budget.estimate(350) == 100  # at 3.5 characters a token


def test_context_budget_usable_and_fraction():
    budget = harness.ContextBudget(window_tokens=10_000, output_reserve=2_000)

    assert budget.usable_tokens == 8_000
    # A meter's reading:  of the whole window, not of what is usable.
    assert budget.fraction(2_000) == 0.2


@pytest.mark.parametrize(
    "window, reserve",
    [(1_000, 2_000), (4_096, 4_096), (10_000, -1)],
)
def test_context_budget_w_no_room(window, reserve):
    # A reserve which leaves nothing usable is refused, not silently
    # taken for an unknown window.
    with pytest.raises(harness.InvalidContextBudget, match="larger than"):
        harness.ContextBudget(window_tokens=window, output_reserve=reserve)


def test_context_budget_anchor():
    budget = harness.ContextBudget(window_tokens=100_000)

    budget.anchor(
        {"final_input_tokens": 5_000, "final_output_tokens": 200}, 7_000
    )

    assert (budget.anchor_tokens, budget.anchor_chars, budget.stale) == (
        5_200,
        7_000,
        False,
    )
    # Measured, plus (or less) the characters since, at 3.5 a token.
    assert budget.estimate(7_000) == 5_200
    assert budget.estimate(14_000) == 7_200
    assert budget.estimate(3_500) == 4_200
    assert budget.estimate(0) == 3_200
    assert (
        harness.ContextBudget(anchor_tokens=10, anchor_chars=10_000).estimate(
            0
        )
        == 0
    )

    # A run with no measurement keeps the anchor, but it is stale.
    budget.anchor(None, 20_000)

    assert (budget.anchor_tokens, budget.anchor_chars, budget.stale) == (
        5_200,
        7_000,
        True,
    )

    budget.anchor({"final_input_tokens": 9_000}, 21_000)

    assert (budget.anchor_tokens, budget.stale) == (9_000, False)


def test_context_budget_no_anchor_is_not_stale():
    budget = harness.ContextBudget()

    budget.anchor(None, 100)

    assert budget.anchor_tokens is None
    assert budget.stale is False


def _auto(**kwargs):
    return harness.CompactionPolicy(mode=harness.COMPACTION_AUTO, **kwargs)


def _searches(count, start=0):
    return [
        ("search", _search_result(f"s{k}"))
        for k in range(start, start + count)
    ]


def _tool_contents(messages):
    return [m.content for m in messages if m.role == "tool"]


def test_harness_auto_waits_for_the_high_water_mark():
    messages = _history(*_searches(6))
    chars = harness.wire_chars(messages)
    # Estimate (chars / 3.5) just under 70% of the usable window.
    window = int(chars / 3.5 / 0.69) + harness.DEFAULT_OUTPUT_RESERVE
    the_harness = harness.Harness(
        compaction=_auto(),
        budget=harness.ContextBudget(window_tokens=window),
    )
    run_input = _run_input(messages)

    found = the_harness.before_post(mock.Mock(), run_input, first=True)

    assert found is run_input
    (report,) = the_harness.reports
    assert report.compacted == 0
    assert report.window_tokens == window
    assert report.est_tokens == round(chars / 3.5)


def test_harness_auto_compacts_one_batch_to_the_low_water_mark():
    messages = _history(*_searches(20))
    chars = harness.wire_chars(messages)
    # Estimate at 80% of the usable window:  over the 70% trigger.
    window = int(chars / 3.5 / 0.8) + harness.DEFAULT_OUTPUT_RESERVE
    budget = harness.ContextBudget(window_tokens=window, window_source="room")
    the_harness = harness.Harness(compaction=_auto(), budget=budget)

    found = the_harness.before_post(
        mock.Mock(), _run_input(messages), first=True
    )

    (report,) = the_harness.reports
    # Oldest first, just enough to get under 40%;  the newest 4 are kept.
    compacted = [
        harness.is_compacted(c) for c in _tool_contents(found.messages)
    ]
    assert compacted == [True] * report.compacted + [False] * (
        20 - report.compacted
    )
    assert 0 < report.compacted < 16  # not every candidate:  just enough
    assert report.est_tokens <= 0.4 * budget.usable_tokens
    assert report.window_source == "room"
    # One fewer, and it would still be over the low-water mark.
    partial = harness.compact_history(
        messages,
        [2 * k + 2 for k in range(report.compacted - 1)],
    )
    assert (
        budget.estimate(harness.wire_chars(partial))
        > 0.4 * budget.usable_tokens
    )

    # Growing again, below the trigger:  the history is only appended
    # to, so the prefix already sent stays byte-identical.
    grown = [
        *found.messages,
        *_history(*_searches(1, start=20), start=20)[1:],
    ]
    again = the_harness.before_post(mock.Mock(), _run_input(grown), first=True)

    assert the_harness.reports[-1].compacted == 0
    sent_before = [m.model_dump_json() for m in found.messages]
    sent_after = [m.model_dump_json() for m in again.messages]
    assert sent_after[: len(sent_before)] == sent_before


def test_harness_auto_stops_at_keep_recent():
    # Even over the trigger, the newest 'keep_recent' are never compacted.
    messages = _history(*_searches(5))
    budget = harness.ContextBudget(
        window_tokens=harness.DEFAULT_OUTPUT_RESERVE + 10
    )
    the_harness = harness.Harness(compaction=_auto(), budget=budget)

    found = the_harness.before_post(
        mock.Mock(), _run_input(messages), first=True
    )

    assert [
        harness.is_compacted(c) for c in _tool_contents(found.messages)
    ] == [
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize("window", [None])
def test_harness_auto_without_a_usable_window(window):
    # No window known (or none usable):  nothing is compacted.
    messages = _history(*_searches(10))
    the_harness = harness.Harness(
        compaction=_auto(),
        budget=harness.ContextBudget(window_tokens=window),
    )
    run_input = _run_input(messages)

    assert (
        the_harness.before_post(mock.Mock(), run_input, first=True)
        is run_input
    )


def test_harness_after_run_anchors_the_budget():
    run_input = _run_input(_history(*_searches(1)))
    client = mock.Mock(spec=["run_usage"])
    client.run_usage.return_value = {
        "final_input_tokens": 1_000,
        "final_output_tokens": 50,
    }
    the_harness = harness.Harness()

    the_harness.after_run(client, run_input)

    client.run_usage.assert_called_once_with("thread-1", "run-1")
    assert the_harness.budget.anchor_tokens == 1_050
    assert the_harness.budget.anchor_chars == harness.wire_chars(
        run_input.messages
    )

    # The next POST's estimate starts from it.
    the_harness.before_post(client, run_input, first=True)

    assert the_harness.reports[-1].est_tokens == 1_050
    assert the_harness.reports[-1].stale is False


def test_harness_after_run_w_failed_usage():
    run_input = _run_input(_history(*_searches(1)))
    client = mock.Mock(spec=["run_usage"])
    client.run_usage.side_effect = client_tools.TransportFailure(
        httpx.ConnectError("refused"),
    )
    the_harness = harness.Harness(
        budget=harness.ContextBudget(anchor_tokens=10, anchor_chars=10),
    )

    the_harness.after_run(client, run_input)  # does not raise

    assert the_harness.budget.anchor_tokens == 10
    assert the_harness.budget.stale is True
    the_harness.before_post(client, run_input, first=True)
    assert the_harness.reports[-1].stale is True


@pytest.mark.parametrize("probe_window", [False, True])
def test_make_harness(probe_window):
    room_info = _vllm_room()

    with mock.patch.object(
        harness,
        "probe_model_window",
        return_value=262_144,
    ) as probe:
        found = harness.make_harness(
            room_info,
            probe_window=probe_window,
            output_reserve=100,
            pairing_check=False,
        )

    assert found.pairing_check is False
    assert found.budget.output_reserve == 100

    if probe_window:
        ((info, http), _) = probe.call_args
        assert info is room_info
        assert isinstance(http, httpx.Client)
        assert http.is_closed
        assert (found.budget.window_tokens, found.budget.window_source) == (
            262_144,
            "model server",
        )
    else:
        probe.assert_not_called()
        assert found.budget.window_tokens is None


def test_make_harness_w_context_window():
    found = harness.make_harness({}, context_window=5000)

    assert (found.budget.window_tokens, found.budget.window_source) == (
        5000,
        "option",
    )
    assert found.budget.output_reserve == harness.DEFAULT_OUTPUT_RESERVE


# -- compaction:  review findings --------------------------------------------


@pytest.mark.parametrize(
    "keep_recent, min_elide_chars",
    [(-1, 1024), (4, 0), (4, -5)],
)
def test_compaction_policy_w_bad_counts(keep_recent, min_elide_chars):
    with pytest.raises(harness.InvalidCompactionCounts, match="keep_recent"):
        harness.CompactionPolicy(
            keep_recent=keep_recent,
            min_elide_chars=min_elide_chars,
        )


def test_compact_content_haiku_rag_search_contract():
    # The real formatter's output, joined as 'search_corpus' joins it.
    from haiku.rag.store.models import chunk

    results = [
        chunk.SearchResult(
            content=f"Body of hit {k}.\n\n---\n\nA rule within it. " * 60,
            score=0.5,
            chunk_id=f"chunk-{k}",
            source="afman",
            document_title="AFMAN 10-3500\nContent: Vol 2",  # a newline
            headings=["Chapter 3", f"3.{k}"],
            labels=["paragraph"],
            picture_captions={"#/pictures/0": "A map"},
        )
        for k in range(1, 4)
    ]
    content = "\n\n---\n\n".join(
        [
            result.format_for_agent(
                rank=k,
                total=4,
                include_collection=True,
            )
            for k, result in enumerate(results, 1)
        ]
        + ["Also matched, shown above: [chunk-1] [rank 4 of 4]"],
    )

    found = harness.compact_content("search", content)

    assert harness.compacted_info(found)["format"] == "headers"
    skeleton = found.split("\n", 2)[2]
    blocks = skeleton.split("\n---\n")
    assert blocks[-1] == "Also matched, shown above: [chunk-1] [rank 4 of 4]"
    for k, block in enumerate(blocks[:-1], 1):
        # The whole header block, the multi-line title included.
        assert block == (
            f"[chunk-{k}] [rank {k} of 4]\n"
            "Collection: afman\n"
            'Source: "AFMAN 10-3500\nContent: Vol 2" > Chapter 3 > '
            f"3.{k}\n"
            "Type: paragraph\n"
            "Figure caption (#/pictures/0): A map"
        )
    assert "Body of hit" not in found


def test_compact_content_shell_never_as_a_search():
    # A (non-JSON) shell result shaped like a search is left alone:  its
    # failure must not be cut away with the "body".
    content = "[c1] [rank 1 of 1]\nContent:\n" + "x" * 2000 + "\nexit_code=1"

    assert harness.compact_content("shell", content) == content


def test_compact_content_shell_w_deep_nesting():
    # Too deep for the JSON decoder:  left alone, not a failed POST.
    content = "[" * 10_000 + "0" + "]" * 10_000
    messages = _history(("shell", content))

    assert harness.compact_content("shell", content) == content
    assert (
        harness.compaction_candidates(messages, _always(keep_recent=0)) == []
    )


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"timed_out": True, "exit_code": -9}, {"timed_out": True}),
        ({"truncated": True}, {"truncated": True}),
    ],
)
def test_compact_content_shell_keeps_its_flags(fields, expected):
    content = _shell_json(**fields)
    content = json.dumps(json.loads(content) | fields)

    body = json.loads(
        harness.compact_content("shell", content).split("\n", 1)[1],
    )

    assert body.items() >= expected.items()


def test_compact_content_shell_w_bad_truncated():
    content = json.dumps(json.loads(_shell_json()) | {"truncated": "yes"})

    assert harness.compact_content("shell", content) == content


def test_compact_content_original_bytes_are_utf8():
    content = "\u20ac" * 1000  # 3 bytes each

    found = harness.compact_content("other", content)

    assert harness.compacted_info(found)["original_bytes"] == "3000"


def test_compaction_candidates_protects_search_tools():
    messages = _history(("search_tools", "x" * 5000), ("other", "y" * 5000))

    found = harness.compaction_candidates(messages, _always(keep_recent=0))

    assert found == [4]


@pytest.mark.parametrize("size, expected", [(1023, []), (1024, [2])])
def test_compaction_candidates_at_min_elide_chars(size, expected):
    messages = _history(("other", "x" * size))

    found = harness.compaction_candidates(messages, _always(keep_recent=0))

    assert found == expected


def test_compaction_candidates_keep_recent_counts_only_the_eligible():
    # Small, protected and compacted results are not among the "recent":
    # the newest 2 *eligible* results are kept.
    big = _search_result()
    messages = _history(
        ("search", big),
        ("search", big),
        ("search", big),
        ("cite", "x" * 5000),
        ("search", "small"),
    )

    found = harness.compaction_candidates(messages, _always(keep_recent=2))

    assert found == [2]


def test_harness_before_post_keeps_the_rest_of_the_input():
    messages = _history(*_searches(6))
    run_input = agui_core.RunAgentInput(
        thread_id="thread-1",
        run_id="run-2",
        parent_run_id="run-1",
        state={"rag": {"searches": {"q": [1, 2]}}},
        messages=messages,
        tools=[
            agui_core.Tool(name="shell", description="d", parameters={}),
        ],
        context=[agui_core.Context(description="c", value="v")],
        forwarded_props={"x": 1},
    )
    the_harness = harness.Harness(compaction=_always())

    found = the_harness.before_post(mock.Mock(), run_input, first=False)

    assert found.model_dump(exclude={"messages"}) == run_input.model_dump(
        exclude={"messages"},
    )
    assert the_harness.reports[-1].parent_run_id == "run-1"
    assert the_harness.reports[-1].compacted == 2


def test_harness_before_post_compacts_then_checks_pairing():
    # Compaction never hides an inconsistent history.
    messages = [*_history(*_searches(6)), _result("t-orphan", "no-call")]
    the_harness = harness.Harness(compaction=_always())

    with pytest.raises(harness.InconsistentHistory, match="'no-call'"):
        the_harness.before_post(
            mock.Mock(),
            _run_input(messages),
            first=True,
        )

    assert the_harness.reports == []


# -- unanswered calls --------------------------------------------------------


def _unanswered_content():
    return json.loads(
        client_tools.tool_result_content(
            client_tools._not_run(harness.UNANSWERED_ERROR),
        ),
    )


def test_answer_unanswered_w_duplicate_call_ids():
    # The same id twice is one call:  answered once.
    messages = [_user(), _assistant("a1", "c1"), _assistant("a2", "c1")]

    found, added = harness.answer_unanswered(messages)

    assert added == 1
    assert [m.role for m in found] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_answer_unanswered_w_nothing_to_do():
    messages = [_user(), _assistant("a1", "c1"), _result("t1", "c1")]

    found, added = harness.answer_unanswered(messages)

    assert (found, added) == (messages, 0)


def test_answer_unanswered():
    # A reloaded thread:  a run cancelled with 'c2' unanswered (its
    # sibling 'c1' answered), then a new prompt;  and a trailing call.
    messages = [
        _user("u1"),
        _assistant("a1", "c1", "c2"),
        _result("t1", "c1"),
        _user("u2"),
        _assistant("a2", "c3"),
        _assistant("a3", content="thinking"),
    ]

    found, added = harness.answer_unanswered(messages)

    assert added == 2
    assert [m.id for m in found if m.role != "tool" or m.id == "t1"] == [
        "u1",
        "a1",
        "t1",
        "u2",
        "a2",
        "a3",
    ]
    assert [(m.role, getattr(m, "tool_call_id", None)) for m in found] == [
        ("user", None),
        ("assistant", None),
        ("tool", "c1"),
        ("tool", "c2"),  # after its call's own results
        ("user", None),
        ("assistant", None),
        ("tool", "c3"),
        ("assistant", None),
    ]
    assert json.loads(found[3].content) == _unanswered_content()
    assert harness.pairing_problems(found) == []


def test_harness_before_post_answers_unanswered_at_a_prompt_start():
    messages = [_user("u1"), _assistant("a1", "c1"), _user("u2")]
    the_harness = harness.Harness()

    found = the_harness.before_post(
        mock.Mock(),
        _run_input(messages),
        first=True,
    )

    assert [m.role for m in found.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert the_harness.reports[-1].answered == 1


def test_harness_before_post_mid_chain_does_not_answer():
    # Within 'run_loop', every call has its result by then:  a missing
    # one is a bug, and refused.
    messages = [_user("u1"), _assistant("a1", "c1")]
    the_harness = harness.Harness()

    with pytest.raises(harness.InconsistentHistory, match="no result"):
        the_harness.before_post(
            mock.Mock(),
            _run_input(messages),
            first=False,
        )


def test_harness_before_post_without_pairing_check_answers_nothing():
    messages = [_user("u1"), _assistant("a1", "c1"), _user("u2")]
    the_harness = harness.Harness(pairing_check=False)
    run_input = _run_input(messages)

    found = the_harness.before_post(mock.Mock(), run_input, first=True)

    assert found is run_input


# -- RAG state ---------------------------------------------------------------


def _rag(in_progress, searches=None, **extra):
    return {
        "citation_index": {"c-1": {"chunk_id": "c-1"}},
        "citations": ["c-1"],
        "evidence": {"question": 3, "in_progress": in_progress},
        "document_filter": None,
        "sources": None,
        "searches": (
            {"q": [{"content": "x" * 5000}]} if searches is None else searches
        ),
        "executions": [{"code": "print(1)"}],
        **extra,
    }


def test_trim_rag_state_at_a_finished_question():
    state = {"rag": _rag(False), "other": {"keep": 1}}

    found, names = harness.trim_rag_state(state, harness.TRIM_BOUNDARY)

    assert names == ["rag"]
    assert found["other"] is state["other"]
    assert found["rag"] == {
        "citation_index": {"c-1": {"chunk_id": "c-1"}},
        "citations": [],
        "evidence": {"question": 3, "in_progress": False},
        "document_filter": None,
        "sources": None,
        "searches": {},
        "executions": [],
    }
    assert state["rag"]["searches"]  # the input is not changed

    # Idempotent:  nothing more to trim, the same object back.
    again, names = harness.trim_rag_state(found, harness.TRIM_BOUNDARY)

    assert again is found
    assert names == []


def test_trim_rag_state_mid_question():
    state = {"rag": _rag(True)}

    found, names = harness.trim_rag_state(state, harness.TRIM_BOUNDARY)

    assert (found, names) == (state, [])

    found, names = harness.trim_rag_state(state, harness.TRIM_AGGRESSIVE)

    assert names == ["rag"]
    assert found["rag"]["searches"] == {}
    assert found["rag"]["executions"] == []
    assert found["rag"]["citations"] == ["c-1"]  # kept mid-question


def test_trim_rag_state_w_several_namespaces():
    # Any namespace of that shape, not just 'rag';  but not one lacking
    # a 'citation_index'.
    unrelated = _rag(False)
    del unrelated["citation_index"]
    state = {
        "rag": _rag(False),
        "rag_b": _rag(False),
        "rag_c": _rag(True),
        "other": unrelated,
    }

    found, names = harness.trim_rag_state(state, harness.TRIM_BOUNDARY)

    assert names == ["rag", "rag_b"]
    assert found["rag_c"] is state["rag_c"]
    assert found["other"] is state["other"]


@pytest.mark.parametrize(
    "state",
    [
        None,
        [],
        {"rag": "text"},
        {"rag": {"searches": {"q": []}, "citation_index": {}}},  # no evidence
        {"rag": {"searches": {"q": []}, "citation_index": {}, "evidence": []}},
        {
            "rag": {
                "searches": {"q": []},
                "citation_index": {},
                "evidence": {"in_progress": "no"},
            },
        },
        {"rag": {"searches": [1], "evidence": {"in_progress": False}}},
        # Already trimmed, and fields missing:  nothing to do.
        {
            "rag": {
                "searches": {},
                "citation_index": {},
                "evidence": {"in_progress": False},
            },
        },
    ],
)
def test_trim_rag_state_w_nothing_to_trim(state):
    found, names = harness.trim_rag_state(state, harness.TRIM_AGGRESSIVE)

    assert found is state
    assert names == []


def test_trim_rag_state_off():
    state = {"rag": _rag(False)}

    assert harness.trim_rag_state(state, harness.TRIM_OFF) == (state, [])


def test_trim_rag_state_w_bad_mode():
    with pytest.raises(harness.InvalidTrimMode, match="'some'"):
        harness.trim_rag_state({}, "some")

    with pytest.raises(harness.InvalidTrimMode, match="'some'"):
        harness.Harness(trim_rag_state="some")


@pytest.mark.parametrize(
    "mode, first, trimmed",
    [
        (harness.TRIM_BOUNDARY, True, True),
        (harness.TRIM_BOUNDARY, False, False),  # mid-chain:  never
        (harness.TRIM_AGGRESSIVE, False, True),
        (harness.TRIM_OFF, True, False),
    ],
)
def test_harness_before_post_trims_rag_state(mode, first, trimmed):
    messages = _history(("search", "small"))
    state = {"rag": _rag(False)}
    run_input = _run_input(messages, state=state)
    the_harness = harness.Harness(trim_rag_state=mode)

    found = the_harness.before_post(mock.Mock(), run_input, first=first)

    (report,) = the_harness.reports
    if trimmed:
        assert found.state["rag"]["searches"] == {}
        assert found.messages is run_input.messages
        assert report.trimmed == ["rag"]
        assert report.state_chars == len(json.dumps(found.state))
        assert report.state_chars < len(json.dumps(state))
    else:
        assert found is run_input
        assert report.trimmed == []


# -- context budget:  review findings ----------------------------------------


@pytest.mark.parametrize(
    "body",
    [{"data": None}, {"data": 123}, {"data": "text"}],
)
def test_probe_model_window_w_bad_data(body):
    found = harness.probe_model_window(
        _vllm_room(),
        _models_http(lambda request: httpx.Response(200, json=body)),
    )

    assert found is None


@pytest.mark.parametrize("window", [True, -1, 1.5, None])
def test_probe_model_window_w_bad_window(window):
    body = {"data": [{"id": "glimmer", "max_model_len": window}]}

    found = harness.probe_model_window(
        _vllm_room(),
        _models_http(lambda request: httpx.Response(200, json=body)),
    )

    assert found is None


@pytest.mark.parametrize(
    "kwargs, room_info",
    [
        ({"context_window": 1000}, {}),
        ({}, {"agent": {"context_window": 2000}}),
    ],
)
def test_resolve_window_does_not_probe_when_known(kwargs, room_info):
    probe = mock.Mock(side_effect=AssertionError("probed"))

    window, _ = harness.resolve_window(room_info, probe=probe, **kwargs)

    assert window is not None
    probe.assert_not_called()


@pytest.mark.parametrize(
    "usage",
    [
        [1],
        "text",
        {"final_input_tokens": "100"},
        {"final_input_tokens": True},
        {"final_input_tokens": -5},
    ],
)
def test_measured_tokens_w_bad_usage(usage):
    assert harness.measured_tokens(usage) is None


def test_measured_tokens_w_bad_output_tokens():
    # A record which cannot be read reports nothing:  not an undercount.
    usage = {"final_input_tokens": 100, "final_output_tokens": "8000"}

    assert harness.measured_tokens(usage) is None

    budget = harness.ContextBudget(anchor_tokens=9_000, anchor_chars=10)
    budget.anchor(usage, 20)

    assert (budget.anchor_tokens, budget.stale) == (9_000, True)


def test_probe_model_window_w_zero_window():
    # No window, rather than a budget refused.
    body = {"data": [{"id": "glimmer", "max_model_len": 0}]}
    room_info = _vllm_room()

    found = harness.probe_model_window(
        room_info,
        _models_http(lambda request: httpx.Response(200, json=body)),
    )

    assert found is None

    with mock.patch.object(harness, "probe_model_window", return_value=None):
        the_harness = harness.make_harness(room_info, probe_window=True)

    assert the_harness.budget.window_tokens is None


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Expecting value"),  # a body which is not JSON
        client_tools.HTTPFailure(
            mock.Mock(status_code=500, url="u", text="x")
        ),
    ],
)
def test_harness_after_run_w_bad_usage_body(error):
    run_input = _run_input(_history(*_searches(1)))
    client = mock.Mock(spec=["run_usage"])
    client.run_usage.side_effect = error
    the_harness = harness.Harness(
        budget=harness.ContextBudget(anchor_tokens=10, anchor_chars=10),
    )

    the_harness.after_run(client, run_input)  # does not raise

    assert the_harness.budget.stale is True


@pytest.mark.parametrize("usage", [[1], {"final_input_tokens": "100"}])
def test_harness_after_run_w_bad_usage_shape(usage):
    run_input = _run_input(_history(*_searches(1)))
    client = mock.Mock(spec=["run_usage"])
    client.run_usage.return_value = usage
    the_harness = harness.Harness()

    the_harness.after_run(client, run_input)  # does not raise

    assert the_harness.budget.anchor_tokens is None


def test_harness_auto_exactly_at_the_trigger():
    # At the trigger, not over it:  nothing compacted.
    messages = _history(*_searches(6))
    chars = harness.wire_chars(messages)
    budget = harness.ContextBudget(
        window_tokens=10_000,
        output_reserve=0,
        anchor_tokens=7_000,
        anchor_chars=chars,
    )
    the_harness = harness.Harness(compaction=_auto(), budget=budget)
    run_input = _run_input(messages)

    assert budget.estimate(chars) == 7_000
    assert the_harness.before_post(mock.Mock(), run_input, first=True) is (
        run_input
    )


def test_harness_auto_savings_counted_as_json():
    # Escape-heavy content:  savings are counted as the estimate counts
    # characters (as JSON), so the batch stops as soon as it is enough.
    content = "\u0001" * 10_000
    messages = _history(*[("other", content) for _ in range(3)])
    chars = harness.wire_chars(messages)
    window = int(chars / 3.5 / 0.75)
    budget = harness.ContextBudget(window_tokens=window, output_reserve=0)
    the_harness = harness.Harness(
        compaction=_auto(keep_recent=0),
        budget=budget,
    )

    found = the_harness.before_post(
        mock.Mock(),
        _run_input(messages),
        first=True,
    )

    report = the_harness.reports[-1]
    assert report.est_tokens <= 0.4 * budget.usable_tokens
    # Not all three:  two were enough.
    assert report.compacted == 2
    assert harness.is_compacted(_tool_contents(found.messages)[1])
    assert not harness.is_compacted(_tool_contents(found.messages)[2])


def test_make_harness_w_window_under_the_reserve():
    with pytest.raises(harness.InvalidContextBudget):
        harness.make_harness({}, context_window=4096)


# -- verification round ------------------------------------------------------


def test_compact_content_search_w_a_content_line_in_a_title():
    # A title holding a line 'Content:' makes the header's end ambiguous:
    # the result is left alone, rather than lose part of its header.
    content = (
        "[c-1] [rank 1 of 1]\n"
        'Source: "Manual\nContent:\nVolume 2" > Chapter 1\n'
        "Type: paragraph\n"
        "Content:\n" + "text " * 500
    )

    assert harness.compact_content("search", content) == content


def test_compaction_candidates_counts_utf8_bytes():
    # Fewer characters once escapes are gone is not fewer bytes sent:
    # four-byte characters in the body, a long header kept.
    content = (
        "[c] [rank 1 of 1]\nSource: " + "a" * 1000 + "\nContent:\n"
        "\U0001f600" * 20
    )
    messages = _history(("search", content))

    found = harness.compaction_candidates(
        messages,
        _always(keep_recent=0, min_elide_chars=1),
    )

    assert found == []


def test_unanswered_error_says_the_outcome_is_unknown():
    assert "unknown" in harness.UNANSWERED_ERROR
    assert "not run" not in harness.UNANSWERED_ERROR


def test_trim_rag_state_aggressive_is_idempotent_mid_question():
    state = {"rag": _rag(True)}

    once, _ = harness.trim_rag_state(state, harness.TRIM_AGGRESSIVE)
    again, names = harness.trim_rag_state(once, harness.TRIM_AGGRESSIVE)

    assert again is once
    assert names == []
