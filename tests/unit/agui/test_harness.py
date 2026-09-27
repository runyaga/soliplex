from __future__ import annotations

import json
from unittest import mock

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

    assert policy.mode == harness.COMPACTION_OFF
    assert policy.keep_recent == harness.DEFAULT_KEEP_RECENT
    assert policy.min_elide_chars == harness.DEFAULT_MIN_ELIDE_CHARS


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
    assert "Figure caption" not in found
    assert len(found) < 1000
    # Byte-stable:  the same result always compacts to the same text.
    assert harness.compact_content("search", content) == found


def test_compact_content_search_w_score_header():
    content = "[c-1] (score: 0.87)\nContent:\n" + "text " * 400

    found = harness.compact_content("search", content)

    assert harness.compacted_info(found)["format"] == "headers"
    assert found.endswith("[c-1] (score: 0.87)")


@pytest.mark.parametrize(
    "content",
    [
        # A hit's first line is not a search header.
        "Not a search\nContent:\n" + "text " * 400,
        # A header, but no 'Content:'.
        "[c-1] [rank 1 of 1]\n" + "text " * 400,
    ],
)
def test_compact_content_not_a_search(content):
    found = harness.compact_content("search", content)

    assert harness.compacted_info(found)["format"] == "head_tail"


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


def _history(*results, calls=None):
    """A user prompt, then one assistant call and result per 'results'

    'results' are '(tool, content)';  'calls' may override a call.
    """
    messages = [_user()]

    for index, (tool, content) in enumerate(results):
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
    the_harness = harness.Harness()

    found = the_harness.before_post(mock.Mock(), run_input, first=True)

    assert found is run_input
    (report,) = the_harness.reports
    assert report.mode == "off"
    assert report.compacted == 0
