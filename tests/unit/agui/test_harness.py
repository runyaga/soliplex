from __future__ import annotations

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
