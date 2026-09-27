"""A smoke test of the TUI's harness display, driven by Textual's pilot

Not part of the unit suite (the TUI, and so Textual, is an optional
group):  run it with

    uv run --group tui pytest --no-cov tests/tui

It serves the end-to-end test's scripted rooms (no LLM, no network) and
drives the real TUI against them:  the context meter, the compaction
notice and its collapsed full results, '/context', and a multi-line
prompt pasted whole.  With 'HARNESS_SHOTS_DIR' set, it saves SVG
screenshots there.
"""

from __future__ import annotations

import os
import pathlib
import time
from unittest import mock

import pytest

textual = pytest.importorskip("textual")

from textual import events as t_events  # noqa: E402
from textual import widgets as t_widgets  # noqa: E402

from soliplex.agui import client_tools  # noqa: E402
from soliplex.agui import harness  # noqa: E402
from soliplex.config import routing as config_routing  # noqa: E402
from soliplex.tui import main  # noqa: E402
from tests.unit.agui import harness_room  # noqa: E402
from tests.unit.agui import test_harness_e2e as e2e  # noqa: E402

pytestmark = pytest.mark.asyncio

SIZE = (140, 48)


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        mock.patch.dict(config_routing.__dict__) as patched_routing,
    ):
        patched_routing["APP_ROUTERS_BY_GROUP_NAME"] = {}
        monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
        monkeypatch.setenv(
            "LOGFIRE_CREDENTIALS_DIR",
            str(tmp_path_factory.mktemp("no-logfire")),
        )
        yield from e2e._serve(tmp_path_factory.mktemp("tui-pilot"))


def _app(url: str, tmp_path: pathlib.Path, **harness_kwargs):
    app = main.SoliplexTUI(
        soliplex_url=url,
        tool_context=client_tools.ToolContext(root=tmp_path),
        auto_approve=True,
        max_turns=harness_room.TURNS + 5,
        harness_options={
            "compaction": harness.CompactionPolicy(keep_recent=4),
            "on_report": harness.HarnessLog(tmp_path / "harness.jsonl").record,
            **harness_kwargs,
        },
    )
    # No command runs on this machine:  the scripted 'shell'.
    app.client_tools = e2e.TOOLS
    return app


async def _wait(pilot, condition, secs: float = 60.0) -> None:
    deadline = time.monotonic() + secs

    while not condition():
        if time.monotonic() > deadline:
            chat = [
                str(widget.source)[-600:]
                for widget in pilot.app.screen.query(main.Response)
            ]
            pytest.fail(f"timed out; the chat ends: {chat[-1:]}")

        await pilot.pause(0.1)


async def _enter_room(pilot, room_id: str) -> main.RoomView:
    await _wait(pilot, lambda: isinstance(pilot.app.screen, main.RoomListView))
    (button,) = [
        button
        for button in pilot.app.screen.query(t_widgets.Button)
        if button.name == room_id
    ]
    button.press()
    await _wait(pilot, lambda: isinstance(pilot.app.screen, main.RoomView))
    return pilot.app.screen


def _shot(app, name: str) -> None:
    shots = os.environ.get("HARNESS_SHOTS_DIR")

    if shots:
        pathlib.Path(shots).mkdir(parents=True, exist_ok=True)
        app.save_screenshot(filename=f"{name}.svg", path=shots)


async def _send(pilot, room: main.RoomView, text: str) -> None:
    prompt = room.query_one("#prompt", main.PromptArea)
    prompt.focus()
    prompt.post_message(t_events.Paste(text))
    await pilot.pause()
    await pilot.press("enter")


async def test_meter_and_compaction_notice(server_url, tmp_path):
    harness_room.reset(e2e.WINDOW_CHARS)
    app = _app(server_url, tmp_path, context_window=e2e.CONTEXT_WINDOW)

    async with app.run_test(size=SIZE) as pilot:
        room = await _enter_room(pilot, e2e.CHAIN_ROOM)
        meter = room.query_one("#context-meter", t_widgets.Static)
        assert str(meter.render()) == "context: not measured yet"

        await _send(pilot, room, "chain")
        await _wait(
            pilot,
            lambda: (
                room.run_agent_input is not None
                and "DONE r=20"
                in (room.run_agent_input.messages[-1].content or "")
            ),
        )
        await pilot.pause(0.5)

        # The meter:  a measured percentage of the window.
        text = str(meter.render())
        assert text.startswith("context ")
        assert "%" in text
        assert f"/ {e2e.CONTEXT_WINDOW:,})" in text
        reading = room.harness.meter.reading
        assert reading.fraction_used is not None

        # The compaction notice, and the full results, collapsed.
        notices = [
            report
            for report in room.harness.reports
            if harness.compaction_notice(report)
        ]
        assert notices
        blocks = room.query(".compacted-results")
        assert len(blocks) == len(notices)
        assert all(block.collapsed for block in blocks)
        _shot(app, "meter-and-compaction")

        # Expanded:  the full text of a compacted result.
        first = blocks.first()
        first.collapsed = False
        await pilot.pause(0.3)
        inner = first.query(t_widgets.Collapsible).first()
        inner.collapsed = False
        await pilot.pause(0.3)
        _shot(app, "compacted-result-expanded")

        # '/context':  the breakdown, sent nowhere.
        first.collapsed = True
        posts = len(room.harness.reports)
        await _send(pilot, room, "/context")
        await pilot.pause(0.5)
        room.query_one("#chat-view").scroll_end(animate=False)
        await pilot.pause(0.5)
        assert len(room.harness.reports) == posts
        (table,) = [
            widget
            for widget in room.query(main.Response)
            if "| what | messages |" in widget.source
        ]
        assert "tool result: big_search (compacted)" in table.source
        _shot(app, "context-command")

    logged = (tmp_path / "harness.jsonl").read_text(encoding="utf-8")
    assert "compacted" in logged


async def test_multi_line_prompt_sent_whole(server_url, tmp_path):
    harness_room.reset()
    app = _app(server_url, tmp_path)

    async with app.run_test(size=SIZE) as pilot:
        room = await _enter_room(pilot, e2e.QUESTIONS_ROOM)
        await _send(pilot, room, "line one\nline two\nline three")
        await _wait(pilot, lambda: room.run_agent_input is not None)
        await _wait(
            pilot,
            lambda: (
                "ANSWER" in (room.run_agent_input.messages[-1].content or "")
            ),
        )

        (user,) = [
            message
            for message in room.run_agent_input.messages
            if message.role == "user"
        ]
        assert user.content == "line one\nline two\nline three"

        # Ctrl+J starts a new line;  Enter sends.
        prompt = room.query_one("#prompt", main.PromptArea)
        prompt.focus()
        await pilot.press("a", "ctrl+j", "b")
        assert prompt.text == "a\nb"
        _shot(app, "multi-line-prompt")
