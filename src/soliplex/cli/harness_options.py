"""Context-harness options, shared by 'soliplex-tui' and 'ask --url'

Each can also be set through its 'SOLIPLEX_TUI_*' environment variable.
See 'soliplex.agui.harness', and 'docs/tui_harness.md'.
"""

from __future__ import annotations

import pathlib

import click
import typer

from soliplex.agui import client_tools
from soliplex.agui import harness as agui_harness

OUTPUT_CAP_BYTES = typer.Option(
    client_tools.DEFAULT_OUTPUT_CAP_BYTES,
    "--output-cap-bytes",
    min=client_tools.MIN_OUTPUT_CAP_BYTES,
    envvar="SOLIPLEX_TUI_OUTPUT_CAP_BYTES",
    help=(
        "The most bytes of each output stream (stdout, stderr) of a "
        "client tool call sent to the model."
    ),
)

OUTPUT_CAP_MODE = typer.Option(
    client_tools.DEFAULT_OUTPUT_CAP_MODE,
    "--output-cap-mode",
    click_type=click.Choice(client_tools.OUTPUT_CAP_MODES),
    envvar="SOLIPLEX_TUI_OUTPUT_CAP_MODE",
    help=(
        "How output over --output-cap-bytes is cut:  keep its start "
        "('head'), or its start and its end ('head_tail')."
    ),
)

PAIRING_CHECK = typer.Option(
    True,
    "--pairing-check/--no-pairing-check",
    envvar="SOLIPLEX_TUI_PAIRING_CHECK",
    help=(
        "Refuse to send a history whose tool calls and results do not "
        "pair up (which the server would fail)."
    ),
)

COMPACTION = typer.Option(
    agui_harness.COMPACTION_AUTO,
    "--compaction",
    click_type=click.Choice(agui_harness.COMPACTION_MODES),
    envvar="SOLIPLEX_TUI_COMPACTION",
    help=(
        "Compact old, large tool results in the history before each run: "
        " 'auto' once it nears the context window (in one batch), "
        "'always', or 'off'."
    ),
)

KEEP_RECENT = typer.Option(
    agui_harness.DEFAULT_KEEP_RECENT,
    "--keep-recent",
    min=0,
    envvar="SOLIPLEX_TUI_KEEP_RECENT",
    help="The newest large tool results, never compacted.",
)

MIN_ELIDE_CHARS = typer.Option(
    agui_harness.DEFAULT_MIN_ELIDE_CHARS,
    "--min-elide-chars",
    min=1,
    envvar="SOLIPLEX_TUI_MIN_ELIDE_CHARS",
    help="Tool results shorter than this are never compacted.",
)

COMPACTION_TRIGGER = typer.Option(
    agui_harness.DEFAULT_TRIGGER_FRACTION,
    "--compaction-trigger",
    click_type=click.FloatRange(0.0, 1.0, min_open=True),
    envvar="SOLIPLEX_TUI_COMPACTION_TRIGGER",
    help=(
        "'auto':  compact once the next request is estimated over this "
        "fraction of the usable context window..."
    ),
)

COMPACTION_TARGET = typer.Option(
    agui_harness.DEFAULT_TARGET_FRACTION,
    "--compaction-target",
    click_type=click.FloatRange(0.0, 1.0, min_open=True),
    envvar="SOLIPLEX_TUI_COMPACTION_TARGET",
    help="...down to under this fraction (less than --compaction-trigger).",
)

CONTEXT_WINDOW = typer.Option(
    None,
    "--context-window",
    min=1,
    envvar="SOLIPLEX_TUI_CONTEXT_WINDOW",
    help=(
        "The model's context window, in tokens (default: the room's "
        "'context_window', if it declares one)."
    ),
)

PROBE_MODEL_WINDOW = typer.Option(
    False,
    "--probe-model-window",
    envvar="SOLIPLEX_TUI_PROBE_MODEL_WINDOW",
    help=(
        "With no window from --context-window or the room, ask the room's "
        "model server ('GET /v1/models', 'max_model_len') directly."
    ),
)

OUTPUT_RESERVE = typer.Option(
    agui_harness.DEFAULT_OUTPUT_RESERVE,
    "--output-reserve",
    min=0,
    envvar="SOLIPLEX_TUI_OUTPUT_RESERVE",
    help="Tokens of the context window kept for the model's reply.",
)


TRIM_RAG_STATE = typer.Option(
    agui_harness.TRIM_BOUNDARY,
    "--trim-rag-state",
    click_type=click.Choice(agui_harness.TRIM_MODES),
    envvar="SOLIPLEX_TUI_TRIM_RAG_STATE",
    help=(
        "Drop haiku.rag's working evidence (search results) from the state "
        "sent:  once a question is answered ('boundary', lossless), also "
        "mid-question ('aggressive', lossy), or never ('off')."
    ),
)

HARNESS_LOG = typer.Option(
    None,
    "--harness-log",
    dir_okay=False,
    envvar="SOLIPLEX_TUI_HARNESS_LOG",
    help=(
        "Append one JSON line per run's POST (the history's size, what was "
        "compacted or trimmed, the estimate) to this file."
    ),
)

#   Shown when '--trim-rag-state aggressive' is chosen.
AGGRESSIVE_TRIM_WARNING = (
    "--trim-rag-state aggressive drops search results mid-question:  "
    "'cite' can then no longer correct a mangled chunk id, and its "
    "citations lose their expanded text."
)


def harness_options(
    *,
    pairing_check: bool,
    compaction: str,
    keep_recent: int,
    min_elide_chars: int,
    compaction_trigger: float,
    compaction_target: float,
    context_window: int | None,
    probe_model_window: bool,
    output_reserve: int,
    trim_rag_state: str = agui_harness.TRIM_BOUNDARY,
    harness_log: pathlib.Path | None = None,
) -> dict:
    """'agui_harness.make_harness' options, from the command line's

    With 'harness_log', each POST's report is appended to it
    ('on_report').  Raises 'agui_harness.InvalidCompactionFractions' if
    the target is not below the trigger.
    """
    options = {
        "pairing_check": pairing_check,
        "compaction": agui_harness.CompactionPolicy(
            mode=compaction,
            keep_recent=keep_recent,
            min_elide_chars=min_elide_chars,
            trigger_fraction=compaction_trigger,
            target_fraction=compaction_target,
        ),
        "context_window": context_window,
        "probe_window": probe_model_window,
        "output_reserve": output_reserve,
        "trim_rag_state": trim_rag_state,
    }

    if harness_log is not None:
        options["on_report"] = agui_harness.HarnessLog(harness_log).record

    return options
