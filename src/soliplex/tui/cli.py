import pathlib
from importlib.metadata import version

import click
import typer
from rich import console

from soliplex.agui import client_tools
from soliplex.agui import harness as agui_harness
from soliplex.cli import harness_options
from soliplex.tui import main

the_cli = typer.Typer(
    context_settings={
        "help_option_names": ["-h", "--help"],
    },
    # no_args_is_help=True,
    add_completion=False,
)

the_console = console.Console()


def get_version():
    v = version("soliplex-tui")
    the_console.print(f"soliplex-tui version {v}")
    raise typer.Exit()


def version_callback(value: bool):
    if value:
        get_version()


BASE_URL = typer.Option(
    "http://127.0.0.1:8000",
    "--url",
    help="Base URL for Soliplex back-end",
)


ROOT = typer.Option(
    pathlib.Path("."),
    "--root",
    help=(
        "Working directory for client tool calls ('shell'), each of which "
        "you confirm first (unless auto-approve is on);  commands naming "
        "paths outside it are refused (a guard rail, not a sandbox)."
    ),
)


TOOL_LOG = typer.Option(
    None,
    "--tool-log",
    help=(
        "Append one JSON line per client tool call (what ran, its exit "
        "code, timing and output sizes) to this file."
    ),
)


@the_cli.command()
def tui(
    version: bool = typer.Option(None, "--version", "-V"),
    soliplex_url: str = BASE_URL,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    root: pathlib.Path = ROOT,
    allow_anywhere: bool = typer.Option(
        False,
        "--allow-anywhere",
        help="Do not refuse client tool calls naming paths outside --root.",
    ),
    tool_timeout: float = typer.Option(
        client_tools.DEFAULT_TOOL_TIMEOUT_SECS,
        "--tool-timeout",
        min=0.1,
        help="Seconds before a client tool call is killed.",
    ),
    client_tools_enabled: bool = typer.Option(
        True,
        "--client-tools/--no-client-tools",
        help=(
            "Advertise client tools ('shell') to rooms, to run on this "
            "machine after you confirm each call (unless auto-approve is "
            "on)."
        ),
    ),
    max_turns: int = typer.Option(
        client_tools.DEFAULT_MAX_TURNS,
        "--max-turns",
        min=1,
        help="The most runs (model turns) to make for one prompt.",
    ),
    pass_env: bool = typer.Option(
        False,
        "--pass-env",
        help=(
            "Run client tool calls with your full environment.  By default, "
            "variables named like secrets ('*TOKEN*', '*SECRET*', "
            "'*PASSWORD*', '*_KEY', 'AWS_*', 'GITHUB_*', ...) are left out; "
            " 'SOLIPLEX_TOKEN' always is."
        ),
    ),
    tool_log: pathlib.Path | None = TOOL_LOG,
    output_cap_bytes: int = harness_options.OUTPUT_CAP_BYTES,
    output_cap_mode: str = harness_options.OUTPUT_CAP_MODE,
    pairing_check: bool = harness_options.PAIRING_CHECK,
    compaction: str = harness_options.COMPACTION,
    keep_recent: int = harness_options.KEEP_RECENT,
    min_elide_chars: int = harness_options.MIN_ELIDE_CHARS,
    compaction_trigger: float = harness_options.COMPACTION_TRIGGER,
    compaction_target: float = harness_options.COMPACTION_TARGET,
    context_window: int | None = harness_options.CONTEXT_WINDOW,
    probe_model_window: bool = harness_options.PROBE_MODEL_WINDOW,
    output_reserve: int = harness_options.OUTPUT_RESERVE,
    trim_rag_state: str = harness_options.TRIM_RAG_STATE,
    harness_log: pathlib.Path | None = harness_options.HARNESS_LOG,
    input_mode: str = typer.Option(
        main.INPUT_MULTI,
        "--input-mode",
        click_type=click.Choice(main.INPUT_MODES),
        envvar="SOLIPLEX_TUI_INPUT_MODE",
        help=(
            "The prompt:  'multi' line (Enter sends, Ctrl+J or Alt+Enter a "
            "new line, pastes keep their newlines), or 'single' line."
        ),
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        "--yolo",
        help=(
            "Run client tool calls WITHOUT asking first (the path check, "
            "the timeout and the scrubbed environment still apply)."
        ),
    ),
):
    try:
        options = harness_options.harness_options(
            pairing_check=pairing_check,
            compaction=compaction,
            keep_recent=keep_recent,
            min_elide_chars=min_elide_chars,
            compaction_trigger=compaction_trigger,
            compaction_target=compaction_target,
            context_window=context_window,
            probe_model_window=probe_model_window,
            output_reserve=output_reserve,
            trim_rag_state=trim_rag_state,
            harness_log=harness_log,
        )
        tool_context = client_tools.ToolContext(
            root=root,
            allow_anywhere=allow_anywhere,
            timeout_secs=tool_timeout,
            output_cap_bytes=output_cap_bytes,
            output_cap_mode=output_cap_mode,
            pass_env=pass_env,
        )
    except (OSError, ValueError) as exc:
        the_console.print(f"Error: {exc}")
        raise typer.Exit(1) from None

    if trim_rag_state == agui_harness.TRIM_AGGRESSIVE:
        the_console.print(
            f"Warning: {harness_options.AGGRESSIVE_TRIM_WARNING}",
        )

    tui_app = main.SoliplexTUI(
        soliplex_url=soliplex_url,
        verbose=verbose,
        tool_context=tool_context,
        client_tools_enabled=client_tools_enabled,
        max_turns=max_turns,
        tool_log=(
            client_tools.ToolLog(tool_log) if tool_log is not None else None
        ),
        auto_approve=auto_approve,
        harness_options=options,
        input_mode=input_mode,
    )

    tui_app.run()
