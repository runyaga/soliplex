import pathlib
from importlib.metadata import version

import click
import typer
from rich import console

from soliplex.agui import client_tools
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
    output_cap_bytes: int = OUTPUT_CAP_BYTES,
    output_cap_mode: str = OUTPUT_CAP_MODE,
    pairing_check: bool = typer.Option(
        True,
        "--pairing-check/--no-pairing-check",
        envvar="SOLIPLEX_TUI_PAIRING_CHECK",
        help=(
            "Refuse to send a history whose tool calls and results do not "
            "pair up (which the server would fail)."
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
        harness_options={"pairing_check": pairing_check},
    )

    tui_app.run()
