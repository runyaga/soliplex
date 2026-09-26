from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import click
import pytest

from soliplex import loggers
from soliplex.agui import client_tools
from soliplex.cli import ask as cli_ask


@pytest.fixture
def no_cli_logging():
    # 'ask' calls the process-global one-shot
    # 'cli_util._configure_cli_logging', which would replace the
    # 'soliplex-audit' handlers (dropping the 'audit_records' capture) and
    # silence the logger for the rest of the run. Neutralize it here so the
    # emitted audit records stay observable and the global state is left
    # untouched.
    with mock.patch("soliplex.cli.cli_util._configure_cli_logging") as patched:
        yield patched


def _invoke(cli_runner, scratch_installation, *args):
    # 'ask.app' has a single command, which Typer promotes to the top
    # level, so the command name is not part of the invocation. Testing
    # the module's own 'app' (rather than 'cli.main') mirrors the other
    # CLI suites, e.g. 'test_cli_audit_package.py' invoking 'audit.app'.
    return cli_runner.invoke(
        cli_ask.app,
        [str(scratch_installation.path), *args],
    )


def _records(audit_records, message):
    return [
        record for record in audit_records if record.getMessage() == message
    ]


def _access_records(audit_records):
    return _records(audit_records, loggers.AUDIT_ROOM_ACCESS)


def _run_records(audit_records):
    return _records(audit_records, loggers.AUDIT_ROOM_AGENT_RUN)


def test_ask_plain_success(
    no_cli_logging,
    cli_runner,
    scratch_installation,
    audit_records,
):
    result = _invoke(cli_runner, scratch_installation, "faux", "what is up?")

    assert result.exit_code == 0
    assert "I don't know!" in result.output
    # The access is recorded before the run; its success after.
    access = _access_records(audit_records)
    assert access
    assert access[-1].outcome == loggers.AUDIT_OUTCOME_SUCCESS
    assert access[-1].room_id == "faux"
    run = _run_records(audit_records)
    assert run
    assert run[-1].outcome == loggers.AUDIT_OUTCOME_SUCCESS


def test_ask_json_success(no_cli_logging, cli_runner, scratch_installation):
    result = _invoke(
        cli_runner,
        scratch_installation,
        "faux",
        "what is up?",
        "--json",
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["room_id"] == "faux"
    assert payload["prompt"] == "what is up?"
    assert payload["response"] == "I don't know!"
    assert payload["thread_id"]
    # The faux agent contacts no model, so no token usage is reported.
    assert payload["usage"] == {}


def test_ask_plain_failure_reports_on_stderr(
    no_cli_logging,
    cli_runner,
    scratch_installation,
    audit_records,
):
    result = _invoke(cli_runner, scratch_installation, "faux", "fail")

    assert result.exit_code == 1
    assert "failing on request" in result.stderr
    # Access recorded (success) before the run; the failure is the run event.
    assert _access_records(audit_records)[-1].outcome == (
        loggers.AUDIT_OUTCOME_SUCCESS
    )
    run = _run_records(audit_records)
    assert run
    assert run[-1].outcome == loggers.AUDIT_OUTCOME_ERROR


def test_ask_json_failure_reports_error_object(
    no_cli_logging,
    cli_runner,
    scratch_installation,
):
    result = _invoke(
        cli_runner,
        scratch_installation,
        "faux",
        "fail",
        "--json",
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr) == {"error": "failing on request"}


def test_ask_unknown_room_fails(
    no_cli_logging,
    cli_runner,
    scratch_installation,
):
    result = _invoke(cli_runner, scratch_installation, "nope", "hi")

    assert result.exit_code == 1
    assert "No room configured with id 'nope'" in result.stderr


def test_ask_configures_cli_logging(
    no_cli_logging,
    cli_runner,
    scratch_installation,
):
    result = _invoke(cli_runner, scratch_installation, "faux", "hi")

    assert result.exit_code == 0
    no_cli_logging.assert_called_once()


def test_ask_reports_run_exception(
    no_cli_logging,
    cli_runner,
    scratch_installation,
    monkeypatch,
):
    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli_ask, "_run_ask", _boom)

    result = _invoke(cli_runner, scratch_installation, "faux", "hi")

    assert result.exit_code == 1
    assert "kaboom" in result.stderr


def test_ask_reports_missing_result(
    no_cli_logging,
    cli_runner,
    scratch_installation,
    audit_records,
):
    async def _one_nonterminal(**kwargs):
        # A stream that ends without a RUN_FINISHED / RUN_ERROR event.
        yield SimpleNamespace(type="SOMETHING_ELSE")

    with mock.patch.object(
        cli_ask.agui_persistence,
        "drive_agui_turn",
        _one_nonterminal,
    ):
        result = _invoke(cli_runner, scratch_installation, "faux", "hi")

    assert result.exit_code == 1
    assert "no result" in result.stderr
    assert _access_records(audit_records)[-1].outcome == (
        loggers.AUDIT_OUTCOME_SUCCESS
    )
    run = _run_records(audit_records)
    assert run
    assert run[-1].outcome == loggers.AUDIT_OUTCOME_ERROR


class Boom(RuntimeError):
    def __init__(self):
        super().__init__("stream boom")


def test_ask_audits_run_exception(
    no_cli_logging,
    cli_runner,
    scratch_installation,
    audit_records,
    monkeypatch,
):
    def _boom(**kwargs):
        raise Boom()

    monkeypatch.setattr(cli_ask.agui_persistence, "drive_agui_turn", _boom)

    result = _invoke(cli_runner, scratch_installation, "faux", "hi")

    assert result.exit_code == 1
    assert "stream boom" in result.stderr
    # Access recorded before the run; an exception mid-run is a run failure.
    assert _access_records(audit_records)[-1].outcome == (
        loggers.AUDIT_OUTCOME_SUCCESS
    )
    run = _run_records(audit_records)
    assert run
    assert run[-1].outcome == loggers.AUDIT_OUTCOME_ERROR


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, {}),
        (
            SimpleNamespace(input_tokens=3, output_tokens=5),
            {"input_tokens": 3, "output_tokens": 5},
        ),
    ],
)
def test_usage_as_dict(usage, expected):
    result = cli_ask._usage_as_dict(usage)

    assert result == expected


# -- 'ask' without '--url':  arguments ---------------------------------------


@pytest.mark.parametrize(
    "args, missing",
    [
        ([], "INSTALLATION_PATH"),
        (["installation.yaml"], "ROOM_ID"),
        (["installation.yaml", "faux"], "PROMPT"),
    ],
)
def test_ask_local_w_missing_argument(cli_runner, monkeypatch, args, missing):
    monkeypatch.delenv("SOLIPLEX_INSTALLATION_PATH", raising=False)

    result = cli_runner.invoke(cli_ask.app, args)

    assert result.exit_code == 2
    assert f"Missing argument '{missing}'" in result.stderr


# -- 'ask --url' -------------------------------------------------------------

URL = "http://server:8000"

REMOTE_RESULT = client_tools.LoopResult(
    thread_id="thread-1",
    run_ids=["run-1", "run-2"],
    response="The answer.",
    tool_calls=[
        client_tools.ToolCallRecord(
            name="shell",
            args={"command": "ls"},
            exit_code=0,
        ),
    ],
)


@pytest.fixture
def ask_remote():
    with mock.patch.object(
        cli_ask,
        "_ask_remote",
        return_value=REMOTE_RESULT,
    ) as patched:
        yield patched


@pytest.mark.parametrize("args", [["room"], ["install.yaml", "room", "hi"]])
def test_ask_remote_w_bad_arguments(cli_runner, ask_remote, args):
    result = cli_runner.invoke(cli_ask.app, ["--url", URL, *args])

    assert result.exit_code == 2
    assert "With '--url'" in result.stderr
    ask_remote.assert_not_called()


@pytest.mark.parametrize("json_output", [False, True])
def test_ask_remote_defaults(
    cli_runner,
    ask_remote,
    monkeypatch,
    tmp_path,
    json_output,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOLIPLEX_TOKEN", raising=False)
    flags = ["--json"] if json_output else []

    result = cli_runner.invoke(
        cli_ask.app,
        ["--url", URL, "a-room", "a prompt", *flags],
    )

    assert result.exit_code == 0, result.output
    if json_output:
        assert json.loads(result.stdout) == REMOTE_RESULT.as_json()
    else:
        assert result.stdout == "The answer.\n"

    kwargs = ask_remote.call_args.kwargs
    context = kwargs.pop("context")
    assert kwargs == {
        "url": URL,
        "room_id": "a-room",
        "prompt": "a prompt",
        "token": None,
        "max_turns": client_tools.DEFAULT_MAX_TURNS,
        "confirm": False,
        "tool_log": None,
    }
    assert context == client_tools.ToolContext(root=tmp_path)
    assert context.pass_env is False  # scrubbed environment by default


def test_ask_remote_w_options(cli_runner, ask_remote, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    log_path = tmp_path / "tools.jsonl"

    result = cli_runner.invoke(
        cli_ask.app,
        [
            "a-room",
            "a prompt",
            "--url",
            URL,
            "--token",
            "secret",
            "--root",
            str(root),
            "--allow-anywhere",
            "--max-turns",
            "3",
            "--tool-timeout",
            "5",
            "--confirm",
            "--tool-log",
            str(log_path),
            "--pass-env",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = ask_remote.call_args.kwargs
    assert kwargs["token"] == "secret"
    assert kwargs["max_turns"] == 3
    assert kwargs["confirm"] is True
    assert kwargs["tool_log"] == log_path
    assert kwargs["context"] == client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        timeout_secs=5.0,
        pass_env=True,
    )


def test_ask_remote_w_token_from_env(cli_runner, ask_remote, monkeypatch):
    monkeypatch.setenv("SOLIPLEX_TOKEN", "from-env")

    result = cli_runner.invoke(cli_ask.app, ["--url", URL, "room", "hi"])

    assert result.exit_code == 0, result.output
    assert ask_remote.call_args.kwargs["token"] == "from-env"


@pytest.mark.parametrize("json_output", [False, True])
def test_ask_remote_w_error(cli_runner, ask_remote, json_output):
    ask_remote.side_effect = client_tools.RunErrored("boom")
    flags = ["--json"] if json_output else []

    result = cli_runner.invoke(
        cli_ask.app,
        ["--url", URL, "room", "hi", *flags],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    if json_output:
        assert json.loads(result.stderr) == {"error": "Run failed: boom"}
    else:
        assert "Error: Run failed: boom" in result.stderr


def test_ask_remote_w_bad_root(cli_runner, ask_remote, tmp_path):
    result = cli_runner.invoke(
        cli_ask.app,
        ["--url", URL, "room", "hi", "--root", str(tmp_path / "nonesuch")],
    )

    assert result.exit_code == 1
    assert "nonesuch" in result.stderr
    ask_remote.assert_not_called()


@pytest.mark.parametrize("w_confirm", [False, True])
@pytest.mark.parametrize("w_tool_log", [False, True])
def test__ask_remote(tmp_path, w_confirm, w_tool_log):
    context = client_tools.ToolContext(root=tmp_path)
    tool_log = tmp_path / "tools.jsonl" if w_tool_log else None
    thread = {"thread_id": "thread-1", "runs": {"run-1": {}}}

    with (
        mock.patch.object(client_tools, "SoliplexClient") as client_klass,
        mock.patch.object(client_tools, "run_loop") as run_loop,
    ):
        client = client_klass.return_value.__enter__.return_value
        client.new_thread.return_value = thread

        found = cli_ask._ask_remote(
            url=URL,
            room_id="room",
            prompt="hi",
            token="secret",
            context=context,
            max_turns=4,
            confirm=w_confirm,
            tool_log=tool_log,
        )

    assert found is run_loop.return_value
    client_klass.assert_called_once_with(URL, "room", token="secret")
    (client_arg, run_input, context_arg), kwargs = run_loop.call_args
    assert client_arg is client
    assert context_arg is context
    assert run_input.thread_id == "thread-1"
    assert run_input.run_id == "run-1"
    assert run_input.messages[0].content == "hi"
    assert kwargs["max_turns"] == 4
    assert kwargs["max_turns_option"] == "--max-turns"

    if w_confirm:
        assert kwargs["confirm"] is cli_ask._confirm_tool_call
    else:
        assert kwargs["confirm"] is None

    if w_tool_log:
        assert kwargs["tool_log"] == client_tools.ToolLog(tool_log)
    else:
        assert kwargs["tool_log"] is None


@pytest.mark.parametrize(
    "name, args, shown",
    [
        ("shell", {"command": "ls -la"}, "ls -la"),
        ("other", {"a": 1}, '{"a": 1}'),
    ],
)
@pytest.mark.parametrize("answer", [False, True])
def test_confirm_tool_call(name, args, shown, answer):
    with mock.patch.object(
        cli_ask.typer,
        "confirm",
        return_value=answer,
    ) as confirm:
        found = cli_ask._confirm_tool_call(name, args)

    assert found is answer
    confirm.assert_called_once_with(f"Run {name}: {shown}?", err=True)


def test_confirm_tool_call_w_abort():
    with mock.patch.object(
        cli_ask.typer,
        "confirm",
        side_effect=click.Abort(),
    ):
        assert cli_ask._confirm_tool_call("shell", {"command": "ls"}) is False
