from __future__ import annotations

import _thread
import asyncio
import dataclasses
import json
import os
import pathlib
import signal
import socket
import sys
import threading
import time
import types
from unittest import mock

import httpx
import markdown_it
import pytest
from ag_ui import core as agui_core
from fastapi import testclient
from pydantic_ai import messages as ai_messages

from soliplex import main
from soliplex.agui import client_tools
from soliplex.config import routing as config_routing
from tests._dburi import sqlite_dburi
from tests.unit.agui import scripted_shell_room

THREAD_ID = "thread-1"
RUN_ID = "run-1"
CHILD_RUN_ID = "run-2"

PYTHON = f'"{sys.executable}"'


@pytest.fixture
def root(tmp_path):
    the_root = tmp_path / "root"
    the_root.mkdir()
    return the_root.resolve()


@pytest.fixture
def context(root):
    return client_tools.ToolContext(root=root)


def _script(root, name, source):
    (root / name).write_text(source, encoding="utf-8")
    return f"{PYTHON} {name}"


# -- ToolContext -----------------------------------------------------------


def test_toolcontext_resolves_root(root, monkeypatch):
    monkeypatch.chdir(root)

    found = client_tools.ToolContext(root=pathlib.Path("."))

    assert found.root == root
    assert found.allow_anywhere is False
    assert found.timeout_secs == client_tools.DEFAULT_TOOL_TIMEOUT_SECS
    assert found.output_cap_bytes == client_tools.DEFAULT_OUTPUT_CAP_BYTES


def test_toolcontext_w_file_root(root):
    a_file = root / "file.txt"
    a_file.write_text("x", encoding="utf-8")

    with pytest.raises(client_tools.RootNotADirectory, match="file.txt"):
        client_tools.ToolContext(root=a_file)


def test_toolcontext_w_missing_root(root):
    with pytest.raises(FileNotFoundError):
        client_tools.ToolContext(root=root / "nonesuch")


# -- check_command_paths ---------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "cat /outside/file",
        "cat '/outside/a b'",
        "cat </outside/file",
        "echo hi >/outside/file",
        "echo hi;/outside/program",
        "cat --file=/outside/file",
        "python -c \"open('/outside/file')\"",
        'sh -c "cat /outside/file"',
        "cat /outside/../file",
        "ls ~",
        "cat ~/secret",
        "cat $SOLIPLEX_TEST_OUTSIDE/file",
        "cat ../sibling",
        "cat sub/../../sibling",
        "cp --from=../sibling x",
        "ls ~nonesuch-user-zzz/file",
    ],
)
def test_check_command_paths_refused(root, monkeypatch, command):
    monkeypatch.setenv("SOLIPLEX_TEST_OUTSIDE", "/outside")

    with pytest.raises(client_tools.PathRefused, match="outside --root"):
        client_tools.check_command_paths(command, root)


def test_check_command_paths_w_symlink_out(root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()

    with mock.patch.object(pathlib.Path, "resolve", return_value=outside):
        with pytest.raises(client_tools.PathRefused):
            client_tools.check_command_paths(f"cat {root}/link", root)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat file.txt 2>/dev/null",
        "cat sub/../file.txt",
        "git log HEAD~1",
        "echo a/b",
        "curl https://example.com/x",
    ],
)
def test_check_command_paths_allowed(root, command):
    client_tools.check_command_paths(command, root)


def test_check_command_paths_w_absolute_inside(root):
    inside = root / "a file.txt"

    client_tools.check_command_paths(f'cat "{inside}"', root)


def test_check_command_paths_w_unclosed_quote(root):
    with pytest.raises(ValueError, match="No closing quotation"):
        client_tools.check_command_paths("echo 'oops", root)


def test_check_command_paths_w_windows_lexing(root, monkeypatch):
    monkeypatch.setattr(client_tools, "_POSIX", False)

    client_tools.check_command_paths(f'type "{root}/a file.txt"', root)

    with pytest.raises(client_tools.PathRefused):
        client_tools.check_command_paths('type "/outside/file"', root)

    # 'cmd.exe' switches are not paths.
    client_tools.check_command_paths("dir /s /b", root)


def test_check_command_paths_w_posix_slash_word(root, monkeypatch):
    monkeypatch.setattr(client_tools, "_POSIX", True)

    with pytest.raises(client_tools.PathRefused):
        client_tools.check_command_paths("dir /s", root)


def test_check_command_paths_w_resolve_error(root):
    with mock.patch.object(pathlib.Path, "resolve", side_effect=OSError):
        with pytest.raises(client_tools.PathRefused):
            client_tools.check_command_paths("cat ../loop", root)


# -- check_command_paths:  here-documents, URLs, 'cd' ----------------------

FLASK_APP = """\
from flask import Flask

app = Flask(__name__)


@app.route('/contacts')
def contacts():
    return open('/etc/hosts').read()
"""


@pytest.fixture
def posix_lexing(monkeypatch):
    monkeypatch.setattr(client_tools, "_POSIX", True)


def test_check_command_paths_w_heredoc_writing_a_file(root, posix_lexing):
    # The user's session:  a heredoc writing a Flask app into the root.
    # The body names '/contacts' and '/etc/hosts', but it is data.
    app_dir = root.as_posix() + "/crm"
    command = (
        f"mkdir -p {app_dir} && cat > {app_dir}/app.py <<'PY'\n"
        f"{FLASK_APP}"
        "PY\n"
        f"python3 {app_dir}/app.py"
    )

    client_tools.check_command_paths(command, root)


@pytest.mark.parametrize(
    "command",
    [
        "cat > app.py <<'PY'\n/etc/passwd\nPY",
        'cat > app.py <<"PY"\n/etc/passwd\nPY',
        "cat > app.py <<\\PY\n/etc/passwd\nPY",
        "cat > app.py << PY\n/etc/passwd\nPY",
        "cat > app.py <<PY\n/etc/passwd\nPY\nls",
        "cat > app.py <<-PY\n\t/etc/passwd\n\tPY\nls",
        "cat <<A <<B\n/etc/a\nA\n/etc/b\nB\nls",
        "cat <<'PY' > x\n$(cat /etc/passwd)\nPY",  # quoted:  no expansion
        "echo $((1 << 2))",
        "echo 'a <<EOF' \"b <<EOF\"",
        "echo a\\ b",
        "ls \\\n  sub",
        "curl 'http://localhost:5000/contacts'",
        'curl -X POST -d "name=a" "http://127.0.0.1:5000/contacts/1"',
        "curl localhost:5000/contacts",
    ],
)
def test_check_command_paths_allowed_w_heredoc_or_url(
    root,
    posix_lexing,
    command,
):
    client_tools.check_command_paths(command, root)


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd",
        'cat "/etc/passwd"',
        "cat > /outside/x <<EOF\nhello\nEOF",
        "cat <<EOF > /outside/x\nhello\nEOF",
        "cat <<'EOF' >/outside/x\nhello\nEOF",
        # An unquoted here-document's command substitutions run.
        "cat <<EOF > x\n$(cat /etc/passwd)\nEOF",
        "cat <<EOF > x\n`cat /etc/passwd`\nEOF",
        # The command goes on after the body.
        "cat <<EOF\nhello\nEOF\ncat /etc/passwd",
        # Not here-documents.
        "echo $((1 << 2))\ncat /etc/passwd",
        "echo '<<EOF'\ncat /etc/passwd",
        "cat <<< /etc/passwd",
        "cat << ;cat /etc/passwd",
        "(( x\ncat /etc/passwd",
        "ls \\\n /etc",
        # Comments end at the newline;  no heredoc starts in one.
        "echo ok # comment\ncat /etc/passwd",
        "echo ok # <<EOF\ncat /etc/passwd\nEOF",
        "echo a#b /etc/passwd",
        "echo a\\ #b; cat /etc/passwd",
        "echo x\\\n#y /etc/passwd",
        # A here-string is not a here-document.
        "cat <<< hello\ncat /etc/passwd",
        # Delimiters, as the shell reads them.
        "cat <<'E\"F'\nx\nE\"F\ncat /etc/passwd\nEF",
        "cat <<EOF\ndata\nE\\\nOF\ncat /etc/passwd",
        # A multi-line command substitution in an unquoted body.
        "cat <<EOF\n$(\ncat /etc/passwd\n)\nEOF",
        "cat <<EOF\n`\ncat /etc/passwd\n`\nEOF",
        # Conservative:  a quoted '/x' may be a route or a file;  refused.
        "grep \"route('/contacts')\" app.py",
        "python3 -c \"print(open('/etc/passwd').read())\"",
    ],
)
def test_check_command_paths_refused_w_heredoc_or_quotes(
    root,
    posix_lexing,
    command,
):
    with pytest.raises(client_tools.PathRefused, match="outside --root"):
        client_tools.check_command_paths(command, root)


@pytest.mark.parametrize(
    "command",
    ["cat > app.py <<PY\nhello", "cat > app.py <<PY", "cat <<A <<B\nA\n"],
)
def test_check_command_paths_w_unterminated_heredoc(
    root,
    posix_lexing,
    command,
):
    # Refused, rather than take the rest of the command for the body.
    with pytest.raises(client_tools.UnterminatedHereDoc, match="PY|B"):
        client_tools.check_command_paths(command, root)


@pytest.mark.parametrize(
    "command, path",
    [
        ("cd", "$HOME"),
        ("cd && ls", "$HOME"),
        ("cd; ls", "$HOME"),
        ("cd\nls", "$HOME"),
        ("ls; cd -P", "$HOME"),
        ("cd --", "$HOME"),
        ("pushd", "$HOME"),
        ("cd -", "$OLDPWD"),
        ("cd -L -", "$OLDPWD"),
        ("if cd; then ls; fi", "$HOME"),
        ("cd > out.txt", "$HOME"),
        ("ls | { cd; }", "$HOME"),
        ("cd 2>/dev/null; pwd", "$HOME"),
        ("> /dev/null cd; pwd", "$HOME"),
        ("2>/dev/null cd", "$HOME"),
        ("HOME=/x cd", "$HOME"),
        ("! cd", "$HOME"),
        # Quoted operators are words, not operators.
        ("echo '>'; cd; pwd", "$HOME"),
        ('echo ">"; cd', "$HOME"),
        ("echo \\>; cd", "$HOME"),
        ("cat <<'a;b'\nx\na;b\ncd", "$HOME"),
        # Operators come from the command, not from the variables in it.
        ("echo $SOLIPLEX_TEST_OP; cd", "$HOME"),
    ],
)
def test_check_command_paths_refuses_cd_home(
    root,
    posix_lexing,
    monkeypatch,
    command,
    path,
):
    monkeypatch.setenv("SOLIPLEX_TEST_OP", ">")

    with pytest.raises(client_tools.CdRefused, match="outside --root") as ei:
        client_tools.check_command_paths(command, root)

    assert ei.value.path == path


@pytest.mark.parametrize(
    "command",
    [
        "cd sub",
        "cd -P sub && ls",
        "cd .",
        "cd > out.txt sub",
        # Not in a command's place.
        "echo cd",
        "printf '%s' pushd",
        "grep -w cd notes.txt",
        "echo if cd",
        "echo then pushd",
        "ls # cd",
        "echo ';' cd",
        "echo \\; cd",
        "echo $SOLIPLEX_TEST_OP cd",
    ],
)
def test_check_command_paths_allows_cd_inside(
    root,
    posix_lexing,
    monkeypatch,
    command,
):
    monkeypatch.setenv("SOLIPLEX_TEST_OP", ";")

    client_tools.check_command_paths(command, root)


def test_check_command_paths_w_windows_cd(root, monkeypatch):
    monkeypatch.setattr(client_tools, "_POSIX", False)

    client_tools.check_command_paths("cd", root)  # 'cmd.exe':  prints it


def test_check_command_paths_w_sibling_of_root(tmp_path):
    # '--root' a subdirectory:  a sibling directory, even one whose name
    # starts with the root's, is outside it.
    tmp = tmp_path / "tmp"
    root = tmp / "crm"
    root.mkdir(parents=True)
    root = root.resolve()
    outside = (tmp.resolve() / "crm_app").as_posix()

    for command in [
        f"mkdir -p {outside} && cd {outside} && cat > app.py <<'PY'\nPY",
        f"cd {outside}",
        f"echo x > {outside}/requirements.txt",
        "mkdir -p ../crm_app",
    ]:
        with pytest.raises(client_tools.PathRefused, match="crm_app"):
            client_tools.check_command_paths(command, root)

    client_tools.check_command_paths(f"mkdir -p {root.as_posix()}/app", root)


# -- run_shell ---------------------------------------------------------------


@pytest.mark.parametrize("args", [{}, {"command": 1}, {"command": "  "}])
def test_run_shell_w_invalid_args(context, args):
    with pytest.raises(client_tools.InvalidShellArgs):
        client_tools.run_shell(args, context)


def test_run_shell_refuses_before_running(context):
    with mock.patch.object(client_tools.subprocess, "Popen") as popen:
        with pytest.raises(client_tools.PathRefused):
            client_tools.run_shell({"command": "cat /outside/file"}, context)

    popen.assert_not_called()


def test_run_shell_output_cwd_stdin_exit_code(root):
    command = _script(
        root,
        "probe.py",
        "import os, sys\n"
        "print(os.getcwd())\n"
        "print(repr(sys.stdin.read()))\n"
        "print('to-stderr', file=sys.stderr)\n"
        "sys.exit(7)\n",
    )
    # The interpreter lives outside the root.
    context = client_tools.ToolContext(root=root, allow_anywhere=True)

    found = client_tools.run_shell({"command": command}, context)

    cwd, stdin = found["stdout"].splitlines()
    assert pathlib.Path(cwd).resolve() == root
    assert stdin == "''"  # no stdin: reads EOF at once
    assert found["stderr"].strip() == "to-stderr"
    assert found["exit_code"] == 7
    assert found["timed_out"] is False
    assert found["truncated"] is False
    assert found["stdout_bytes"] == len(found["stdout"].encode())


def test_run_shell_caps_output(root):
    command = _script(root, "big.py", "print('x' * 100)\n")
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        output_cap_bytes=10,
    )

    found = client_tools.run_shell({"command": command}, context)

    assert found["stdout"] == "x" * 10
    assert found["stdout_bytes"] > 100
    assert found["truncated"] is True
    assert found["exit_code"] == 0


def test_run_shell_times_out_and_kills_the_command(root):
    # The shell runs 'parent.py', which runs 'child.py':  the timeout must
    # stop the grandchild, not just the shell, or 'late.txt' appears.
    (root / "child.py").write_text(
        "import pathlib, time\n"
        "pathlib.Path('started.txt').write_text('', encoding='utf-8')\n"
        "time.sleep(3.5)\n"
        "pathlib.Path('late.txt').write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )
    command = _script(
        root,
        "parent.py",
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, 'child.py'])\n",
    )
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        timeout_secs=2.5,
    )

    found = client_tools.run_shell({"command": command}, context)

    assert found["timed_out"] is True
    assert found["exit_code"] != 0
    assert "timed out after 2.5 seconds" in found["stderr"]

    # Else the test proves nothing:  the child never ran.
    assert (root / "started.txt").exists()

    time.sleep(4.0)  # longer than the child sleeps, however late it began
    assert not (root / "late.txt").exists()


@pytest.mark.parametrize("w_gone", [False, True])
def test_kill_posix(monkeypatch, w_gone):
    monkeypatch.setattr(client_tools, "_POSIX", True)
    fake_signal = types.SimpleNamespace(SIGKILL=9)
    monkeypatch.setattr(client_tools, "signal", fake_signal)
    killpg = mock.Mock(side_effect=ProcessLookupError if w_gone else None)
    monkeypatch.setattr(client_tools.os, "killpg", killpg, raising=False)
    proc = mock.Mock(pid=1234)

    client_tools._kill(proc)

    killpg.assert_called_once_with(1234, 9)
    proc.kill.assert_not_called()


def test_kill_windows(monkeypatch):
    monkeypatch.setattr(client_tools, "_POSIX", False)
    proc = mock.Mock(pid=1234)

    with mock.patch.object(client_tools.subprocess, "run") as run:
        client_tools._kill(proc)

    (argv,), kwargs = run.call_args
    assert argv == ["taskkill", "/F", "/T", "/PID", "1234"]
    assert kwargs["check"] is False
    proc.kill.assert_called_once_with()


# -- run_shell:  environment -------------------------------------------------


def test_child_environment_scrubs_secrets():
    environ = {
        "PATH": "/bin",
        "HOME": "/home/me",
        "LANG": "C.UTF-8",
        "TERM": "xterm",
        "SHELL": "/bin/sh",
        "TMPDIR": "/tmp",
        "SOLIPLEX_TOKEN": "t",
        "GITHUB_TOKEN": "t",
        "github_token": "t",
        "OPENAI_API_KEY": "k",
        "MY_API_KEY_FILE": "k",
        "GPG_KEY": "k",
        "DB_PASSWORD": "p",
        "PGPASSWD": "p",
        "CLIENT_SECRET": "s",
        "AWS_PROFILE": "x",
        "GH_HOST": "x",
        "HF_TOKEN_PATH": "x",
        "CDPATH": "/",
        "OLDPWD": "/",
        "KEYBOARD": "us",  # 'KEY' but not '*_KEY'
    }

    found = client_tools.child_environment(environ)

    assert found == {
        "PATH": "/bin",
        "HOME": "/home/me",
        "LANG": "C.UTF-8",
        "TERM": "xterm",
        "SHELL": "/bin/sh",
        "TMPDIR": "/tmp",
        "KEYBOARD": "us",
    }


def test_child_environment_w_pass_env():
    environ = {"PATH": "/bin", "SOLIPLEX_TOKEN": "t", "OPENAI_API_KEY": "k"}

    found = client_tools.child_environment(environ, pass_env=True)

    # 'SOLIPLEX_TOKEN' is dropped even so.
    assert found == {"PATH": "/bin", "OPENAI_API_KEY": "k"}


def test_child_environment_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("SOLIPLEX_TEST_KEEP", "yes")
    monkeypatch.setenv("SOLIPLEX_TOKEN", "t")

    found = client_tools.child_environment()

    assert found["SOLIPLEX_TEST_KEEP"] == "yes"
    assert "SOLIPLEX_TOKEN" not in found


@pytest.mark.parametrize("pass_env", [False, True])
def test_run_shell_environment(root, monkeypatch, pass_env):
    monkeypatch.setenv("SOLIPLEX_TOKEN", "the-token")
    monkeypatch.setenv("SOLIPLEX_TEST_API_KEY", "the-key")
    monkeypatch.setenv("SOLIPLEX_TEST_KEEP", "kept")
    command = _script(
        root,
        "env.py",
        "import json, os\nprint(json.dumps(dict(os.environ)))\n",
    )
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        pass_env=pass_env,
    )

    found = client_tools.run_shell({"command": command}, context)

    environ = json.loads(found["stdout"])
    assert "SOLIPLEX_TOKEN" not in environ
    assert ("SOLIPLEX_TEST_API_KEY" in environ) is pass_env
    assert environ["SOLIPLEX_TEST_KEEP"] == "kept"
    assert "PATH" in environ  # ordinary commands still work


# -- run_shell:  the command never outlives an interrupted wait ----------

#   'parent.py' runs 'child.py', which listens on a socket until killed,
#   and writes 'late.txt' if it lives for about 5 seconds.  Once
#   'port.txt' exists, the command is running;  once a connection to the
#   port is refused, the grandchild is dead (its socket closed with it).
_CHILD_SOURCE = """\
import pathlib, socket, time
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen(128)
port = pathlib.Path('port.txt')
port.with_suffix('.tmp').write_text(
    str(listener.getsockname()[1]), encoding='utf-8',
)
port.with_suffix('.tmp').replace(port)
time.sleep(5)
pathlib.Path('late.txt').write_text('late', encoding='utf-8')
"""


def _child_command(root) -> str:
    """The grandchild, run by the shell in the background (POSIX)

    The shell exits at once, leaving it running:  only a kill of the
    whole process group stops it.  ('cmd.exe' has no '&':  on Windows,
    the grandchild runs in the foreground, as '_grandchild_command'.)
    """
    foreground = _grandchild_command(root)
    return f"{PYTHON} child.py &" if os.name == "posix" else foreground


def _grandchild_command(root) -> str:
    (root / "child.py").write_text(_CHILD_SOURCE, encoding="utf-8")
    return _script(
        root,
        "parent.py",
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, 'child.py'])\n",
    )


def _child_started(root) -> bool:
    return (root / "port.txt").exists()


def _child_port(root) -> int:
    return int((root / "port.txt").read_text(encoding="utf-8"))


def _connection_refused(port: int) -> bool:
    """Is nothing listening on 'port'?  (Once the grandchild is dead.)"""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1.0).close()
    except OSError as exc:  # e.g. reset:  it is going
        return isinstance(exc, ConnectionRefusedError)

    return False


def _eventually(predicate, message, deadline_secs=10.0) -> None:
    deadline = time.monotonic() + deadline_secs
    while not predicate():
        assert time.monotonic() < deadline, message
        time.sleep(0.02)


def _wait_for_start(root) -> None:
    _eventually(lambda: _child_started(root), "the command never started")


def _assert_all_dead(root) -> None:
    assert _child_started(root)  # else the test proves nothing
    port = _child_port(root)

    _eventually(lambda: _connection_refused(port), "the grandchild is alive")
    assert not (root / "late.txt").exists()


def _started_then(root, outcome):
    """An 'is_cancelled' which, once the command started, does 'outcome'"""

    def is_cancelled():
        if not _child_started(root):
            return False

        # Else '_assert_all_dead' would prove nothing.
        assert not _connection_refused(_child_port(root))

        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return is_cancelled


def test_run_shell_w_cancel_kills_the_command(root):
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=_started_then(root, True),
    )

    with pytest.raises(client_tools.CommandCancelled, match="killed"):
        client_tools.run_shell({"command": _grandchild_command(root)}, context)

    _assert_all_dead(root)


@pytest.mark.parametrize(
    "exc_class",
    [KeyboardInterrupt, asyncio.CancelledError, RuntimeError],
)
def test_run_shell_w_exception_while_waiting_kills_the_command(
    root,
    exc_class,
):
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=_started_then(root, exc_class()),
    )

    with pytest.raises(exc_class):
        client_tools.run_shell({"command": _grandchild_command(root)}, context)

    _assert_all_dead(root)


@pytest.fixture
def sigint_raises_keyboard_interrupt():
    # As in a terminal:  pytest-xdist workers may ignore SIGINT.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("w_is_cancelled", [False, True])
def test_run_shell_w_ctrl_c_kills_the_command(
    root,
    sigint_raises_keyboard_interrupt,
    w_is_cancelled,
):
    # The default (as for 'ask'):  a long timeout, no 'is_cancelled'.
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=(lambda: False) if w_is_cancelled else None,
    )

    done = threading.Event()

    def interrupt_once_started():
        _eventually(
            lambda: _child_started(root) or done.is_set(),
            "the command never started",
        )

        # Ctrl-C, to this process -- unless 'run_shell' is done already
        # (it failed:  there is nothing to interrupt).
        done.is_set() or _thread.interrupt_main(signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_once_started)
    interrupter.start()

    def run():
        try:
            client_tools.run_shell(
                {"command": _grandchild_command(root)},
                context,
            )
        finally:
            done.set()

    try:
        with pytest.raises(KeyboardInterrupt):
            run()
    finally:
        interrupter.join()

    _assert_all_dead(root)


@pytest.mark.asyncio
async def test_run_shell_w_asyncio_cancel_kills_the_command(root):
    # A caller running 'run_shell' in a thread, whose task is cancelled,
    # sets the event 'is_cancelled' polls.
    cancelled = threading.Event()
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=cancelled.is_set,
    )
    thread_done = threading.Event()

    def in_thread():
        try:
            return client_tools.run_shell(
                {"command": _grandchild_command(root)},
                context,
            )
        finally:
            thread_done.set()

    async def runner():
        try:
            return await asyncio.to_thread(in_thread)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(runner())
    await asyncio.to_thread(_wait_for_start, root)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert await asyncio.to_thread(thread_done.wait, 10.0)
    _assert_all_dead(root)


def test_run_shell_w_interrupt_after_the_shell_exited(root):
    # Ctrl-C arrives just as the shell exits, leaving a background process
    # (POSIX):  the wait was interrupted, so the group is killed anyway.
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=_started_then(root, KeyboardInterrupt()),
    )

    with mock.patch.object(client_tools, "_exited", return_value=False):
        with pytest.raises(KeyboardInterrupt):
            client_tools.run_shell({"command": _child_command(root)}, context)

    _assert_all_dead(root)


def test_run_shell_kills_before_reaping(context):
    # The command's ID (and its group's) stays its own until it is reaped:
    # so kill first, then reap.
    calls = mock.Mock()
    proc = calls.proc
    proc.wait.return_value = -9

    with (
        mock.patch.object(client_tools.subprocess, "Popen", return_value=proc),
        mock.patch.object(client_tools, "_exited", return_value=False),
        mock.patch.object(client_tools, "_kill", calls.kill),
        mock.patch.object(
            client_tools.time,
            "sleep",
            side_effect=KeyboardInterrupt,
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            client_tools.run_shell({"command": "echo hi"}, context)

    assert calls.mock_calls == [mock.call.kill(proc), mock.call.proc.wait()]


@pytest.mark.parametrize(
    "waitid, expected",
    [
        (mock.Mock(return_value=None), False),
        (mock.Mock(return_value=object()), True),
        (mock.Mock(side_effect=ChildProcessError), True),
    ],
)
def test_exited_posix(monkeypatch, waitid, expected):
    monkeypatch.setattr(client_tools, "_POSIX", True)
    monkeypatch.setattr(client_tools.os, "waitid", waitid, raising=False)
    for name, value in [
        ("P_PID", 1),
        ("WEXITED", 4),
        ("WNOHANG", 1),
        ("WNOWAIT", 0x1000000),
    ]:
        monkeypatch.setattr(client_tools.os, name, value, raising=False)
    proc = mock.Mock(pid=1234)

    assert client_tools._exited(proc) is expected

    waitid.assert_called_once_with(1, 1234, 4 | 1 | 0x1000000)
    proc.poll.assert_not_called()  # it would reap


@pytest.mark.parametrize("returncode, expected", [(None, False), (0, True)])
def test_exited_windows(monkeypatch, returncode, expected):
    monkeypatch.setattr(client_tools, "_POSIX", False)
    proc = mock.Mock()
    proc.poll.return_value = returncode

    assert client_tools._exited(proc) is expected


def test_run_shell_w_cancelled_before_it_starts(context):
    context = dataclasses.replace(context, is_cancelled=lambda: True)

    with mock.patch.object(client_tools.subprocess, "Popen") as popen:
        with pytest.raises(client_tools.CommandCancelled):
            client_tools.run_shell({"command": "echo hi"}, context)

    popen.assert_not_called()


def test_run_shell_polled_w_timeout(root):
    command = _script(root, "slow.py", "import time\ntime.sleep(10)\n")
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        timeout_secs=0.3,
        is_cancelled=lambda: False,
    )

    found = client_tools.run_shell({"command": command}, context)

    assert found["timed_out"] is True
    assert "timed out after 0.3 seconds" in found["stderr"]


def test_run_shell_polled_w_normal_exit(root):
    command = _script(root, "quick.py", "print('done')\n")
    context = client_tools.ToolContext(
        root=root,
        allow_anywhere=True,
        is_cancelled=lambda: False,
    )

    found = client_tools.run_shell({"command": command}, context)

    assert found["stdout"].strip() == "done"
    assert found["exit_code"] == 0
    assert found["timed_out"] is False


# -- registry ----------------------------------------------------------------


def test_agui_tools():
    (found,) = client_tools.agui_tools()

    assert isinstance(found, agui_core.Tool)
    assert found.name == "shell"
    assert found.parameters["required"] == ["command"]
    assert found.description == client_tools.SHELL_TOOL.description


# -- execute_tool_call -------------------------------------------------------


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ("", {}),
        ('{"command": "ls"}', {"command": "ls"}),
        ("{not json", "{not json"),
    ],
)
def test_parse_tool_args(arguments, expected):
    assert client_tools.parse_tool_args(arguments) == expected


def test_execute_tool_call_w_unknown_tool(context):
    found = client_tools.execute_tool_call("nonesuch", {}, context)

    assert found["exit_code"] is None
    assert "Unknown client tool: 'nonesuch'" in found["error"]


def test_execute_tool_call_w_non_dict_args(context):
    found = client_tools.execute_tool_call("shell", "{bad", context)

    assert found["exit_code"] is None
    assert "must be a JSON object" in found["error"]


def test_execute_tool_call_w_declined(context):
    confirm = mock.Mock(return_value=False)
    tool = mock.Mock()
    tools = {"shell": client_tools.ClientTool("shell", "", {}, tool)}

    found = client_tools.execute_tool_call(
        "shell",
        {"command": "ls"},
        context,
        tools=tools,
        confirm=confirm,
    )

    confirm.assert_called_once_with("shell", {"command": "ls"})
    tool.assert_not_called()
    assert found["exit_code"] is None
    assert "declined" in found["error"]


@pytest.mark.parametrize("w_confirm", [False, True])
def test_execute_tool_call_runs(context, w_confirm):
    tool = mock.Mock(return_value={"exit_code": 0})
    tools = {"shell": client_tools.ClientTool("shell", "", {}, tool)}
    confirm = mock.Mock(return_value=True) if w_confirm else None

    found = client_tools.execute_tool_call(
        "shell",
        {"command": "ls"},
        context,
        tools=tools,
        confirm=confirm,
    )

    assert found is tool.return_value
    tool.assert_called_once_with({"command": "ls"}, context)


@pytest.mark.parametrize(
    "exc",
    [client_tools.PathRefused("/outside"), OSError("no shell")],
)
def test_execute_tool_call_w_error(context, exc):
    tool = mock.Mock(side_effect=exc)
    tools = {"shell": client_tools.ClientTool("shell", "", {}, tool)}

    found = client_tools.execute_tool_call(
        "shell",
        {"command": "ls"},
        context,
        tools=tools,
    )

    assert found["exit_code"] is None
    assert found["error"] == str(exc)


def test_tool_result_content():
    result = {
        "stdout": "out",
        "stderr": "",
        "exit_code": 3,
        "timed_out": False,
        "truncated": True,
        "stdout_bytes": 99,
        "stderr_bytes": 0,
    }

    found = json.loads(client_tools.tool_result_content(result))

    assert found == {
        "stdout": "out",
        "stderr": "",
        "exit_code": 3,
        "timed_out": False,
        "truncated": True,
    }


def test_tool_result_content_w_error():
    result = client_tools._not_run("nope")

    found = json.loads(client_tools.tool_result_content(result))

    assert found["error"] == "nope"
    assert found["exit_code"] is None


# -- ToolLog -----------------------------------------------------------------


def test_tool_log_record(tmp_path, context):
    log_path = tmp_path / "tools.jsonl"
    tool_log = client_tools.ToolLog(log_path)
    result = {
        "stdout": "x" * 5,
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
        "truncated": True,
        "stdout_bytes": 500,
        "stderr_bytes": 0,
    }

    tool_log.record(
        name="shell",
        args={"command": "ls"},
        context=context,
        result=result,
        duration_secs=0.12345,
    )
    tool_log.record(
        name="shell",
        args="{bad",
        context=context,
        result=client_tools._not_run("bad args"),
        duration_secs=0.0,
    )

    first, second = (
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    )

    assert first.pop("timestamp")
    assert first == {
        "tool": "shell",
        "args": {"command": "ls"},
        "cwd": str(context.root),
        "exit_code": 0,
        "duration_secs": 0.123,
        "timed_out": False,
        "truncated": True,
        "stdout_bytes": 500,
        "stderr_bytes": 0,
        "error": None,
    }
    assert "x" * 5 not in log_path.read_text(encoding="utf-8")
    assert second["exit_code"] is None
    assert second["stdout_bytes"] == 0
    assert second["error"] == "bad args"


# -- iter_sse_json -----------------------------------------------------------


def test_iter_sse_json():
    lines = [
        ": keepalive 123",
        "",
        "id: run:0",
        'data: {"a": 1}',
        "",
        b"event: message",
        b'data: {"b":',
        b"data: 2}",
        b"",
        "retry: 1000",
        'data: {"incomplete": true}',
    ]

    assert list(client_tools.iter_sse_json(lines)) == [{"a": 1}, {"b": 2}]


# -- SoliplexClient ----------------------------------------------------------


def _sse_body(events) -> str:
    return (
        "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        + ": keepalive\n\n"
    )


def _mock_client(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return client_tools.SoliplexClient("http://server/", "a room", http=http)


@pytest.mark.parametrize("token", [None, "secret"])
def test_soliplexclient_default_http(token):
    with client_tools.SoliplexClient("http://s", "r", token=token) as client:
        assert isinstance(client.http, httpx.Client)
        assert client.http.timeout == client_tools.DEFAULT_HTTP_TIMEOUT
        found = client.http.headers.get("Authorization")

    assert client.http.is_closed
    assert found == (f"Bearer {token}" if token else None)


def test_soliplexclient_agui_url():
    client = client_tools.SoliplexClient(
        "http://server/",
        "a/room",
        http=mock.Mock(),
    )

    assert client.agui_url == "http://server/api/v1/rooms/a%2Froom/agui"


@pytest.mark.parametrize("metadata", [None, {"name": "x"}])
def test_soliplexclient_new_thread(metadata):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"thread_id": THREAD_ID})

    client = _mock_client(handler)

    assert client.new_thread(metadata) == {"thread_id": THREAD_ID}

    (request,) = requests
    assert request.url == "http://server/api/v1/rooms/a%20room/agui"
    expected = {"metadata": metadata} if metadata else {}
    assert json.loads(request.content) == expected


def test_soliplexclient_new_run():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"run_id": CHILD_RUN_ID})

    client = _mock_client(handler)

    found = client.new_run(THREAD_ID, parent_run_id=RUN_ID)

    assert found == {"run_id": CHILD_RUN_ID}
    (request,) = requests
    assert request.url.path.endswith(f"/agui/{THREAD_ID}")
    assert json.loads(request.content) == {"parent_run_id": RUN_ID}


def test_soliplexclient_w_http_error():
    client = _mock_client(lambda request: httpx.Response(403, text="nope"))

    with pytest.raises(client_tools.HTTPFailure, match="HTTP 403.*nope"):
        client.new_thread()


def test_soliplexclient_stream_run():
    requests = []
    events = [_started(), _finished()]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=_sse_body(events))

    client = _mock_client(handler)
    run_input = _run_input()

    found = list(client.stream_run(run_input))

    assert [event.type for event in found] == [
        agui_core.EventType.RUN_STARTED,
        agui_core.EventType.RUN_FINISHED,
    ]
    (request,) = requests
    assert request.url.path.endswith(f"/agui/{THREAD_ID}/{RUN_ID}")
    assert request.headers["Accept"] == "text/event-stream"
    body = json.loads(request.content)
    assert body["threadId"] == THREAD_ID
    assert body["tools"][0]["name"] == "shell"


def test_soliplexclient_stream_run_w_http_error():
    client = _mock_client(lambda request: httpx.Response(404, text="gone"))

    with pytest.raises(client_tools.HTTPFailure, match="HTTP 404.*gone"):
        list(client.stream_run(_run_input()))


# -- initial_run_input -------------------------------------------------------


@pytest.mark.parametrize(
    "run_info, expected_state",
    [
        ({"run_input": None}, {}),
        ({"run_input": {"state": None}}, {}),
        ({"run_input": {"state": {"a": 1}}}, {"a": 1}),
    ],
)
def test_initial_run_input(run_info, expected_state):
    thread = {"thread_id": THREAD_ID, "runs": {RUN_ID: run_info}}

    found = client_tools.initial_run_input(thread, "hello")

    assert found.thread_id == THREAD_ID
    assert found.run_id == RUN_ID
    assert found.state == expected_state
    (message,) = found.messages
    assert message.role == "user"
    assert message.content == "hello"
    assert [tool.name for tool in found.tools] == ["shell"]


# -- run loop (scripted server) ----------------------------------------------


def _run_input(messages=None, run_id=RUN_ID):
    return agui_core.RunAgentInput(
        thread_id=THREAD_ID,
        run_id=run_id,
        state={},
        messages=messages or [agui_core.UserMessage(id="u1", content="hi")],
        tools=client_tools.agui_tools(),
        context=[],
        forwarded_props={},
    )


def _started(run_id=RUN_ID):
    return {"type": "RUN_STARTED", "threadId": THREAD_ID, "runId": run_id}


def _finished(run_id=RUN_ID):
    return {"type": "RUN_FINISHED", "threadId": THREAD_ID, "runId": run_id}


def _text(message_id, text):
    return [
        {"type": "TEXT_MESSAGE_START", "messageId": message_id},
        {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": message_id,
            "delta": text,
        },
        {"type": "TEXT_MESSAGE_END", "messageId": message_id},
    ]


def _call(call_id, name, args, parent="assistant-1", end=True):
    events = [
        {
            "type": "TOOL_CALL_START",
            "toolCallId": call_id,
            "toolCallName": name,
            "parentMessageId": parent,
        },
        {"type": "TOOL_CALL_ARGS", "toolCallId": call_id, "delta": args},
    ]
    if end:
        events.append({"type": "TOOL_CALL_END", "toolCallId": call_id})
    return events


class ScriptedServer:
    """Answer the AG-UI endpoints from scripted run streams

    'runs' maps a run ID to the events its stream sends;  new runs get
    IDs from 'child_ids', in order.  Records the run inputs it receives.
    """

    def __init__(self, runs, child_ids=()):
        self.runs = runs
        self.child_ids = list(child_ids)
        self.run_inputs = []
        self.new_run_bodies = []

    def __call__(self, request):
        path = request.url.path
        body = json.loads(request.content)

        if path.endswith(f"/agui/{THREAD_ID}"):
            self.new_run_bodies.append(body)
            return httpx.Response(
                200,
                json={"run_id": self.child_ids.pop(0)},
            )

        run_id = path.rsplit("/", 1)[-1]
        self.run_inputs.append(agui_core.RunAgentInput.model_validate(body))
        return httpx.Response(200, text=_sse_body(self.runs[run_id]))


def _loop(server, root, **kwargs):
    client = _mock_client(server)
    context = client_tools.ToolContext(root=root)
    return client_tools.run_loop(client, _run_input(), context, **kwargs)


def test_run_loop_w_final_answer_at_once(root):
    server = ScriptedServer(
        {RUN_ID: [_started(), *_text("a1", "Hi!"), _finished()]}
    )
    on_event = mock.Mock()

    found = _loop(server, root, on_event=on_event)

    assert found.as_json() == {
        "thread_id": THREAD_ID,
        "run_ids": [RUN_ID],
        "response": "Hi!",
        "tool_calls": [],
    }
    assert on_event.call_count == 5
    assert found.run_input.messages[-1].content == "Hi!"


@pytest.mark.parametrize("w_tool_log", [False, True])
def test_run_loop_round_trip(root, tmp_path, w_tool_log):
    (root / "marker.txt").write_text("m", encoding="utf-8")
    first = [
        _started(),
        *_text("a0", "Looking."),
        *_call("c1", "shell", '{"command": "echo from-shell"}'),
        # A tool the server ran itself:  its result is in the stream.
        *_call("c2", "server_tool", "{}"),
        {
            "type": "TOOL_CALL_RESULT",
            "toolCallId": "c2",
            "messageId": "r2",
            "content": "server-result",
        },
        # A pending call to a tool nobody knows:  gets an error result.
        *_call("c3", "mystery", "{}", parent=None),
        # A refused command:  the error goes back to the model.
        *_call("c4", "shell", '{"command": "cat /outside/file"}'),
        # A failing command:  its exit code goes back to the model.
        *_call("c5", "shell", '{"command": "exit 3"}'),
        # A home directory which does not exist:  an error, not a crash.
        *_call("c6", "shell", '{"command": "ls ~nonesuch-user-zzz/x"}'),
        _finished(),
    ]
    second = [
        _started(CHILD_RUN_ID),
        *_text("a9", "Done."),
        _finished(CHILD_RUN_ID),
    ]
    server = ScriptedServer(
        {RUN_ID: first, CHILD_RUN_ID: second},
        child_ids=[CHILD_RUN_ID],
    )
    log_path = tmp_path / "tools.jsonl"
    tool_log = client_tools.ToolLog(log_path) if w_tool_log else None

    found = _loop(server, root, tool_log=tool_log)

    assert found.response == "Done."
    assert found.run_ids == [RUN_ID, CHILD_RUN_ID]
    assert [(call.name, call.exit_code) for call in found.tool_calls] == [
        ("shell", 0),
        ("shell", None),
        ("shell", 3),
        ("shell", None),
        # Its parent was made up, after the stream's own message.
        ("mystery", None),
    ]
    assert found.tool_calls[0].args == {"command": "echo from-shell"}

    assert server.new_run_bodies == [{"parent_run_id": RUN_ID}]

    # The next run's history holds each call AND its result.
    _, continuation = server.run_inputs
    assert continuation.run_id == CHILD_RUN_ID
    assert continuation.parent_run_id == RUN_ID
    calls = {
        call.id: call
        for message in continuation.messages
        if isinstance(message, agui_core.AssistantMessage)
        for call in message.tool_calls or ()
    }
    results = {
        message.tool_call_id: message.content
        for message in continuation.messages
        if isinstance(message, agui_core.ToolMessage)
    }
    assert (
        set(calls)
        == set(results)
        == {
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "c6",
        }
    )
    assert results["c2"] == "server-result"
    shell_result = json.loads(results["c1"])
    assert shell_result["stdout"].strip() == "from-shell"
    assert shell_result["exit_code"] == 0
    assert "Unknown client tool" in json.loads(results["c3"])["error"]
    assert "outside --root" in json.loads(results["c4"])["error"]
    assert json.loads(results["c5"])["exit_code"] == 3
    assert "outside --root" in json.loads(results["c6"])["error"]

    if w_tool_log:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["exit_code"] for line in lines] == [
            0,
            None,
            3,
            None,
            None,
        ]
    else:
        assert not log_path.exists()


@pytest.mark.parametrize(
    "option_kwargs, hint",
    [
        ({}, ""),
        (
            {"max_turns_option": "--max-turns"},
            " (raise --max-turns to allow more)",
        ),
    ],
)
def test_run_loop_w_max_turns(root, option_kwargs, hint):
    first = [
        _started(),
        *_call("c1", "shell", '{"command": "echo hi"}'),
        _finished(),
    ]
    server = ScriptedServer({RUN_ID: first})

    with mock.patch.object(client_tools, "execute_tool_call") as etc:
        with pytest.raises(client_tools.MaxTurnsExceeded) as exc_info:
            _loop(server, root, max_turns=1, **option_kwargs)

    assert str(exc_info.value) == (
        f"The model still wanted to call tools after 1 runs{hint}"
    )

    # The calls it could not send back were never executed.
    etc.assert_not_called()
    assert server.new_run_bodies == []


def test_run_loop_passes_confirm(root):
    first = [
        _started(),
        *_call("c1", "shell", '{"command": "echo hi"}'),
        _finished(),
    ]
    second = [
        _started(CHILD_RUN_ID),
        *_text("a", "ok"),
        _finished(CHILD_RUN_ID),
    ]
    server = ScriptedServer(
        {RUN_ID: first, CHILD_RUN_ID: second},
        child_ids=[CHILD_RUN_ID],
    )
    confirm = mock.Mock(return_value=False)

    found = _loop(server, root, confirm=confirm)

    confirm.assert_called_once_with("shell", {"command": "echo hi"})
    assert found.tool_calls[0].exit_code is None
    result = json.loads(server.run_inputs[-1].messages[-1].content)
    assert "declined" in result["error"]


@pytest.mark.parametrize(
    "events, match",
    [
        ([_started(), {"type": "RUN_ERROR", "message": "boom"}], "boom"),
        ([_started()], "without 'RUN_FINISHED'"),
        ([], "without 'RUN_FINISHED'"),
        (
            [_started(), *_call("c1", "shell", "{}", end=False), _finished()],
            "incomplete tool call",
        ),
        ([_started(), {"type": "NONESUCH"}], "Invalid AG-UI stream"),
        (
            [
                _started(),
                {
                    "type": "TEXT_MESSAGE_CONTENT",
                    "messageId": "x",
                    "delta": "y",
                },
            ],
            "Invalid AG-UI stream",
        ),
    ],
)
def test_run_loop_w_bad_run(root, events, match):
    server = ScriptedServer({RUN_ID: events})

    with pytest.raises(client_tools.RunFailed, match=match):
        _loop(server, root)


def test_run_loop_w_bad_sse_json(root):
    client = _mock_client(
        lambda request: httpx.Response(200, text="data: {nope\n\n"),
    )
    context = client_tools.ToolContext(root=root)

    with pytest.raises(client_tools.InvalidStream):
        client_tools.run_loop(client, _run_input(), context)


def test_pending_tool_calls():
    call = agui_core.ToolCall(
        id="c1",
        function=agui_core.FunctionCall(name="shell", arguments="{}"),
    )
    answered = call.model_copy(update={"id": "c2"})
    messages = [
        agui_core.UserMessage(id="u", content="hi"),
        agui_core.AssistantMessage(id="a0", content="no calls"),
        agui_core.AssistantMessage(id="a1", tool_calls=[call, answered]),
        agui_core.ToolMessage(id="t2", tool_call_id="c2", content="x"),
    ]

    assert client_tools.pending_tool_calls(messages) == [call]


# -- routed, end to end ------------------------------------------------------

ROOM_ID = "scripted-shell"

INSTALLATION_YAML = """\
id: "client-tools-e2e"
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
  - "./rooms/scripted-shell"
oidc_paths: []
completion_paths: []
quizzes_paths: []
"""

ROOM_YAML = f"""\
id: "{ROOM_ID}"
name: "Scripted shell"
description: "Calls the 'shell' client tool"
agent:
  kind: "factory"
  factory_name: "tests.unit.agui.scripted_shell_room.agent_factory"
allow_mcp: false
"""


@pytest.fixture
def e2e_client(tmp_path, patched_app_routers):
    install = tmp_path / "install"
    room_dir = install / "rooms" / ROOM_ID
    room_dir.mkdir(parents=True)
    (room_dir / "room_config.yaml").write_text(ROOM_YAML, encoding="utf-8")
    agui_db = tmp_path / "agui.sqlite"
    authz_db = tmp_path / "authz.sqlite"
    (install / "installation.yaml").write_text(
        INSTALLATION_YAML.format(
            agui_sync=sqlite_dburi(agui_db),
            agui_async=sqlite_dburi(agui_db, "+aiosqlite"),
            authz_sync=sqlite_dburi(authz_db),
            authz_async=sqlite_dburi(authz_db, "+aiosqlite"),
        ),
        encoding="utf-8",
    )

    config_routing.register_default_routers()
    app = main.create_app(install, no_auth_mode=True)
    config_routing.add_registered_routers(app)

    scripted_shell_room.HISTORIES.clear()

    with testclient.TestClient(app) as http:
        yield client_tools.SoliplexClient("", ROOM_ID, http=http)


def _parts(history, part_type):
    return [
        part
        for message in history
        for part in message.parts
        if isinstance(part, part_type)
    ]


@pytest.mark.parametrize("prompt", ["just shell", "mixed tools"])
def test_run_loop_end_to_end(e2e_client, root, prompt):
    context = client_tools.ToolContext(root=root)
    thread = e2e_client.new_thread()
    run_input = client_tools.initial_run_input(thread, prompt)

    found = client_tools.run_loop(e2e_client, run_input, context)

    assert len(found.run_ids) == 2
    assert [(call.name, call.exit_code) for call in found.tool_calls] == [
        ("shell", 0),
    ]
    assert found.response.startswith("shell said: ")
    assert "client-tool-ran" in found.response

    # The second model request's history holds the call AND its result.
    first_history, second_history = scripted_shell_room.HISTORIES
    assert not _parts(first_history, ai_messages.ToolReturnPart)
    (shell_call,) = [
        part
        for part in _parts(second_history, ai_messages.ToolCallPart)
        if part.tool_name == "shell"
    ]
    (shell_return,) = [
        part
        for part in _parts(second_history, ai_messages.ToolReturnPart)
        if part.tool_name == "shell"
    ]
    assert shell_return.tool_call_id == shell_call.tool_call_id
    # The server rehydrates the JSON content the client sent.
    content = shell_return.content
    assert content["stdout"].strip() == "client-tool-ran"
    assert content["exit_code"] == 0

    server_returns = [
        part.content
        for part in _parts(second_history, ai_messages.ToolReturnPart)
        if part.tool_name == "server_tool"
    ]
    if "mixed" in prompt:
        assert server_returns == ["server-tool-ran"]
    else:
        assert server_returns == []


def test_scripted_shell_room_server_tool():
    assert scripted_shell_room.server_tool() == "server-tool-ran"


# -- confirmation helpers ----------------------------------------------------


@pytest.mark.parametrize(
    "name, args, expected",
    [
        ("shell", {"command": "ls -la"}, "shell: ls -la"),
        ("shell", "{bad", 'shell: "{bad"'),
        ("other", {"a": 1}, 'other: {"a": 1}'),
    ],
)
def test_describe_tool_call(name, args, expected):
    assert client_tools.describe_tool_call(name, args) == expected


@pytest.mark.parametrize("answer", [False, True])
def test_confirm_via_callback_w_answer_later(answer):
    requests = []

    def request(name, args, reply):
        requests.append((name, args))
        # A UI answers on its own thread, after the request returns.
        threading.Timer(0.2, reply, args=(answer,)).start()

    confirm = client_tools.confirm_via_callback(request, poll_secs=0.05)

    assert confirm("shell", {"command": "ls"}) is answer
    assert requests == [("shell", {"command": "ls"})]


def test_confirm_via_callback_w_answer_at_once():
    confirm = client_tools.confirm_via_callback(
        lambda name, args, reply: reply(1),
    )

    assert confirm("shell", {}) is True


def test_confirm_via_callback_w_cancelled_before_asking():
    request = mock.Mock()
    confirm = client_tools.confirm_via_callback(
        request,
        is_cancelled=lambda: True,
    )

    with pytest.raises(client_tools.ConfirmationCancelled):
        confirm("shell", {})

    request.assert_not_called()


def test_confirm_via_callback_w_cancelled_while_waiting():
    checks = []

    def is_cancelled():
        checks.append(None)
        return len(checks) > 2  # before asking;  first poll;  second poll

    confirm = client_tools.confirm_via_callback(
        lambda name, args, reply: None,  # never answers
        is_cancelled=is_cancelled,
        poll_secs=0.01,
    )

    with pytest.raises(client_tools.ConfirmationCancelled):
        confirm("shell", {})

    assert len(checks) == 3


def test_confirm_via_callback_w_cancelled_once_answered():
    cancelled = []

    def request(name, args, reply):
        cancelled.append(True)  # e.g., the UI quits as the user says yes
        reply(True)

    confirm = client_tools.confirm_via_callback(
        request,
        is_cancelled=lambda: bool(cancelled),
    )

    with pytest.raises(client_tools.ConfirmationCancelled):
        confirm("shell", {})


# -- run loop:  state, and tool result callback ------------------------------


def test_run_loop_w_invalid_state_delta(root):
    bad_delta = {
        "type": "STATE_DELTA",
        "delta": [{"op": "replace", "path": "/missing/x", "value": 1}],
    }
    server = ScriptedServer(
        {RUN_ID: [_started(), bad_delta, *_text("a1", "Hi"), _finished()]},
    )

    found = _loop(server, root)

    assert found.run_input.state == client_tools.INVALID_STATE
    assert found.run_input.state is not client_tools.INVALID_STATE


def test_run_loop_w_on_tool_result(root):
    first = [
        _started(),
        *_call("c1", "shell", '{"command": "exit 4"}'),
        _finished(),
    ]
    second = [
        _started(CHILD_RUN_ID),
        *_text("a", "ok"),
        _finished(CHILD_RUN_ID),
    ]
    server = ScriptedServer(
        {RUN_ID: first, CHILD_RUN_ID: second},
        child_ids=[CHILD_RUN_ID],
    )
    on_tool_result = mock.Mock()

    found = _loop(server, root, on_tool_result=on_tool_result)

    (record, tool_result), _ = on_tool_result.call_args
    assert record is found.tool_calls[0]
    assert record.exit_code == 4
    assert tool_result["exit_code"] == 4


# -- run loop:  progress survives a failure ---------------------------------


def _shell_then(second, *, commands=('{"command": "echo hi"}',)):
    first = [_started()]
    for index, args in enumerate(commands):
        first += _call(f"c{index}", "shell", args)
    first.append(_finished())
    return ScriptedServer(
        {RUN_ID: first, CHILD_RUN_ID: second},
        child_ids=[CHILD_RUN_ID],
    )


def _results(run_input):
    return {
        message.tool_call_id: json.loads(message.content)
        for message in run_input.messages
        if isinstance(message, agui_core.ToolMessage)
    }


def test_run_loop_w_failed_continuation_keeps_progress(root):
    second = [
        _started(CHILD_RUN_ID),
        {"type": "RUN_ERROR", "message": "boom"},
    ]
    server = _shell_then(second)

    with pytest.raises(client_tools.RunErrored) as exc_info:
        _loop(server, root)

    progress = exc_info.value.result
    assert progress.run_ids == [RUN_ID, CHILD_RUN_ID]
    assert [call.exit_code for call in progress.tool_calls] == [0]
    # The call and its result:  the next prompt will not ask for it again.
    assert client_tools.pending_tool_calls(progress.run_input.messages) == []
    assert _results(progress.run_input)["c0"]["stdout"].strip() == "hi"


def test_run_loop_w_failed_first_run_keeps_input(root):
    server = ScriptedServer({RUN_ID: [_started()]})
    client = _mock_client(server)
    run_input = _run_input()

    with pytest.raises(client_tools.RunNotFinished) as exc_info:
        client_tools.run_loop(
            client,
            run_input,
            client_tools.ToolContext(root=root),
        )

    assert exc_info.value.result.run_input is run_input
    assert exc_info.value.result.tool_calls == []


def test_run_loop_w_cancelled_confirmation(root):
    server = _shell_then(
        [],
        commands=('{"command": "echo one"}', '{"command": "echo two"}'),
    )
    confirm = mock.Mock(side_effect=client_tools.ConfirmationCancelled())
    on_tool_result = mock.Mock()

    with mock.patch.object(client_tools, "run_shell") as run_shell:
        with pytest.raises(client_tools.ConfirmationCancelled) as exc_info:
            _loop(server, root, confirm=confirm, on_tool_result=on_tool_result)

    run_shell.assert_not_called()
    confirm.assert_called_once()  # not asked again, for the second call
    assert server.new_run_bodies == []  # no further run

    progress = exc_info.value.result
    results = _results(progress.run_input)
    assert set(results) == {"c0", "c1"}
    assert all("Cancelled" in result["error"] for result in results.values())
    assert [call.exit_code for call in progress.tool_calls] == [None, None]
    assert on_tool_result.call_count == 2


def test_soliplexclient_w_transport_error_on_post():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = _mock_client(handler)

    with pytest.raises(client_tools.TransportFailure, match="refused"):
        client.new_run(THREAD_ID, parent_run_id=RUN_ID)


def test_soliplexclient_w_transport_error_on_stream():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = _mock_client(handler)

    with pytest.raises(client_tools.TransportFailure, match="slow"):
        list(client.stream_run(_run_input()))


class _StalledStream(httpx.SyncByteStream):
    """Sends 'body', then times out, as a dead connection would"""

    def __init__(self, body: str):
        self.body = body

    def __iter__(self):
        yield self.body.encode()
        raise httpx.ReadTimeout("stalled")


def test_run_loop_w_timeout_mid_continuation_keeps_progress(root):
    server = _shell_then([])
    scripted = server.__call__

    def handler(request):
        if request.url.path.endswith(f"/{CHILD_RUN_ID}"):
            body = f"data: {json.dumps(_started(CHILD_RUN_ID))}\n\n"
            return httpx.Response(200, stream=_StalledStream(body))
        return scripted(request)

    client = _mock_client(handler)

    with pytest.raises(client_tools.TransportFailure, match="stalled") as ei:
        client_tools.run_loop(
            client,
            _run_input(),
            client_tools.ToolContext(root=root),
        )

    progress = ei.value.result
    assert progress.run_ids == [RUN_ID, CHILD_RUN_ID]
    assert _results(progress.run_input)["c0"]["exit_code"] == 0


def test_run_loop_w_command_cancelled(root):
    # E.g. the TUI quits while a command runs:  it is killed, and nothing
    # more runs.
    server = _shell_then(
        [],
        commands=('{"command": "echo one"}', '{"command": "echo two"}'),
    )

    run_shell = mock.Mock(side_effect=client_tools.CommandCancelled())
    tools = {
        "shell": dataclasses.replace(
            client_tools.SHELL_TOOL, execute=run_shell
        )
    }

    on_tool_result = mock.Mock()

    with pytest.raises(client_tools.CommandCancelled) as exc_info:
        _loop(server, root, tools=tools, on_tool_result=on_tool_result)

    assert on_tool_result.call_count == 2
    assert on_tool_result.call_args_list[0].args[1]["killed"] is True
    assert on_tool_result.call_args_list[1].args[1].get("killed") is None

    run_shell.assert_called_once()  # not the second call
    assert server.new_run_bodies == []

    results = _results(exc_info.value.result.run_input)
    assert results["c0"]["error"] == (
        client_tools.CommandCancelled.call_outcome
    )
    assert "killed" not in results["c0"]  # for the UI, not the model
    assert results["c1"]["error"] == (
        client_tools.ConfirmationCancelled.call_outcome
    )
    assert isinstance(exc_info.value, client_tools.Cancelled)


# -- auto-approve ------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, expected",
    [
        (client_tools.Approval.RUN_ALL, client_tools.Approval.RUN_ALL),
        (client_tools.Approval.DECLINE, client_tools.Approval.DECLINE),
        (True, client_tools.Approval.RUN_ONCE),
        (False, client_tools.Approval.DECLINE),
        (None, client_tools.Approval.DECLINE),  # e.g., a dismissed dialog
    ],
)
def test_approval_of(answer, expected):
    assert client_tools.Approval.of(answer) is expected


@pytest.mark.parametrize("w_callback", [False, True])
def test_auto_approve_enable(w_callback):
    on_enabled = mock.Mock() if w_callback else None
    auto = client_tools.AutoApprove(on_enabled=on_enabled)

    auto.enable()
    auto.enable()  # once is enough

    assert auto.enabled is True
    if w_callback:
        on_enabled.assert_called_once_with()


def _answering(*answers):
    requests = []
    answers = list(answers)

    def request(name, args, reply):
        requests.append(name)
        reply(answers.pop(0))

    return request, requests


def test_confirm_via_callback_w_auto_approve_on():
    request, requests = _answering()
    confirm = client_tools.confirm_via_callback(
        request,
        auto_approve=client_tools.AutoApprove(enabled=True),
    )

    assert confirm("shell", {}) is True
    assert confirm("shell", {}) is True
    assert requests == []  # never asked


def test_confirm_via_callback_w_run_all():
    request, requests = _answering(client_tools.Approval.RUN_ALL)
    on_enabled = mock.Mock()
    auto = client_tools.AutoApprove(on_enabled=on_enabled)
    confirm = client_tools.confirm_via_callback(request, auto_approve=auto)

    assert confirm("shell", {}) is True  # this call
    assert confirm("shell", {}) is True  # and every later one
    assert confirm("shell", {}) is True

    assert requests == ["shell"]  # asked once
    assert auto.enabled is True
    on_enabled.assert_called_once_with()


def test_confirm_via_callback_w_run_all_without_auto_approve():
    request, requests = _answering(
        client_tools.Approval.RUN_ALL,
        client_tools.Approval.DECLINE,
    )
    confirm = client_tools.confirm_via_callback(request)

    assert confirm("shell", {}) is True  # just this one
    assert confirm("shell", {}) is False

    assert requests == ["shell", "shell"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        (client_tools.Approval.RUN_ONCE, True),
        (client_tools.Approval.DECLINE, False),
    ],
)
def test_confirm_via_callback_w_run_once_or_decline(answer, expected):
    request, requests = _answering(answer, client_tools.Approval.DECLINE)
    auto = client_tools.AutoApprove()
    confirm = client_tools.confirm_via_callback(request, auto_approve=auto)

    assert confirm("shell", {}) is expected
    assert confirm("shell", {}) is False  # still asks

    assert requests == ["shell", "shell"]
    assert auto.enabled is False


def test_confirm_via_callback_w_auto_approve_and_cancelled():
    request = mock.Mock()
    confirm = client_tools.confirm_via_callback(
        request,
        is_cancelled=lambda: True,
        auto_approve=client_tools.AutoApprove(enabled=True),
    )

    with pytest.raises(client_tools.ConfirmationCancelled):
        confirm("shell", {})

    request.assert_not_called()


def test_run_loop_w_auto_approve_still_checks_paths(root, tmp_path):
    # Auto-approve skips only the question:  the path check, and the log,
    # still apply.
    server = _shell_then(
        [_started(CHILD_RUN_ID), *_text("a2", "ok"), _finished(CHILD_RUN_ID)],
        commands=('{"command": "cat /outside/file"}',),
    )
    request = mock.Mock()
    confirm = client_tools.confirm_via_callback(
        request,
        auto_approve=client_tools.AutoApprove(enabled=True),
    )
    log_path = tmp_path / "tools.jsonl"

    found = _loop(
        server,
        root,
        confirm=confirm,
        tool_log=client_tools.ToolLog(log_path),
    )

    request.assert_not_called()
    assert found.tool_calls[0].exit_code is None
    (message,) = [
        m
        for m in found.run_input.messages
        if isinstance(m, agui_core.ToolMessage)
    ]
    assert "outside --root" in json.loads(message.content)["error"]
    (entry,) = log_path.read_text(encoding="utf-8").splitlines()
    assert "outside --root" in json.loads(entry)["error"]


# -- rendering a tool result, for a UI ----------------------------------------


def _record(exit_code=0, command="ls -la"):
    return client_tools.ToolCallRecord(
        name="shell",
        args={"command": command},
        exit_code=exit_code,
    )


@pytest.mark.parametrize(
    "result, exit_code, expected",
    [
        ({"error": "Refused: x"}, None, "not run"),
        (
            {"error": "Cancelled", "killed": True},
            None,
            "killed while it ran (it may have made changes)",
        ),
        ({"timed_out": True}, -9, "ran, and timed out"),
        ({}, 3, "ran, exit code 3"),
    ],
)
def test_describe_tool_result(result, exit_code, expected):
    found = client_tools.describe_tool_result(_record(exit_code), result)

    assert found == expected


def _fences(markdown):
    """The parsed Markdown's top-level tokens, and its code blocks' text"""
    tokens = markdown_it.MarkdownIt().parse(markdown)
    return (
        [token.type for token in tokens],
        [token.content for token in tokens if token.type == "fence"],
    )


def test_render_tool_result_w_no_output():
    found = client_tools.render_tool_result(
        _record(),
        {"stdout": "", "stderr": ""},
    )

    assert found == (
        "\n\n** client tool call -- ran, exit code 0 **"
        "\n\n```text\n$ ls -la\n```"
    )


def test_render_tool_result_previews_output():
    stdout = "".join(f"line {n}\n" for n in range(25))
    long_line = "x" * (client_tools.PREVIEW_LINE_CHARS + 10)
    result = {
        "stdout": stdout,
        "stderr": f"{long_line}\n",
        "truncated": True,
    }

    found = client_tools.render_tool_result(_record(), result)

    heading, block = found.strip().split("\n\n", 1)
    assert heading == "** client tool call -- ran, exit code 0 **"
    lines = block.splitlines()
    assert lines[0] == "```text"
    assert lines[1] == "$ ls -la"
    assert lines[2] == "[stdout]"
    assert lines[3:23] == [f"line {n}" for n in range(20)]
    assert lines[23] == "[... 5 more lines]"
    assert lines[24] == "[stderr]"
    assert lines[25] == "x" * client_tools.PREVIEW_LINE_CHARS + " [...]"
    assert lines[26] == "[output truncated before it reached the model]"
    assert lines[27] == "```"


def test_render_tool_result_w_error_and_other_tool():
    record = client_tools.ToolCallRecord(
        name="other",
        args={"a": 1},
        exit_code=None,
    )

    found = client_tools.render_tool_result(record, {"error": "# no *x*"})

    types_, fences = _fences(found)
    assert types_ == ["paragraph_open", "inline", "paragraph_close", "fence"]
    assert fences == ['other: {"a": 1}\n[error] # no *x*\n']


def test_render_tool_result_markup_stays_in_the_block():
    # The command and its output are the model's and the command's:  a
    # heredoc holding a fence and a heading, echoed back, must not escape
    # the code block.
    command = "cat <<'EOF'\n```\n# heading\n````\nEOF"
    record = _record(command=command)
    result = {"stdout": "```\n# heading\n````\n", "stderr": "*bold*\n"}

    found = client_tools.render_tool_result(record, result, max_lines=10)

    types_, fences = _fences(found)
    assert types_ == ["paragraph_open", "inline", "paragraph_close", "fence"]
    assert fences == [
        f"$ {command}\n[stdout]\n```\n# heading\n````\n[stderr]\n*bold*\n",
    ]
    assert "`````text\n" in found  # longer than any run within
