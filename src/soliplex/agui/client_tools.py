"""Client-side tools for AG-UI clients of a Soliplex server

A client advertises these tools in 'RunAgentInput.tools'.  The server
hands them to the room's agent as deferred tools: when the model calls
one, the run ends with the call pending, and the *client* executes it on
its own host and starts the next run with the call's result.

Round-trip rule: the next run's history must carry BOTH the assistant
message holding the tool call AND the tool message holding its result.
A result whose call is missing from the history makes the server fail
the run ('Tool call with ID ... not found in the history').

The 'shell' tool runs a command on the client's host, in the '--root'
directory.  Its path check is a guard rail, NOT a sandbox:  it refuses a
command which names an absolute path (or '~', '$VAR' or '..' path)
outside the root, but a command can still build a path at runtime, and
it runs with the user's own permissions and network access.  It runs
with a scrubbed copy of the user's environment (no 'SOLIPLEX_TOKEN', and
nothing named like a secret) unless told to pass the full one, and it
is killed, with every process it started, if it times out, is cancelled,
or its caller is interrupted (e.g., by Ctrl-C).

This module holds everything but the user interface, so that clients
('soliplex-cli ask --url', the TUI) can be thin layers over it:

- the tool registry, and its AG-UI 'Tool' definitions;
- the 'shell' executor;
- SSE decoding, and a small 'httpx' client for the AG-UI endpoints;
- the run loop, which executes client tool calls until the model gives
  its final answer.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import fnmatch
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import typing
import uuid
from collections import abc
from urllib import parse as urllib_parse

import httpx
from ag_ui import core as agui_core

from soliplex.agui import parser as agui_parser

DEFAULT_TOOL_TIMEOUT_SECS = 60.0
DEFAULT_OUTPUT_CAP_BYTES = 16 * 1024
DEFAULT_MAX_TURNS = 10

#   Connect / write / pool timeouts, and the read timeout:  the server
#   sends an SSE keepalive every 15 seconds, so a long read timeout only
#   fires on a dead connection.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(30.0, read=300.0)

_POSIX = os.name != "nt"

#   Absolute paths which name no file, and which commands use all the time
#   (e.g. '2>/dev/null').
_HARMLESS_PATHS = frozenset(["/dev/null"])

#   An absolute (or home-relative) path embedded in a larger word, e.g.
#   '--file=/etc/passwd', or "open('/etc/passwd')";  not the '//host' of
#   a URL ('https://host/path'), since a ':' may not precede it.
_EMBEDDED_PATH = re.compile(
    r"(?<![\w.:~\\/-])(?:[A-Za-z]:[\\/]|[\\/]|~)[^\s'\"`;|&<>()]*",
)
_PATH_SEPARATORS = re.compile(r"[\\/]")

#   A 'cmd.exe' switch, e.g. the '/s' of 'dir /s /b':  on Windows, a word
#   with one leading '/' and no other separator is not taken for a path.
_WINDOWS_SWITCH = re.compile(r"/[^\\/]*")


class ClientToolsError(Exception):
    """A client-tools run which could not complete

    When raised from 'run_loop', 'result' is the 'LoopResult' so far:  its
    'run_input' is the last consistent history (every tool call in it has
    its result), so a client can carry on from there without repeating a
    call which already ran.
    """

    result: LoopResult | None = None


class PathRefused(ValueError):
    """A 'shell' command named a path outside the root"""

    def __init__(self, path):
        self.path = path
        super().__init__(f"Refused: path outside --root: {path}")


class CdRefused(PathRefused):
    """A 'cd' to '$HOME' or '$OLDPWD', which names no path"""

    def __init__(self, command: str, *, to_oldpwd: bool = False):
        if to_oldpwd:
            command, self.path = f"{command} -", "$OLDPWD"
        else:
            self.path = "$HOME"

        ValueError.__init__(
            self,
            f"Refused: '{command}' goes to {self.path}, outside --root",
        )


class MaxTurnsExceeded(ClientToolsError):
    """The model still called tools on the last run 'run_loop' could make

    'option' names the setting which raises the limit, in the front end in
    use (e.g., '--max-turns');  without one, the message names none.
    """

    def __init__(self, max_turns: int, option: str | None = None):
        self.max_turns = max_turns
        hint = f" (raise {option} to allow more)" if option else ""
        super().__init__(
            f"The model still wanted to call tools after {max_turns} "
            f"runs{hint}",
        )


class Cancelled(ClientToolsError):
    """The caller cancelled a client tool call;  'run_loop' stops

    'call_outcome' is the error the cancelled call's result reports.
    """

    call_outcome = "Cancelled"


class CommandCancelled(Cancelled):
    call_outcome = "Cancelled while it ran:  the command was killed"

    def __init__(self):
        super().__init__(
            "Cancelled while running a client tool call (it was killed)",
        )


class HTTPFailure(ClientToolsError):
    def __init__(self, response: httpx.Response):
        self.status_code = response.status_code
        super().__init__(
            f"HTTP {response.status_code} from {response.url}: "
            f"{response.text}",
        )


class TransportFailure(ClientToolsError):
    def __init__(self, exc: httpx.HTTPError):
        super().__init__(f"Could not talk to the server: {exc!r}")


class ConfirmationCancelled(Cancelled):
    call_outcome = "Cancelled before it could run"

    def __init__(self):
        super().__init__("Cancelled while confirming a client tool call")


class RunFailed(ClientToolsError):
    """The server reported a 'RUN_ERROR', or the stream was not usable"""


class RunErrored(RunFailed):
    def __init__(self, message: str | None):
        super().__init__(f"Run failed: {message}")


class InvalidStream(RunFailed):
    def __init__(self, reason):
        super().__init__(f"Invalid AG-UI stream: {reason}")


class RunNotFinished(InvalidStream):
    def __init__(self):
        super().__init__("it ended without 'RUN_FINISHED'")


class IncompleteToolCall(InvalidStream):
    def __init__(self):
        super().__init__("the run finished with an incomplete tool call")


class RootNotADirectory(ValueError):
    def __init__(self, root: pathlib.Path):
        self.root = root
        super().__init__(f"--root is not a directory: {root}")


class InvalidShellArgs(ValueError):
    def __init__(self):
        super().__init__("'shell' requires a non-empty string 'command'")


@dataclasses.dataclass(frozen=True)
class ToolContext:
    """Where, and within what limits, client tools execute

    'root' is the working directory for commands (resolved on creation);
    unless 'allow_anywhere', a command naming a path outside it is
    refused.  'timeout_secs' bounds each command, and 'output_cap_bytes'
    each of its output streams.  Commands see 'child_environment()':  a
    scrubbed copy of this process's environment, unless 'pass_env'.

    'is_cancelled', if given, is polled while a command runs (e.g., from a
    UI's worker thread, or set when an asyncio task is cancelled):  once
    it returns true, the command is killed and 'CommandCancelled' raised.
    """

    root: pathlib.Path
    allow_anywhere: bool = False
    timeout_secs: float = DEFAULT_TOOL_TIMEOUT_SECS
    output_cap_bytes: int = DEFAULT_OUTPUT_CAP_BYTES
    pass_env: bool = False
    is_cancelled: abc.Callable[[], bool] | None = dataclasses.field(
        default=None,
        compare=False,
    )

    def __post_init__(self):
        root = pathlib.Path(self.root).expanduser().resolve(strict=True)

        if not root.is_dir():
            raise RootNotADirectory(root)

        object.__setattr__(self, "root", root)


#
#   'shell' path check
#
#   A here-document's delimiter word:  '<<EOF', '<<-EOF', "<< 'PY'", ...
_HEREDOC_DELIMITER = re.compile(
    r"[ \t]*((?:[^\s;&|<>()'\"\\]|'[^']*'|\"(?:[^\"\\]|\\.)*\"|\\.)+)"
)
_HEREDOC_DELIMITER_PART = re.compile(r"'([^']*)'|\"((?:[^\"\\]|\\.)*)\"|\\(.)")
#   Unquoted, unescaped characters after which a word starts.
_WORD_BREAKS = frozenset(" \t;&|()<>")


class UnterminatedHereDoc(ValueError):
    def __init__(self, delimiter: str):
        super().__init__(
            f"Refused: here-document not terminated by {delimiter!r}",
        )


@dataclasses.dataclass(frozen=True)
class _HereDoc:
    delimiter: str
    strip_tabs: bool  # '<<-'
    literal: bool  # a quoted delimiter:  the shell expands nothing


def _unquote_delimiter(word: str) -> str:
    def part(match: re.Match) -> str:
        single, double, escaped = match.groups()

        if single is not None:
            return single

        if double is not None:
            return re.sub(r"\\([\\\"$`])", r"\1", double)

        return escaped

    return _HEREDOC_DELIMITER_PART.sub(part, word)


def _here_doc_at(command: str, start: int) -> tuple[_HereDoc, int] | None:
    """Parse the here-document operator at 'start' (at a '<<')"""
    index = start + 2
    strip_tabs = command.startswith("-", index)
    index += strip_tabs

    match = _HEREDOC_DELIMITER.match(command, index)

    if match is None:
        return None

    word = match.group(1)
    literal = any(char in word for char in "'\"\\")
    here_doc = _HereDoc(_unquote_delimiter(word), strip_tabs, literal)

    return here_doc, match.end()


def _here_doc_body(
    command: str,
    start: int,
    here_doc: _HereDoc,
) -> tuple[list[str], int]:
    """Return the checked lines of the body at 'start', and where it ends

    The body of a quoted-delimiter here-document ('<<'EOF'') is literal
    text, so none of it is checked.  An unquoted one's is expanded by the
    shell (after joining lines ending in '\'), so if it holds a command
    substitution ('$(...)' or backquotes), all of it is checked;  else,
    it is data.

    Raises 'UnterminatedHereDoc' if the delimiter never comes:  rather
    than risk taking the rest of the command for the body.
    """
    lines = []

    while start < len(command):
        end = command.find("\n", start)
        end = len(command) if end < 0 else end
        line = command[start:end]
        start = end + 1

        # A line continuation, in an unquoted body.
        while (
            not here_doc.literal
            and line.endswith("\\")
            and (len(line) - len(line.rstrip("\\"))) % 2
            and start < len(command)
        ):
            end = command.find("\n", start)
            end = len(command) if end < 0 else end
            line = line[:-1] + command[start:end]
            start = end + 1

        if (line.lstrip("\t") if here_doc.strip_tabs else line) == (
            here_doc.delimiter
        ):
            break

        lines.append(line)

    else:
        raise UnterminatedHereDoc(here_doc.delimiter)

    body = "\n".join(lines)

    if not here_doc.literal and ("$(" in body or "`" in body):
        return lines, start

    return [], start


_OPERATOR_CHAR = re.compile(r"[;&|()<>]")


def _without_here_doc_bodies(command: str) -> tuple[str, str]:
    """Return a POSIX shell command's text as one line, for the path check

    Drops the bodies of here-documents (see '_here_doc_body'), but keeps
    their operators and delimiters (so '> /outside/x <<EOF' is still
    checked), drops comments, and joins the command's lines with ';' (as
    the shell runs them), so that each line's first word is still the
    start of a command.  Quotes are tracked, so a '<<', '#' or newline
    inside one is left alone.

    Also returns its "shape":  the same text, but with every quoted or
    escaped operator character ('echo ">"') replaced by '_', so that only
    the shell's own operators split it into commands.
    """
    kept: list[tuple[str, bool]] = []  # text, and whether it is quoted
    pending: list[_HereDoc] = []
    quote = None
    word_start = True  # where an unquoted '#' starts a comment
    index = 0

    while index < len(command):
        char = command[index]
        at_word_start, word_start = word_start, False
        quoted = quote is not None

        if quote == "'":
            quote = None if char == "'" else quote
        elif char == "\\":
            if command.startswith("\n", index + 1):  # a line continuation
                word_start = at_word_start
                index += 2
                continue
            kept.append((command[index : index + 2], True))
            index += 2
            continue
        elif quote == '"':
            quote = None if char == '"' else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and at_word_start:
            end = command.find("\n", index)  # the newline is kept
            index = len(command) if end < 0 else end
            continue
        elif command.startswith("((", index):  # arithmetic:  '$((1 << 2))'
            end = command.find("))", index)
            end = len(command) if end < 0 else end + 2
            kept.append((command[index:end], True))  # no operators within
            index = end
            continue
        elif command.startswith("<<<", index):  # a here-string
            kept.append(("<<<", False))
            index += 3
            continue
        elif command.startswith("<<", index):
            found = _here_doc_at(command, index)

            if found is not None:
                here_doc, end = found
                pending.append(here_doc)
                kept.append(("<<", False))
                kept.append((command[index + 2 : end], True))
                index = end
                continue
        elif char == "\n":
            word_start = True
            kept.append((" ; ", False))
            index += 1

            for here_doc in pending:
                checked, index = _here_doc_body(command, index, here_doc)
                kept.extend((f"{line} ; ", False) for line in checked)

            pending.clear()
            continue

        elif char in _WORD_BREAKS:
            word_start = True

        kept.append((char, quoted))
        index += 1

    if pending:  # no newline, so no body
        raise UnterminatedHereDoc(pending[0].delimiter)

    text = "".join(piece for piece, _ in kept)
    shape = "".join(
        _OPERATOR_CHAR.sub("_", piece) if quoted else piece
        for piece, quoted in kept
    )
    return text, shape


def _split_words(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=_POSIX, punctuation_chars=True)
    lexer.whitespace_split = True
    # Comments are dropped beforehand, where the shell has them;  'shlex'
    # would also take a '#' within a word ('a#b') for one.
    lexer.commenters = ""
    return [token.strip("\"'") for token in lexer]


#   'cd' (or 'pushd') with no directory goes to '$HOME', and 'cd -' to
#   '$OLDPWD':  outside the root, though the command names no path.
_CD_COMMANDS = frozenset(["cd", "pushd"])
_CD_OPTION = re.compile(r"-[LPe@]+|--")
_PUNCTUATION = frozenset("();<>|&")
#   Reserved words after which (in a command's place) comes a command.
_COMMAND_PREFIXES = frozenset(
    ["if", "then", "else", "elif", "while", "until", "do", "{", "!", "time"],
)
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=.*")


def _is_separator(token: str) -> bool:
    return set(token) <= _PUNCTUATION and not set(token) & set("<>")


def _is_redirection(token: str) -> bool:
    return set(token) <= _PUNCTUATION and bool(set(token) & set("<>"))


def _simple_commands(tokens: list[str]) -> abc.Iterator[list[str]]:
    """Yield each simple command's words:  its name, then its arguments

    Leaves out redirections ('> file', '2>/dev/null':  a number just
    before a redirection is taken for its descriptor), and the reserved
    words and variable assignments which may come before the name.
    """
    words: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        index += 1

        if _is_separator(token):
            if words:
                yield words
            words = []

        elif _is_redirection(token):
            index += 1  # its target

        elif (
            token.isdigit()
            and index < len(tokens)
            and (_is_redirection(tokens[index]))
        ):
            pass  # a descriptor:  '2>'

        elif not words and (
            token in _COMMAND_PREFIXES or _ASSIGNMENT.fullmatch(token)
        ):
            pass

        else:
            words.append(token)

    if words:
        yield words


def _check_cd(words: list[str]) -> None:
    """Refuse a 'cd' / 'pushd' to '$HOME' or '$OLDPWD'"""
    name, *args = words
    args = [arg for arg in args if not _CD_OPTION.fullmatch(arg)]

    if not args:
        raise CdRefused(name)

    if args[0] == "-":
        raise CdRefused(name, to_oldpwd=True)


def _candidate_paths(token: str) -> list[str]:
    """Return the words in 'token' which the shell may treat as paths"""
    if pathlib.Path(token).is_absolute():
        found = [token]
    else:
        found = _EMBEDDED_PATH.findall(token)

    if not _POSIX:
        found = [path for path in found if not _WINDOWS_SWITCH.fullmatch(path)]

    # Relative paths climbing out of the root, e.g. '../x' or '--in=../x'.
    for piece in token.split("="):
        if ".." in _PATH_SEPARATORS.split(piece):
            found.append(piece)

    return found


def check_command_paths(command: str, root: pathlib.Path) -> None:
    """Refuse a command naming a path outside 'root'

    Expands environment variables and '~' as the shell would, then checks
    every absolute path in the command's words, and every relative one
    which contains '..', following symlinks.  On POSIX, it also refuses a
    'cd' / 'pushd' command to '$HOME' or '$OLDPWD' (no directory, or '-').

    It reads the command itself, including a here-document's operator
    and target ('cat > /outside/x <<EOF'), but not the here-document's
    body, which is data (e.g., the source of a file being written) --
    unless it is unquoted and holds a command substitution (see
    '_here_doc_body').  Comments are skipped.  Paths inside quoted
    words are still checked ('python -c "open('/etc/passwd')"'), unless
    they are part of a URL ('curl "http://host/path"'):  a quoted '/x'
    might be a route, or a file the command opens, and the check cannot
    tell which, so it refuses.

    This is a *lexical* guard rail, not a sandbox:  a command can build
    a path at runtime (command substitution, a script, 'cd'), and nothing
    here limits what the command does with the user's permissions.

    Raises 'PathRefused', or 'ValueError' if the command cannot be split
    into words (e.g., an unclosed quote, or an unterminated here-document).
    """
    # The shell parses the command (its operators, here-documents and
    # comments) before it expands variables in it.
    if _POSIX:
        command, shape = _without_here_doc_bodies(command)

        for words in _simple_commands(_split_words(shape)):
            if words[0] in _CD_COMMANDS:
                _check_cd(words)

    for token in _split_words(os.path.expandvars(command)):
        for found in _candidate_paths(token):
            if found in _HARMLESS_PATHS:
                continue

            try:
                path = pathlib.Path(found).expanduser()  # e.g. '~nobody'
                if not path.is_absolute():
                    path = root / path

                inside = path.resolve().is_relative_to(root)
            except (RuntimeError, OSError):
                inside = False

            if not inside:
                raise PathRefused(found)


#
#   'shell' executor
#
#   Environment variables a command never sees, unless 'pass_env':  this
#   client's own token, and whatever is named like a secret (matched
#   case-insensitively).  'CDPATH' and 'OLDPWD' would let a bare 'cd'
#   leave the root.
SECRET_ENV_PATTERNS = (
    "*TOKEN*",
    "*SECRET*",
    "*PASSWORD*",
    "*PASSWD*",
    "*_KEY",
    "*API_KEY*",
    "AWS_*",
    "GITHUB_*",
    "GH_*",
)
_DROPPED_ENV_NAMES = frozenset(["CDPATH", "OLDPWD"])
#   Dropped even with 'pass_env':  the command has no use for it.
ALWAYS_DROPPED_ENV_NAMES = frozenset(["SOLIPLEX_TOKEN"])


def child_environment(
    environ: typing.Mapping[str, str] | None = None,
    *,
    pass_env: bool = False,
) -> dict[str, str]:
    """The environment a client tool's command runs with

    A copy of 'environ' (default: this process's) without
    'SOLIPLEX_TOKEN';  unless 'pass_env', also without any variable
    matching 'SECRET_ENV_PATTERNS', 'CDPATH' or 'OLDPWD'.  Everything else
    ('PATH', 'HOME', 'LANG', 'TERM', 'SHELL', ...) is kept, so that
    ordinary commands work.
    """
    if environ is None:
        environ = os.environ

    def dropped(name: str) -> bool:
        upper = name.upper()

        if upper in ALWAYS_DROPPED_ENV_NAMES:
            return True

        if pass_env:
            return False

        return upper in _DROPPED_ENV_NAMES or any(
            fnmatch.fnmatchcase(upper, pattern)
            for pattern in SECRET_ENV_PATTERNS
        )

    return {
        name: value for name, value in environ.items() if not dropped(name)
    }


def _kill(proc: subprocess.Popen) -> None:
    """Kill a command, with the processes it started

    'proc' is the shell ('shell=True'), so killing it alone would leave
    the command running.  On POSIX, kill its process group (it leads its
    own session);  on Windows, its process tree.
    """
    if _POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:  # already gone
            pass
    else:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        proc.kill()  # in case 'taskkill' could not


def _read_capped(stream, cap: int) -> tuple[str, int, bool]:
    size = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    data = stream.read(cap)
    text = data.decode("utf-8", errors="replace")
    return text, size, size > cap


#   How often 'run_shell' checks whether the command has exited, or been
#   cancelled.  (A short sleep, rather than one long 'Popen.wait', so that
#   Ctrl-C is handled at once on Windows too, and so that nothing reaps
#   the command before '_kill' has seen to it.)
POLL_SECS = 0.05


def _exited(proc: subprocess.Popen) -> bool:
    """Has 'proc' exited?  On POSIX, leave it unreaped

    An unreaped (zombie) process keeps its ID, and so its process group's:
    '_kill' can then kill the group without risking another process which
    reused the ID.  (Windows:  'proc' holds a handle to the process, which
    keeps its ID from being reused.)
    """
    if not _POSIX:
        return proc.poll() is not None

    try:
        found = os.waitid(
            os.P_PID,
            proc.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:  # reaped already (e.g., SIGCHLD ignored)
        return True

    return found is not None


def _wait(proc: subprocess.Popen, context: ToolContext) -> None:
    """Wait for 'proc' to exit;  raise 'TimeoutExpired' or 'CommandCancelled'

    Does not reap it (see '_exited').
    """
    deadline = time.monotonic() + context.timeout_secs

    while not _exited(proc):
        if context.is_cancelled is not None and context.is_cancelled():
            raise CommandCancelled()

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, context.timeout_secs)

        time.sleep(min(POLL_SECS, remaining))


def run_shell(args: dict, context: ToolContext) -> dict:
    """Execute the 'shell' tool

    Runs 'args["command"]' through the system shell, in 'context.root',
    with no stdin and 'child_environment()', killing it (and every
    process it started) after 'context.timeout_secs' -- or at once if
    'context.is_cancelled()' (raising 'CommandCancelled'), or if anything
    else interrupts the wait (e.g., 'KeyboardInterrupt' on Ctrl-C, which
    does not reach the command:  it runs in its own session).  Output goes
    to temporary files rather than
    pipes, so a runaway command cannot exhaust memory, and a grandchild
    holding the output open cannot hang us;  each stream is capped at
    'context.output_cap_bytes'.

    Returns '{stdout, stderr, exit_code, timed_out, truncated}', plus the
    streams' full sizes for the tool log.  A non-zero exit is data for the
    model, not an error.
    """
    command = args.get("command")

    if not isinstance(command, str) or not command.strip():
        raise InvalidShellArgs()

    if not context.allow_anywhere:
        check_command_paths(command, context.root)

    if context.is_cancelled is not None and context.is_cancelled():
        raise CommandCancelled()

    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=context.root,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=child_environment(pass_env=context.pass_env),
            start_new_session=True,  # own process group, to kill it all
        )
        exited = timed_out = False

        try:
            _wait(proc, context)
            exited = True
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Whatever else ended the wait -- a timeout, a cancellation,
            # Ctrl-C, any other exception -- the command must not outlive
            # it, even if it exited meanwhile, leaving processes behind.
            if not exited:
                _kill(proc)

            exit_code = proc.wait()  # reap it

        cap = context.output_cap_bytes
        stdout, stdout_bytes, stdout_cut = _read_capped(stdout_file, cap)
        stderr, stderr_bytes, stderr_cut = _read_capped(stderr_file, cap)

    if timed_out:
        stderr += f"\n[timed out after {context.timeout_secs:g} seconds]"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": stdout_cut or stderr_cut,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
    }


#
#   Tool registry
#
ToolExecutor = abc.Callable[[dict, ToolContext], dict]


@dataclasses.dataclass(frozen=True)
class ClientTool:
    name: str
    description: str
    parameters: dict
    execute: ToolExecutor

    def as_agui_tool(self) -> agui_core.Tool:
        return agui_core.Tool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


SHELL_TOOL = ClientTool(
    name="shell",
    description=(
        "Run a shell command on the user's machine, in the user's chosen "
        "working directory.  Returns the command's stdout, stderr and "
        "exit code (long output is truncated).  Commands naming paths "
        "outside the working directory may be refused, and there is no "
        "stdin."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command line to run",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    execute=run_shell,
)

ClientTools = typing.Mapping[str, ClientTool]

CLIENT_TOOLS: ClientTools = {SHELL_TOOL.name: SHELL_TOOL}


def agui_tools(tools: ClientTools = CLIENT_TOOLS) -> list[agui_core.Tool]:
    """AG-UI 'Tool' definitions, for 'RunAgentInput.tools'"""
    return [tool.as_agui_tool() for tool in tools.values()]


#
#   Executing one tool call
#
_RESULT_KEYS = ("stdout", "stderr", "exit_code", "timed_out", "truncated")


def _not_run(message: str) -> dict:
    return {
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "timed_out": False,
        "truncated": False,
        "error": message,
    }


def parse_tool_args(arguments: str) -> typing.Any:
    """Parse a tool call's JSON arguments;  return the raw text if bad"""
    try:
        return json.loads(arguments) if arguments else {}
    except ValueError:
        return arguments


Confirm = abc.Callable[[str, typing.Any], bool]


class Approval(enum.Enum):
    """A user's answer to 'run this tool call?'"""

    DECLINE = "decline"
    RUN_ONCE = "run"
    RUN_ALL = "run-all"  # this call, and every later one this session

    @classmethod
    def of(cls, answer: typing.Any) -> Approval:
        """An 'Approval', from one or from a yes / no (e.g., None:  no)"""
        if isinstance(answer, cls):
            return answer

        return cls.RUN_ONCE if answer else cls.DECLINE


Reply = abc.Callable[[Approval | bool | None], None]

AUTO_APPROVE_NOTICE = "auto-approve ON: commands run without asking"


@dataclasses.dataclass
class AutoApprove:
    """Whether client tool calls run without asking, for a UI's session

    On from the start ('enabled'), or once the user answers 'RUN_ALL';
    'on_enabled' is called then (from the thread asking), e.g. for the UI
    to show 'AUTO_APPROVE_NOTICE' for the rest of the session.  Only the
    question goes:  the path check, the scrubbed environment, the timeout
    and the tool log apply as ever.
    """

    enabled: bool = False
    on_enabled: abc.Callable[[], None] | None = None

    def enable(self) -> None:
        if self.enabled:
            return

        self.enabled = True

        if self.on_enabled is not None:
            self.on_enabled()


def describe_tool_call(name: str, args: typing.Any) -> str:
    """One line saying what a tool call would do, for a confirmation"""
    if name == SHELL_TOOL.name and isinstance(args, dict):
        return f"{name}: {args.get('command')}"

    return f"{name}: {json.dumps(args)}"


#   How much of a call's output a UI shows:  its first lines, cut short.
PREVIEW_LINES = 20
PREVIEW_LINE_CHARS = 200


def _preview(text: str, max_lines: int) -> list[str]:
    lines = text.splitlines()
    shown = [
        line
        if len(line) <= PREVIEW_LINE_CHARS
        else line[:PREVIEW_LINE_CHARS] + " [...]"
        for line in lines[:max_lines]
    ]

    if len(lines) > max_lines:
        shown.append(f"[... {len(lines) - max_lines} more lines]")

    return shown


def describe_tool_result(record: ToolCallRecord, result: dict) -> str:
    """A few words saying how a tool call ended, e.g. 'ran, exit code 0'

    Made of fixed words (and the exit code) only:  nothing the model or
    the command chose.
    """
    if result.get("killed"):
        return "killed while it ran (it may have made changes)"

    if result.get("error"):
        return "not run"

    if result.get("timed_out"):
        return "ran, and timed out"

    return f"ran, exit code {record.exit_code}"


def render_tool_result(
    record: ToolCallRecord,
    result: dict,
    *,
    max_lines: int = PREVIEW_LINES,
) -> str:
    """Markdown for a UI's chat:  the call, how it ended, and its output

    A heading of fixed words ('describe_tool_result'), then a code block
    holding everything the model or the command chose -- the call itself,
    why it did not run, and the first 'max_lines' lines of each of stdout
    and stderr (each line cut at 'PREVIEW_LINE_CHARS') -- behind a fence
    longer than any run of backquotes in it, so none of it is markup.
    """
    heading = (
        f"** client tool call -- {describe_tool_result(record, result)} **"
    )

    if record.name == SHELL_TOOL.name and isinstance(record.args, dict):
        call = [f"$ {record.args.get('command')}"]
    else:
        call = [describe_tool_call(record.name, record.args)]

    block = _preview("\n".join(call), max_lines)

    if result.get("error"):
        block.append(f"[error] {result['error']}")

    for stream in ("stdout", "stderr"):
        shown = _preview(result.get(stream) or "", max_lines)

        if shown:
            block.extend([f"[{stream}]", *shown])

    if result.get("truncated"):
        block.append("[output truncated before it reached the model]")

    body = "\n".join(block)
    longest = max(
        (len(run) for run in re.findall(r"`+", body)),
        default=0,
    )
    fence = "`" * max(3, longest + 1)

    return f"\n\n{heading}\n\n{fence}text\n{body}\n{fence}"


def confirm_via_callback(
    request: abc.Callable[[str, typing.Any, Reply], None],
    *,
    is_cancelled: abc.Callable[[], bool] = lambda: False,
    poll_secs: float = 0.1,
    auto_approve: AutoApprove | None = None,
) -> Confirm:
    """Adapt a callback-style confirmation (e.g., a UI dialog) to 'Confirm'

    'run_loop' asks for confirmation synchronously, from the thread it
    runs in;  a UI answers later, on its own thread.  The returned
    'Confirm' calls 'request(name, args, reply)' (which should show the
    question and return at once) and waits until 'reply' is called with
    the answer.

    If 'is_cancelled()' is true (e.g., the UI is shutting down) before
    asking, while waiting, or once answered, it raises
    'ConfirmationCancelled', which ends 'run_loop':  a cancelled caller
    never runs a call, whatever the answer.

    The answer is an 'Approval' (or a yes / no).  With 'auto_approve',
    'RUN_ALL' enables it, and once it is enabled nothing is asked:  every
    call is approved (unless cancelled).  Without, 'RUN_ALL' is 'RUN_ONCE'.
    """

    def confirm(name: str, args: typing.Any) -> bool:
        if is_cancelled():
            raise ConfirmationCancelled()

        if auto_approve is not None and auto_approve.enabled:
            return True

        answered = threading.Event()
        answers: list[Approval] = []

        def reply(answer: Approval | bool | None) -> None:
            answers.append(Approval.of(answer))
            answered.set()

        request(name, args, reply)

        while not answered.wait(poll_secs):
            if is_cancelled():
                raise ConfirmationCancelled()

        if is_cancelled():
            raise ConfirmationCancelled()

        (answer,) = answers

        if answer is Approval.RUN_ALL and auto_approve is not None:
            auto_approve.enable()

        return answer is not Approval.DECLINE

    return confirm


def execute_tool_call(
    name: str,
    args: typing.Any,
    context: ToolContext,
    *,
    tools: ClientTools = CLIENT_TOOLS,
    confirm: Confirm | None = None,
) -> dict:
    """Execute one client tool call, returning its result as data

    Every failure -- an unknown tool, bad arguments, a refused path, a
    declined confirmation, an OS error -- comes back as a result with
    'exit_code' None and an 'error', so the model can see it and adjust,
    and the history stays consistent.
    """
    tool = tools.get(name)

    if tool is None:
        return _not_run(f"Unknown client tool: {name!r}")

    if not isinstance(args, dict):
        return _not_run(f"Arguments for {name!r} must be a JSON object")

    if confirm is not None and not confirm(name, args):
        return _not_run("The user declined to run this tool call")

    try:
        return tool.execute(args, context)
    except (ValueError, OSError) as exc:
        return _not_run(str(exc))


def tool_result_content(result: dict) -> str:
    """The tool message content sent to the model:  the result as JSON

    Leaves out the full output sizes, which are for the local log.
    """
    sent = {key: result[key] for key in _RESULT_KEYS if key in result}

    if "error" in result:
        sent["error"] = result["error"]

    return json.dumps(sent)


#
#   Local tool log
#
@dataclasses.dataclass
class ToolLog:
    """Append one JSON line per client tool call to a local file

    Records what ran, where, how it ended, and how much output it made;
    not the output itself.
    """

    path: pathlib.Path

    def record(
        self,
        *,
        name: str,
        args: typing.Any,
        context: ToolContext,
        result: dict,
        duration_secs: float,
    ) -> None:
        entry = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "tool": name,
            "args": args,
            "cwd": str(context.root),
            "exit_code": result["exit_code"],
            "duration_secs": round(duration_secs, 3),
            "timed_out": result["timed_out"],
            "truncated": result["truncated"],
            "stdout_bytes": result.get("stdout_bytes", 0),
            "stderr_bytes": result.get("stderr_bytes", 0),
            "error": result.get("error"),
        }

        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")


#
#   SSE / HTTP
#
def iter_sse_json(
    lines: abc.Iterable[str | bytes],
) -> abc.Iterator[dict]:
    """Decode the JSON 'data' of each server-sent event

    'lines' are the stream's lines without their line endings (as from
    'httpx.Response.iter_lines' or 'requests.Response.iter_lines').
    Skips comments (keepalives), and the 'id' / 'event' / 'retry' fields;
    joins multi-line data.  An event still open when the stream ends is
    incomplete, and dropped (SSE spec).
    """
    data: list[str] = []

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if not line:
            if data:
                yield json.loads("\n".join(data))
                data.clear()

        elif line.startswith("data:"):
            data.append(line[len("data:") :].removeprefix(" "))


@dataclasses.dataclass
class SoliplexClient:
    """A minimal client for one room's AG-UI endpoints

    'http' is any 'httpx.Client' (e.g., a FastAPI 'TestClient');  by
    default, one is made for 'url', sending 'token' as a bearer token.
    """

    url: str
    room_id: str
    token: dataclasses.InitVar[str | None] = None
    http: httpx.Client | None = None

    def __post_init__(self, token):
        if self.http is None:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            self.http = httpx.Client(
                headers=headers,
                timeout=DEFAULT_HTTP_TIMEOUT,
            )

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @property
    def agui_url(self) -> str:
        room = urllib_parse.quote(self.room_id, safe="")
        return f"{self.url.rstrip('/')}/api/v1/rooms/{room}/agui"

    @staticmethod
    def _check(response: httpx.Response) -> None:
        if response.is_error:
            response.read()
            raise HTTPFailure(response)

    def _post(self, url: str, body: dict) -> dict:
        try:
            response = self.http.post(url, json=body)
        except httpx.HTTPError as exc:
            raise TransportFailure(exc) from exc

        self._check(response)
        return response.json()

    def new_thread(self, metadata: dict | None = None) -> dict:
        body = {"metadata": metadata} if metadata else {}
        return self._post(self.agui_url, body)

    def new_run(self, thread_id: str, parent_run_id: str) -> dict:
        return self._post(
            f"{self.agui_url}/{thread_id}",
            {"parent_run_id": parent_run_id},
        )

    def stream_run(
        self,
        run_input: agui_core.RunAgentInput,
    ) -> abc.Iterator[agui_core.Event]:
        url = f"{self.agui_url}/{run_input.thread_id}/{run_input.run_id}"

        try:
            with self.http.stream(
                "POST",
                url,
                json=run_input.model_dump(mode="json", by_alias=True),
                headers={"Accept": "text/event-stream"},
            ) as response:
                self._check(response)

                for event_json in iter_sse_json(response.iter_lines()):
                    yield agui_parser.agui_event_from_json(event_json)

        except httpx.HTTPError as exc:  # e.g., the read timed out
            raise TransportFailure(exc) from exc


def initial_run_input(
    thread: dict,
    prompt: str,
    tools: ClientTools = CLIENT_TOOLS,
) -> agui_core.RunAgentInput:
    """The first run's input, for a thread just made by 'new_thread'"""
    ((run_id, run),) = thread["runs"].items()
    run_input = run.get("run_input") or {}

    return agui_core.RunAgentInput(
        thread_id=thread["thread_id"],
        run_id=run_id,
        state=run_input.get("state") or {},
        messages=[
            agui_core.UserMessage(id=uuid.uuid4().hex, content=prompt),
        ],
        tools=agui_tools(tools),
        context=[],
        forwarded_props={},
    )


#
#   Run loop
#
#   The state sent back after a run whose 'STATE_DELTA's could not be
#   applied (and no 'STATE_SNAPSHOT' healed them):  see #1260.
INVALID_STATE = {"error": "'STATE_DELTA' without final 'STATE_SNAPSHOT'"}


@dataclasses.dataclass
class ToolCallRecord:
    name: str
    args: typing.Any
    exit_code: int | None


@dataclasses.dataclass
class LoopResult:
    thread_id: str
    run_ids: list[str] = dataclasses.field(default_factory=list)
    response: str = ""
    tool_calls: list[ToolCallRecord] = dataclasses.field(
        default_factory=list,
    )
    run_input: agui_core.RunAgentInput | None = None

    def as_json(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "run_ids": self.run_ids,
            "response": self.response,
            "tool_calls": [
                dataclasses.asdict(call) for call in self.tool_calls
            ],
        }


def pending_tool_calls(
    messages: abc.Sequence[agui_core.Message],
) -> list[agui_core.ToolCall]:
    """Tool calls in 'messages' which have no tool result yet"""
    answered = {
        message.tool_call_id
        for message in messages
        if isinstance(message, agui_core.ToolMessage)
    }
    return [
        call
        for message in messages
        if isinstance(message, agui_core.AssistantMessage)
        for call in message.tool_calls or ()
        if call.id not in answered
    ]


def _parse_run(
    client: SoliplexClient,
    run_input: agui_core.RunAgentInput,
    on_event: abc.Callable[[agui_core.Event], None] | None,
) -> agui_core.RunAgentInput:
    esp = agui_parser.EventStreamParser(run_input)

    try:
        for event in client.stream_run(run_input):
            # AG-UI allows a tool call without a parent message, but the
            # parser keeps a call in the history only under one.
            if (
                event.type == agui_core.EventType.TOOL_CALL_START
                and event.parent_message_id is None
            ):
                event.parent_message_id = uuid.uuid4().hex

            esp(event)

            if on_event is not None:
                on_event(event)

    except ValueError as exc:  # incl. parser and JSON errors
        raise InvalidStream(exc) from exc

    if esp.run_status == agui_parser.RunStatus.ERROR:
        raise RunErrored(esp.error_message)

    if esp.run_status != agui_parser.RunStatus.FINISHED:
        raise RunNotFinished()

    if esp.active_tool_calls:
        raise IncompleteToolCall()

    if esp.invalid_state_deltas:  # see #1260
        esp.state = dict(INVALID_STATE)

    return esp.as_run_agent_input


def run_loop(
    client: SoliplexClient,
    run_input: agui_core.RunAgentInput,
    context: ToolContext,
    *,
    tools: ClientTools = CLIENT_TOOLS,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_turns_option: str | None = None,
    confirm: Confirm | None = None,
    on_event: abc.Callable[[agui_core.Event], None] | None = None,
    on_tool_result: (abc.Callable[[ToolCallRecord, dict], None] | None) = None,
    tool_log: ToolLog | None = None,
) -> LoopResult:
    """Run 'run_input', executing client tool calls, until a final answer

    After each run whose model called client tools, execute them, append
    their results to the history (which already holds the assistant's
    calls), make a child run with 'parent_run_id', and start it with that
    history.  Stop when a run ends with no pending call, or raise
    'MaxTurnsExceeded' when 'max_turns' runs have not been enough (before
    executing the calls it cannot send back);  its message names
    'max_turns_option', the front end's setting for the limit, if given.

    A pending call to a tool the server's agent does not run and we do
    not know still gets a result (an error), so the history stays
    consistent.

    'on_event' sees every event of every run;  'on_tool_result' each
    executed (or refused) call and its result, e.g. for a UI to show.

    Returns the result, whose 'run_input' is the final history (e.g., for
    the TUI's next prompt).  A 'ClientToolsError' carries the result so
    far as its 'result' (see there).
    """
    result = LoopResult(thread_id=run_input.thread_id, run_input=run_input)

    try:
        _run_loop(
            result,
            client,
            run_input,
            context,
            tools=tools,
            max_turns=max_turns,
            max_turns_option=max_turns_option,
            confirm=confirm,
            on_event=on_event,
            on_tool_result=on_tool_result,
            tool_log=tool_log,
        )
    except ClientToolsError as exc:
        exc.result = result
        raise

    return result


def _run_loop(
    result: LoopResult,
    client: SoliplexClient,
    run_input: agui_core.RunAgentInput,
    context: ToolContext,
    *,
    tools: ClientTools,
    max_turns: int,
    max_turns_option: str | None,
    confirm: Confirm | None,
    on_event: abc.Callable[[agui_core.Event], None] | None,
    on_tool_result: abc.Callable[[ToolCallRecord, dict], None] | None,
    tool_log: ToolLog | None,
) -> None:
    turn = 0

    while True:
        turn += 1
        result.run_ids.append(run_input.run_id)
        previous_ids = {message.id for message in run_input.messages}

        run_input = _parse_run(client, run_input, on_event)
        pending = pending_tool_calls(run_input.messages)

        if not pending:
            result.response = "".join(
                message.content or ""
                for message in run_input.messages
                if isinstance(message, agui_core.AssistantMessage)
                and message.id not in previous_ids
            )
            result.run_input = run_input
            return

        if turn >= max_turns:
            raise MaxTurnsExceeded(max_turns, max_turns_option)

        cancelled = None

        for call in pending:
            args = parse_tool_args(call.function.arguments)
            started = time.monotonic()

            if cancelled is not None:  # answer the rest, but run nothing
                tool_result = _not_run(ConfirmationCancelled.call_outcome)
            else:
                try:
                    tool_result = execute_tool_call(
                        call.function.name,
                        args,
                        context,
                        tools=tools,
                        confirm=confirm,
                    )
                except Cancelled as exc:  # in confirming, or while it ran
                    cancelled = exc
                    tool_result = _not_run(exc.call_outcome)
                    tool_result["killed"] = isinstance(exc, CommandCancelled)

            if tool_log is not None:
                tool_log.record(
                    name=call.function.name,
                    args=args,
                    context=context,
                    result=tool_result,
                    duration_secs=time.monotonic() - started,
                )

            record = ToolCallRecord(
                name=call.function.name,
                args=args,
                exit_code=tool_result["exit_code"],
            )
            result.tool_calls.append(record)

            if on_tool_result is not None:
                on_tool_result(record, tool_result)

            run_input.messages.append(
                agui_core.ToolMessage(
                    id=uuid.uuid4().hex,
                    tool_call_id=call.id,
                    content=tool_result_content(tool_result),
                ),
            )

        # Every call has its result:  a consistent history to carry on
        # from, should what follows fail.
        result.run_input = run_input

        if cancelled is not None:
            raise cancelled

        new_run = client.new_run(
            run_input.thread_id,
            parent_run_id=run_input.run_id,
        )
        run_input = run_input.model_copy(
            update={
                "run_id": new_run["run_id"],
                "parent_run_id": run_input.run_id,
            },
        )
