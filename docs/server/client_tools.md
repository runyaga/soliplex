# Client-Side Tools

A room's model can call tools that run on the **user's own machine**,
outside the Soliplex server. The client advertises them with each run;
the server lets the model call them, but never runs them itself. Soliplex
ships one such tool, `shell`, used by
[`soliplex-cli ask --url`](cli.md#remote-mode-client-tools) and by the
[TUI](../tui.md#client-tools).

The client side lives in `soliplex.agui.client_tools`: the tool registry,
the `shell` executor, and the run loop described below.

## How the server receives client tools

1. The client sends its tool definitions (name, description, JSON schema
   of the arguments) in `RunAgentInput.tools`, in the body of
   `POST /api/v1/rooms/{room_id}/agui/{thread_id}/{run_id}`.
2. pydantic-ai's `AGUIAdapter` turns them into an *external* toolset
   (`AGUIAdapter.toolset`), added to the room agent's own tools for that
   run, and adds `DeferredToolRequests` to the agent's output types.
3. When the model calls one of them, pydantic-ai does not execute it:
   the run ends with the call **deferred**. The SSE stream carries the
   call (`TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END`) with no
   `TOOL_CALL_RESULT`, then `RUN_FINISHED`.
4. Tools the server runs itself (the room's configured tools) still run
   in the same turn; their results appear in the stream as
   `TOOL_CALL_RESULT`. A model can call both kinds at once.

Nothing in a room's configuration enables this: any room accepts the tools
a client sends.

## The round-trip rule

To hand a tool's result back, the client:

1. executes each call that has no result yet;
2. appends the results to the history as `ToolMessage`s (`role: "tool"`,
   `tool_call_id` = the call's id);
3. creates a child run with
   `POST /api/v1/rooms/{room_id}/agui/{thread_id}` and body
   `{"parent_run_id": "<the run just finished>"}`;
4. starts that run with the whole history.

**The history must carry BOTH the assistant message holding the tool call
AND the tool message holding its result.** The server rebuilds the model's
history from the messages the client sends; a result whose call is missing
fails the run with `RUN_ERROR` (`Tool call with ID ... not found in the
history`).

`soliplex.agui.parser.EventStreamParser` builds exactly that history from a
run's events (`as_run_agent_input`), so the loop in `client_tools.run_loop`
is: parse the run, execute its pending calls, append their results, start
the child run, and repeat until a run ends with no pending call (the final
answer), or `--max-turns` runs have been made.

Every outcome goes back to the model as data, as the tool message's JSON
content:

```json
{"stdout": "...", "stderr": "...", "exit_code": 0,
 "timed_out": false, "truncated": false}
```

A non-zero exit is just an `exit_code`. A call that did not run (an
unknown tool, bad arguments, a refused path, a declined confirmation)
has `"exit_code": null` and an `"error"`. Even a call to a tool the
client does not know gets such a result, so that the history stays
consistent.

## Safety model

The `shell` tool runs commands with the **user's own permissions and
network access**. Its safeguards:

- **Working directory.** Commands run in `--root` (default: the current
  directory).
- **Path check.** Unless `--allow-anywhere` is given, a command is
  refused if any of its words names a path outside `--root`: an absolute
  path, a `~` or `$VAR` path (expanded first), or a relative path with
  `..` — following symlinks. On POSIX, a `cd` (or `pushd`) with no
  directory (to `$HOME`) or to `-` (`$OLDPWD`) is refused too.
  `/dev/null`, the `//host` of a URL, and (on Windows) a `cmd.exe` switch
  such as `/s` are not taken for paths. The check errs on the side of
  refusing: a word that merely looks like an outside path (`echo /etc`)
  is refused too, and needs `--allow-anywhere`.
- **Here-documents.** The check reads the command, including a
  here-document's operator and its target (`cat > /outside/x <<EOF` is
  refused), but not its body, which is data — e.g. the source of a file
  being written, whose `@app.route('/contacts')` names no file. A quoted
  here-document (`<<'EOF'`) is literal, and its body skipped; an unquoted
  one (`<<EOF`) is expanded by the shell, so if its body holds a command
  substitution (`$(...)` or backquotes), all of it is checked. A
  here-document whose delimiter never comes is refused, as are commands
  the check cannot split into words (an unclosed quote). Comments are
  skipped.
- **Quoted words are still checked.** A path inside a quoted word
  (`python -c "open('/etc/passwd')"`) is refused, unless it is part of
  a URL (`curl 'http://localhost:5000/contacts'` passes). The check
  cannot tell a route (`grep "route('/contacts')" app.py`) from a file
  the command opens, so it refuses both: rephrase the command, or use
  `--allow-anywhere`.
- **Scrubbed environment.** Commands never see `SOLIPLEX_TOKEN`, nor
  (unless `--pass-env`) variables named like secrets (`*TOKEN*`,
  `*SECRET*`, `*PASSWORD*`, `*_KEY`, `AWS_*`, `GITHUB_*`, ...); see the
  [limitations](cli.md#limitations).
- **No stdin.** Commands read end-of-file from stdin, so nothing can wait
  for input.
- **Timeout and interruption.** A command is killed, with every process
  it started (its process group on POSIX, its process tree on Windows),
  after `--tool-timeout` seconds (default 60) — and at once if the client
  is interrupted (Ctrl-C) or cancelled while it runs.
- **Output cap.** Each of stdout and stderr is capped (16 KiB by default,
  `--output-cap-bytes`) before it is sent to the model; by default the
  start and the end of the stream are kept, with a
  `...[N bytes omitted]...` line between (`--output-cap-mode head` keeps
  the start only). The result says `"truncated": true`.
- **Confirmation.** `soliplex-cli ask --url --confirm` asks before each
  call; the TUI asks in a dialog, unless auto-approve is on (see
  [Auto-approve](../tui.md#auto-approve)).
- **Tool log.** `--tool-log PATH` appends one JSON line per call: what
  ran, where, how it ended, how long it took, and its output sizes (not
  the output).

The path check is a **guard rail, not a sandbox**: it reads the command's
text, and a command can still build a path at runtime (command
substitution, a script, `cd`). See the
[limitations](cli.md#limitations) of `ask --url`.
