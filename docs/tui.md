# Terminal UI (TUI)

Soliplex ships a terminal client built with
[Textual](https://textual.textualize.io/) for interacting with rooms from the
command line -- a quick way to exercise a running backend without the Flutter
[client](client.md). Two entry points are provided:

- `soliplex-tui` -- the interactive terminal client.
- `soliplex-tui-serve` -- serves that same client as a web application (over
  [textual-serve](https://github.com/Textualize/textual-serve)) so it can be
  reached from a browser.

## Installation

The TUI's dependencies live in the `tui` dependency group (Textual,
textual-serve, textual-fspicker, and Typer):

```bash
uv sync --group tui
```

## Running the client

```bash
soliplex-tui
```

By default the client connects to a backend at <http://127.0.0.1:8000>. Point
it at a different server with `--url`:

```bash
soliplex-tui --url https://soliplex.example.com
```

Options:

- `--url URL` -- base URL of the Soliplex backend
  (default: `http://127.0.0.1:8000`)
- `-v` / `--verbose` -- enable verbose output
- `--root PATH` -- working directory for [client tools](#client-tools)
  (default: the current directory)
- `--allow-anywhere` -- do not refuse client tool calls naming paths outside
  `--root`
- `--tool-timeout SECS` -- kill a client tool call after this long
  (default: 60)
- `--client-tools` / `--no-client-tools` -- advertise client tools to rooms
  (default: on)
- `--max-turns N` -- the most runs (model turns) to make for one prompt
  (default: 10); if the model still calls tools on the last one, the TUI
  stops without running them
- `--pass-env` -- run client tool calls with your full environment (by
  default, variables named like secrets are left out; see the
  [limitations](server/cli.md#limitations))
- `--tool-log PATH` -- append one JSON line per client tool call to `PATH`
  (as for [`ask --url`](server/cli.md#remote-mode-client-tools))
- `--output-cap-bytes N` -- the most bytes of each of a client tool call's
  stdout and stderr sent to the model (default: 16384; at least 256;
  `SOLIPLEX_TUI_OUTPUT_CAP_BYTES`)
- `--output-cap-mode head|head_tail` -- how longer output is cut: keep its
  start only, or its start and its end (60% / 40% of the room), with a
  `...[N bytes omitted]...` line between -- the line counts toward the
  cap, and no character is cut in two (default: `head_tail`, since errors
  are usually at the end; `SOLIPLEX_TUI_OUTPUT_CAP_MODE`)
- `--pairing-check/--no-pairing-check`, `--compaction off|auto|always`,
  `--keep-recent N`, `--min-elide-chars N`, `--compaction-trigger F`,
  `--compaction-target F`, `--context-window N`, `--probe-model-window`,
  `--output-reserve N`, `--trim-rag-state off|boundary|aggressive`,
  `--harness-log PATH` -- the [context harness](tui_harness.md), whose
  context meter the room view shows
- `--input-mode multi|single` -- the prompt (default: `multi`;
  `SOLIPLEX_TUI_INPUT_MODE`): see [The prompt](#the-prompt)
- `--auto-approve` / `--yolo` -- run client tool calls **without asking**
  (default: off; see [Auto-approve](#auto-approve))
- `-V` / `--version` -- print the version and exit
- `-h` / `--help` -- show help and exit

### Authentication

On startup the TUI asks the backend which OIDC providers are configured
(`GET /api/login`):

- If the backend defines one or more providers, the TUI prompts you to pick
  one and enter your username and password, then uses the resulting token for
  all subsequent requests.
- If the backend exposes no providers -- for example, it was started with
  `--no-auth-mode` (see [Server Setup](server/index.md)) -- the TUI skips the
  login step and goes straight to the room list.

## Using the client

After authentication (if any), the TUI opens the room list; select a room to
open its chat view. The TUI is a thin client over the same `/api/v1`
endpoints the Flutter client uses, so the actions available mirror the REST
API. Most are reachable from the footer key bindings.

Global:

- `ctrl+n` -- view the installation configuration
- `ctrl+q` -- quit
- `ctrl+\` -- open the command palette

In a room:

- `ctrl+n` -- start a new thread
- `ctrl+t` -- list the room's threads
- `ctrl+r` -- list runs
- `ctrl+s` -- view the current AG-UI state
- `ctrl+z` -- edit thread metadata
- `ctrl+p` -- request an MCP token for the room
- `shift+ctrl+u` -- upload a file
- `esc` -- go back

Viewing a run:

- `ctrl+f` -- submit feedback on the run
- `ctrl+z` -- edit run metadata

### The prompt

The prompt is multi-line (`--input-mode multi`, the default):

- `enter` -- send it;
- `ctrl+j` or `alt+enter` -- start a new line (most terminals cannot tell
  `shift+enter` from `enter`);
- a paste keeps its newlines: the whole text is sent (a single-line
  prompt keeps only a paste's first line);
- `ctrl+e` -- edit it in `$VISUAL` / `$EDITOR` (the TUI is suspended
  until the editor exits).

`--input-mode single` keeps the older single-line prompt.

Under the room and thread names, the room view shows the
[context meter](tui_harness.md#the-context-meter). Typed as a prompt,
`/context` and `/compact` are handled by the TUI itself: see
[TUI Context Harness](tui_harness.md#the-context-meter).

## Client tools

The TUI advertises the same [client-side tools](server/client_tools.md) as
`soliplex-cli ask --url`: a room's model may call `shell`, which runs a
command **on the machine running the TUI**, in `--root`.

**The TUI asks first**, unless auto-approve is on. Before each call it
shows a dialog with the command and the directory it would run in:

- `y` (or **Run**) runs this call;
- `a` (or **Run all**) runs this call, and every later one for the rest of
  the session, without asking (auto-approve);
- `n` / `esc` (or **Decline**) refuses it -- **Decline** has the focus. A
  declined call goes back to the model as a refusal, so it can carry on
  without it.

The response then shows each call: how it ended (`ran, exit code N`,
`not run`, `ran, and timed out`, or `killed while it ran`), then, in a
code block, the command, why it did not run, and the first 20 lines of
its stdout and stderr (long lines cut short). The model receives up to
`--output-cap-bytes` (16 KiB) of each: its start and its end.

The same safeguards as for `ask --url` apply: no stdin, a timeout (and a
command still running when the TUI quits is killed), capped output, a
scrubbed environment, and a path check that refuses commands naming paths
outside `--root` -- a guard rail, **not a sandbox**. See the
[limitations](server/cli.md#limitations). Start the TUI with
`--no-client-tools` to advertise none.

### Auto-approve

With `--auto-approve` (or `--yolo`), or once you answer **Run all**, the
model's commands run **without asking**, for the rest of the session. The
header then says `auto-approve ON: commands run without asking`. Only the
question goes: the path check, the scrubbed environment, the timeout and
`--tool-log` still apply.

**The risk:** the room's model -- and anything it reads (documents, web
pages, the output of earlier commands) -- then decides what runs on your
machine, as you, with your network access, and nothing stops a command
the path check does not catch (see the
[limitations](server/cli.md#limitations)). Use it only with a room and
inputs you trust, with `--root` a scratch directory, ideally inside a
container or VM, and keep a `--tool-log`.

## Serving the TUI over the web

`soliplex-tui-serve` wraps the client with
[textual-serve](https://github.com/Textualize/textual-serve), running it as a
web application you can open in a browser -- handy for demos, or for users who
cannot install the client locally:

```bash
soliplex-tui-serve
```

This serves at <http://127.0.0.1:8002> by default and connects to a backend
at <http://127.0.0.1:8000>. The served client runs with `--no-client-tools`:
client tools would run on the serving host, for anyone who can reach the
page. Options:

- `--backend-url URL` -- base URL of the Soliplex backend
  (default: `http://127.0.0.1:8000`)
- `--host HOST` -- interface to bind (default: `127.0.0.1`)
- `--port PORT` -- port to listen on (default: `8002`)
- `--public-url URL` -- publicly reachable URL of the served app
  (defaults to `http://{host}:{port}`)

For example, to serve on all interfaces behind a known public URL:

```bash
soliplex-tui-serve \
  --host 0.0.0.0 \
  --port 8002 \
  --backend-url https://soliplex.example.com \
  --public-url https://tui.example.com
```
