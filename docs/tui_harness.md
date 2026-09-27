# TUI Context Harness

`soliplex-tui` and `soliplex-cli ask --url` own their thread's history:
they send every message, and the AG-UI state, with each run, and the
room's model reads all of it on every request. In a long, tool-heavy
thread that history is what fills the model's context window. The
*context harness* is what these clients do to the history before each
POST. The logic lives in `soliplex.agui.harness`, so both clients share
it; the TUI only adds its display.

Every feature is behind an option, which both clients accept, and which
can also be set through a `SOLIPLEX_TUI_*` environment variable.

## Pairing check

`--pairing-check` / `--no-pairing-check` (default: on;
`SOLIPLEX_TUI_PAIRING_CHECK`).

Before each POST, the client checks that the history's tool calls and
results pair up:

- tool call ids are unique;
- every tool result answers a call made by an **earlier** assistant
  message;
- no call has more than one result;
- every call has its result.

A history which fails is **not sent**: the run fails locally with
`Refusing to send an inconsistent history: ...`, naming each problem.
The server could not load such a history (pydantic-ai fails the run with
`Tool call with ID ... not found in the history`), and it silently drops
a trailing call which has no result.
