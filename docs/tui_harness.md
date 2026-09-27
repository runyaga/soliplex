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

## Compaction of old tool results

`--compaction off|always` (default: `off`; `SOLIPLEX_TUI_COMPACTION`),
`--keep-recent N` (default: 4), `--min-elide-chars N` (default: 1024).

Most of a RAG-heavy thread's history is tool results: each haiku.rag
`search` returns five context-expanded hits, about 10 KB, and every
later request carries all of them again. With compaction on, the client
rewrites the **content** of old, large tool results before each POST:

- only tool results are touched, and only their content: no message is
  removed, merged or moved, no id changes, and every call keeps its
  result, so the server's history has the same shape (haiku.rag's
  per-question ledger counts messages, and refuses a history which
  shrinks);
- the newest `--keep-recent` large results are kept whole, and so is
  any result shorter than `--min-elide-chars`;
- `cite`, `load_capability` and `search_tools` results are never
  compacted, nor is any call or result carrying an `encrypted_value`.

A compacted result keeps what identifies it:

- a haiku.rag **search** keeps each hit's header block -- its chunk id
  and rank, collection, source and type -- without its text. The model
  can still cite those chunk ids (haiku.rag resolves them from the run's
  state, not from the message), or search again for the text;
- a client **`shell`** result keeps its exit code, timeout flag and
  output sizes as JSON, and the end of stderr when the command failed:
  a failure never reads as a success;
- **`execute_code`** keeps its first and last 600 characters; any other
  tool its first and last 300.

### The compaction marker

The server stores the history each run is sent, so every client reading
the thread back -- the Flutter frontend, the thread REST API, a later
TUI session -- sees compacted results. Each one starts with a line of
the form:

```text
[compacted by soliplex-tui harness: tool=search format=headers original_bytes=10612]
```

`tool` is the tool's name, `format` one of `headers`, `shell` or
`head_tail`, and `original_bytes` the size of the full result, in UTF-8
bytes. A UI can match the prefix `[compacted by soliplex-tui harness:`
to show "result compacted" (in Python,
`soliplex.agui.harness.compacted_info` parses the line). The TUI's own
chat view is drawn from the run's events as they stream, so it keeps
showing the full output.

### Byte-stable compaction

A compacted result depends only on the tool's name and the original
result: no timestamps, counters or budget figures. Once compacted, a
result is byte-identical on every later resend, and a result already
carrying the marker is never compacted again. The tools and the rest of
the history are left as they were.
