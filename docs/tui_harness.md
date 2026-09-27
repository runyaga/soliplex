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

`--compaction off|auto|always` (default: `auto`;
`SOLIPLEX_TUI_COMPACTION`), `--keep-recent N` (default: 4;
`SOLIPLEX_TUI_KEEP_RECENT`), `--min-elide-chars N` (default: 1024;
`SOLIPLEX_TUI_MIN_ELIDE_CHARS`).

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

- a haiku.rag **search** keeps each hit's whole header block -- its
  chunk id and rank, then collection, source, type, figure captions --
  without its text. The model can still cite those chunk ids (haiku.rag
  resolves them from the run's state, not from the message), or search
  again for the text;
- a client **`shell`** result keeps its exit code, timeout flag and
  output sizes as JSON, and the end of stderr when the command failed:
  a failure never reads as a success;
- **`execute_code`** keeps its first and last 600 characters; any other
  tool its first and last 300.

A `search` or `shell` result which is not of the shape expected (hits
the client cannot parse, a result not made by this client) is left as
it is, rather than risk losing its chunk ids or a failure.

### When: `auto`, `always` and `off`

- `auto` (the default) compacts nothing until the next request is
  estimated (see [Context budget](#context-budget)) to take more than
  `--compaction-trigger` (default 0.70) of the usable context window.
  Then it compacts, oldest first, just enough results to bring the
  estimate under `--compaction-target` (default 0.40), **in one batch**,
  and nothing more until the trigger is crossed again. With no window
  known, `auto` compacts nothing.
- `always` compacts every eligible result but the newest
  `--keep-recent`, before every POST.
- `off` sends the history as it is.

Why batches: a model server with prefix caching (vLLM's is on by
default) reuses the computed prefix of a prompt it has seen. An
append-only history keeps that prefix; rewriting an old result
invalidates everything after it. Compacting a little on every turn
(`always`) would bust the cache on every turn; `auto`'s high- and
low-water marks bust it once per batch, and between batches the history
only grows.

### The compaction marker

The server stores the history each run is sent, so every client reading
the thread back -- the Flutter frontend, the thread REST API, a later
TUI session -- sees compacted results. Each one starts with a line of
the form:

```text
[compacted by soliplex-tui harness: tool=search format=headers original_bytes=10612]
```

`tool` is the tool's name, percent-encoded (so a name holding a space,
`]` or `=` cannot break the line), `format` one of `headers`, `shell` or
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

## Context budget

`--context-window N` (`SOLIPLEX_TUI_CONTEXT_WINDOW`),
`--probe-model-window` (`SOLIPLEX_TUI_PROBE_MODEL_WINDOW`),
`--output-reserve N` (default: 4096; `SOLIPLEX_TUI_OUTPUT_RESERVE`),
`--compaction-trigger F` / `--compaction-target F` (defaults: 0.70 /
0.40; `SOLIPLEX_TUI_COMPACTION_TRIGGER` / `_TARGET`).

The **window** comes from, in order:

1. `--context-window`;
2. the room's `agent.context_window` (declare it in the room's YAML for
   a local model: the server cannot know a vLLM model's window
   otherwise);
3. with `--probe-model-window` only, the room's model server:
   `GET {provider_base_url}/v1/models`, the entry whose `id` is the
   room's model, its `max_model_len` (vLLM). This contacts the model
   server directly from the client, so it is off by default;
4. otherwise it is unknown, and `auto` compacts nothing. There is no
   default: a guess (pydantic-ai-harness uses 200k) is badly wrong for
   a 98k local model.

`--output-reserve` tokens of the window are kept for the reply; the
fractions apply to the rest.

The **estimate** of the next request is anchored on what the server
measured: after each run the client reads the run's usage
(`GET .../agui/{thread_id}/{run_id}/usage`) and takes
`final_input_tokens + final_output_tokens` -- the last request, plus the
reply it made, which the next request carries (never the cumulative
`input_tokens`). The next request is estimated as that, plus the
characters added to the history since (less those compaction saved) at
3.5 characters a token. Before the first measurement it is the history's
size at 3.5 characters a token, which leaves out the system prompt and
tool definitions. A run with no usage (e.g., it failed) keeps the last
anchor, marked stale.

With `ask --url --json`, the output's `resends` lists what the client did
before each POST: the history's size in characters (`resend_chars`,
`state_chars`), how many results it compacted (`compacted`, saving
`compacted_chars`; `compacted_total` in all), the estimate
(`est_tokens`) and the window (`window_tokens`, `window_source`).
