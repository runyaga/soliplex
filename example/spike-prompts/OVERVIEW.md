# Monty Spike: Small LLMs Generating Complex Python

We validated that small open-source LLMs (20B and 120B parameters) can
reliably generate multi-step Python data pipelines in a sandboxed
interpreter -- with the right system prompt.

## What is this?

Monty is our sandboxed Python interpreter (no imports, no file I/O, no
network). An LLM receives a natural language task, generates Python code,
and Monty executes it. If execution fails, the error is fed back and the
LLM retries (up to 10 attempts).

The question: **how small can the LLM be and still produce working code?**

## Results

We ran 71 trials across 9 experiment types on two model sizes.

| Model | Trials | Pass Rate |
|-------|--------|-----------|
| gpt-oss:20b | 43 | **84%** |
| gpt-oss:120b | 28 | **93%** |

Both models run locally on a single machine via Ollama.

### By experiment type

| Experiment | What it tests | 20B | 120B |
|-----------|--------------|-----|------|
| Baseline 6-step | create/filter/group/sort/chart | 90% | 100% |
| Parameter extraction | NL to structured args | 100% | 100% |
| Multi-source merge | two datasets, join, compute | 83% | 100% |
| Ambiguous intent | vague 3-word prompts | 100% | 50% |
| Math formulas | std dev, variance by group | 100% | 100% |
| String processing | word frequency, top-N | 67% | 100% |
| Schema mismatch | mismatched keys, rename+merge | 40% | 100% |
| Conditional logic | Fibonacci, grade calculator | 100% | 100% |
| 10+ step pipeline | full ETL in one shot | 60% | 100% |

## Example: 10-Step Pipeline in One Shot

**Prompt sent to the 120B model:**

> Build a complete data pipeline: 1) Create 15 products with product_name,
> category (Tech/Home/Food), base_price, and units_sold. 2) Add a revenue
> column (base_price * units_sold). 3) Add a margin column (revenue * 0.3
> for Tech, 0.2 for Home, 0.15 for Food). 4) Filter to revenue > 1000.
> 5) Drop the base_price column. 6) Rename margin to gross_profit.
> 7) Group by category summing revenue and gross_profit. 8) Sort by
> gross_profit descending. 9) Show top 3 categories. 10) Print a final
> report with the results.

**Result:** The model generates working Python on the first attempt.
It creates the data, chains all 10 transformations using `df_*` host
functions, and prints:

```json
[
  {"category": "Tech", "gross_profit": 12870.0, "revenue": 42900},
  {"category": "Home", "gross_profit": 3394.0, "revenue": 16970},
  {"category": "Food", "gross_profit": 531.0, "revenue": 3540}
]
```

The 120B model passed this experiment 3/3 times, clean first try every time.
The 20B model passed 3/5 (60%), needing 2-5 retries on successful runs.

## What made it work: the system prompt

The single highest-impact change was **enumerating every available function
with parameter types** in the system prompt. Without this:

- The 20B model ignored the API and wrote pure Python (drifting on key names)
- The 120B model hallucinated Pandas-like functions that don't exist

| Metric | Before enumeration | After enumeration |
|--------|-------------------|-------------------|
| 20B schema mismatch | 40% | 100% |
| 120B schema mismatch | 33% | 100% |

The key insight: **larger models need MORE documentation, not less.**
The 120B model's stronger priors cause it to hallucinate missing API
surface when documentation is incomplete.

### Other prompt elements that matter

1. **"NOT Supported" section** listing Monty limitations (no imports, no
   tuples, no dict unpacking) -- prevents first-attempt errors
2. **"Pure Python is fine too"** -- gives the model permission to fall back
   to manual Python if the API doesn't cover its needs
3. **"Do NOT invent functions"** -- prevents the 120B from hallucinating API

## Links

- **Evaluation prompts:** [evaluation-prompts.md](evaluation-prompts.md) --
  exact text for all 9 experiments
- **System prompt:** [../rooms/spike-120b/prompt.txt](../rooms/spike-120b/prompt.txt) --
  the evolved prompt that drives both models
- **Full results:** PR comment on
  [runyaga/flutter#66](https://github.com/runyaga/flutter/pull/66)
- **Schema mismatch deep-dive:** Issue
  [soliplex/soliplex#671](https://github.com/soliplex/soliplex/issues/671)
- **Branch:** `feat/soliplex-cli-monty` on
  [runyaga/soliplex](https://github.com/runyaga/soliplex/tree/feat/soliplex-cli-monty)

## How to reproduce

```bash
# 1. Start the backend (from soliplex-cli-monty repo)
OLLAMA_BASE_URL=http://bizon:11435 \
  .venv/bin/soliplex-cli serve example/minimal.yaml --no-auth-mode

# 2. Run an experiment (from soliplex-flutter-spike-monty repo)
DART_MONTY_LIB_PATH=.../libdart_monty_native.dylib \
  dart run packages/soliplex_cli/bin/soliplex_cli.dart \
  --host http://localhost:8000 \
  --room spike-120b \
  --monty \
  --prompt "Create a dataset of 10 products..."
```

## 20B vs 120B: when to use which

- **120B** for structured tasks with clear specs -- higher first-try success,
  uses the API idiomatically, handles complex multi-step pipelines
- **20B** for exploratory/ambiguous tasks -- more willing to invent sample
  data and just run with it (120B asks for clarification)
- **Both** benefit equally from good system prompts -- prompt quality matters
  more than 6x model size difference
