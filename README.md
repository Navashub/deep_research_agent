# Audi Deep Research Agent - LangGraph Edition

A LangGraph/LangChain agent that takes a topic ("Audi quattro racing history"),
researches it on the web, and writes a 2000-3000 word blog post about it.
Originally built on OpenAI's Agents SDK; this version is a genuine LangGraph
rewrite so the architecture actually matches what's on the CV.

## What's actually happening when you click "Run"

```
                         ┌──────────────────┐
   user query  ───────►  │   plan_searches   │   1 LLM call, structured output
                         └─────────┬─────────┘   → WebSearchPlan (10 WebSearchItems)
                                   │
                    Send() fans out to N parallel branches
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐         ┌──────────┐         ┌──────────┐
        │  search  │   ...   │  search  │   ...   │  search  │   one branch per item,
        └────┬─────┘         └────┬─────┘         └────┬─────┘   runs concurrently
             │                    │                    │
       Tavily search +      Tavily search +      Tavily search +
       LLM summarize        LLM summarize        LLM summarize
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                          (LangGraph waits for
                           every branch to finish)
                                   │
                         ┌──────────────────┐
                         │   write_report    │   1 LLM call, structured output
                         └─────────┬─────────┘   → ReportData
                                   │
                         ┌──────────────────┐
                         │    send_push      │   optional, skipped if unconfigured
                         └─────────┬─────────┘
                                   ▼
                           markdown report
                          streamed to Gradio
```

This whole flow is defined declaratively in `graph.py` as a LangGraph
`StateGraph`. The interesting part is `continue_to_searches()` - it uses
LangGraph's `Send` API to dynamically create one parallel branch per planned
search, which is the graph-native replacement for manually managing
`asyncio.create_task` calls.

## File-by-file

| File | Role |
|---|---|
| `app.py` | Gradio UI. Streams status updates and the final report via `graph.astream(...)`. |
| `graph.py` | Defines the `StateGraph`: nodes, edges, and the parallel fan-out logic. This is the orchestration layer - the LangGraph equivalent of a manually-written controller. |
| `llm_config.py` | One function, `get_llm()`, that returns a chat model based on the `LLM_PROVIDER` env var (`groq` / `ollama` / `openai`). Every other file calls this instead of importing a specific provider's SDK, so swapping models never touches business logic. |
| `planner_agent.py` | Turns the user's query into a `WebSearchPlan` (10 `WebSearchItem`s) via structured output. |
| `search_agent.py` | For each planned item: calls Tavily, then has the LLM turn the raw search snippets into a clean 400-500 word summary. |
| `writer_agent.py` | Takes all the search summaries and writes the final `ReportData` (short summary, full markdown report, follow-up questions). |
| `push_agent.py` | Optional Pushover notification once the report's ready. No-ops quietly if `PUSHOVER_USER`/`PUSHOVER_TOKEN` aren't set. |
| `test_setup.py` | Smoke test - confirms your provider, structured output, and Tavily all work *before* you run a full (slow) end-to-end pass. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real keys
```

Pick a provider in `.env` via `LLM_PROVIDER`:

- **`groq`** - hosted Llama (`llama-3.3-70b-versatile`), fast, free tier, no GPU needed
- **`ollama`** - fully local (e.g. `llama3.2:latest`), needs `ollama serve` running and the model pulled
- **`openai`** - original fallback

You always need `TAVILY_API_KEY` regardless of provider - that's the web search step.

**Important:** `load_dotenv()` must run before anything that imports `graph.py`,
`planner_agent.py`, etc., because those modules call `get_llm()` at import
time. `app.py` is already ordered correctly - keep it that way if you touch it.

## Run

```bash
python test_setup.py   # fast sanity check: provider, structured output, Tavily
python app.py           # full Gradio app
```

---

## This is a working prototype, not a production system yet

It's solid for demoing the architecture and talking through the design in an
interview, but there's a real gap to "production." Roughly in priority order:

### Reliability
- **No retries.** A single failed Tavily call or a flaky local-model parse
  failure just silently drops that one search result (`search_one` catches
  the exception and returns `None`). Fine for a demo, not for anything
  someone depends on. Add retry logic (e.g. `tenacity`) around both the
  Tavily call and the structured-output calls.
- **No timeout handling.** A hung Ollama call or slow Tavily request has
  nothing bounding it. Add per-call timeouts.
- **Structured output on small local models is inherently flaky.**
  `llama3.2` can occasionally return malformed JSON, especially on the
  writer step (long free-text field + a list, in one schema). Worth adding
  a fallback: catch the parsing error and retry once with a simpler prompt,
  or split the writer call into two smaller structured calls.

### Observability
- **No tracing.** The original OpenAI Agents SDK version had a built-in
  trace URL; this version has none. LangSmith (`langchain` integrates with
  it natively - just set `LANGCHAIN_TRACING_V2=true` and the right env vars)
  would give you per-node latency, token usage, and full input/output
  visibility for free.
- **`print()` statements instead of real logging.** Swap for the `logging`
  module with proper levels, so you can dial verbosity up/down without
  editing code.

### Testing
- **Zero automated tests.** `test_setup.py` is a manual smoke test, not a
  test suite. Worth adding actual unit tests for each node function with
  mocked LLM/Tavily responses, plus at least one integration test that runs
  the full graph against a cheap/fast model.

### Architecture / scalability
- **Everything runs synchronously inside one Gradio process.** Fine for one
  person testing locally; won't hold up under concurrent users. A real
  deployment would put this behind a proper async API (FastAPI) with a task
  queue, and have the Gradio/web frontend just poll or subscribe for status.
- **No caching.** Re-running the same query re-does all 10 searches and all
  LLM calls from scratch. Even a simple cache keyed on the search query
  would cut cost and latency a lot for repeated topics.
- **No persistence/checkpointing.** If the process crashes mid-run, the
  entire state is lost. LangGraph supports a `checkpointer` (e.g.
  `MemorySaver`, or a SQLite/Postgres-backed one) that would let you resume
  a run instead of starting over - also opens the door to a "show me
  progress so far" UI.
- **Local model concurrency.** If you deploy with `LLM_PROVIDER=ollama`,
  note that a single local Ollama server typically can't usefully serve many
  concurrent requests - the 10 parallel "search" branches in this graph will
  effectively queue up behind each other on one GPU/CPU. This isn't a problem
  with Groq (or OpenAI), since those are properly multi-tenant hosted
  services. Worth being explicit about this trade-off if asked in an
  interview: local = private/free but serializes under load; hosted = scales
  but costs money and leaves your machine.

### Security / config hygiene
- **No secrets management.** `.env` is fine for local dev, but a production
  deploy should pull keys from a proper secrets manager (AWS Secrets Manager,
  GCP Secret Manager, Vault, etc.), not a flat file.
- **No input validation/guardrails on the user's query.** Nothing stops
  someone from typing something wildly off-topic or adversarial and burning
  10 searches + 2 LLM calls on it. A cheap guard (keyword check or a fast
  classifier) before `plan_searches` would help.
- **No rate limiting.** Nothing stops one user from spamming the Gradio
  button and running up your Tavily/Groq bill.

### Product polish
- **Streaming is step-level, not token-level.** Right now `app.py` only
  yields a new message when an entire node finishes (e.g. "writing
  report..." then the whole report appears at once). True token-by-token
  streaming of the final report would feel much more responsive - LangGraph
  supports `stream_mode="messages"` for this.
- **No way to see the intermediate search results**, only the final report.
  Even just logging/showing the 10 summaries would make debugging bad
  reports much easier.

None of this needs to happen at once - the architecture (clean node
separation, provider abstraction, parallel fan-out) is already structured in
a way that makes most of these additive rather than requiring a rewrite.