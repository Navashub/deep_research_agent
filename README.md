# Audi Deep Research Agent — LangGraph version

This is a LangGraph/LangChain rewrite of your original OpenAI Agents SDK project.
Same end result (Gradio app that researches Audi and writes a long blog post),
different framework under the hood.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your real keys
```

### Choosing your LLM provider

Set `LLM_PROVIDER` in `.env` to one of:

- **`groq`** (default) — hosted Llama models (e.g. `llama-3.3-70b-versatile`), runs in
  the cloud but is extremely fast and has a free tier. This is the easiest way to use
  Llama without needing a GPU. Get a key at https://console.groq.com/keys
- **`ollama`** — truly local. Install https://ollama.com, run `ollama pull llama3.1`,
  then point `OLLAMA_BASE_URL` at your local server (default `http://localhost:11434`).
  No API key needed, but you need decent hardware and it'll be slower than Groq.
- **`openai`** — the original provider, kept as a fallback.

This switch lives in `llm_config.py` and is used by `planner_agent.py`,
`search_agent.py`, and `writer_agent.py` — none of them import a specific
provider's SDK directly anymore, they just call `get_llm()`.

One thing to flag: structured output (`with_structured_output`) relies on tool
calling, which not every local model handles well. `llama3.1` and newer (and most
Groq-hosted Llama models) support it fine. If you try an older/smaller Ollama model
and get parsing errors, that's almost always why.

You also always need:
- `TAVILY_API_KEY` — free key at https://tavily.com, used for web search regardless of LLM provider
- `PUSHOVER_USER` / `PUSHOVER_TOKEN` — optional, only if you want the push
  notification step to actually send something (it's skipped quietly if unset)

## Run

```bash
python app.py
```

Opens a Gradio app in your browser, same as the original.

## How it maps to the original project

| Original (OpenAI Agents SDK)              | This version (LangGraph)                          |
|--------------------------------------------|-----------------------------------------------------|
| `planner_agent.py` → `Agent` with `output_type` | `planner_agent.py` → `ChatOpenAI.with_structured_output()` |
| `search_agent.py` → `Agent` + hosted `WebSearchTool` | `search_agent.py` → `TavilySearch` tool + LLM summarizer |
| `writer_agent.py` → `Agent` with `output_type` | `writer_agent.py` → `ChatOpenAI.with_structured_output()` |
| `push_agent.py` → `Agent` + `@function_tool` | `push_agent.py` → `@tool` from `langchain_core.tools` |
| `research_manager.py` → manual `asyncio` orchestration | `graph.py` → a `StateGraph` with `Send` for fan-out |
| `deep_research.py` → Gradio UI | `app.py` → Gradio UI, streaming via `graph.astream(...)` |

## The key architectural difference

The original manually wires async tasks together in `research_manager.py`
(`asyncio.create_task` + `asyncio.as_completed`). LangGraph replaces that with
an explicit graph:

```
START -> plan_searches -> [search, search, search, ...] (parallel, fanned out via Send) -> write_report -> send_push -> END
```

`continue_to_searches()` in `graph.py` is the part doing the fan-out — it
returns one `Send("search", ...)` per planned query, and LangGraph runs them
concurrently, waiting for all of them before moving on to `write_report`.
That's the direct equivalent of the original's parallel search loop.

## Why Tavily instead of OpenAI's WebSearchTool

The original used OpenAI's *hosted* search tool, which only exists inside
the OpenAI Agents SDK / Responses API — LangChain has no equivalent for it.
Tavily is the standard search tool in the LangGraph ecosystem and is what
most LangGraph tutorials and docs use, so it's a reasonable thing to have on
your CV alongside "built an agent with LangGraph."

## What's identical to the original

- All the prompt instructions (planner, search summarizer, writer) — copied verbatim
- The `WebSearchPlan` / `WebSearchItem` / `ReportData` schemas
- The Pushover push notification logic
- The Gradio UI layout and behavior

## Things worth exploring next, if you want to go deeper for interview purposes

- Swap `with_structured_output` for explicit tool-calling to talk about LangChain's tool-binding patterns
- Add a `checkpointer` (e.g. `MemorySaver`) to demonstrate LangGraph's persistence/resumability
- Add a conditional edge that retries a failed search instead of just dropping it
- Visualize the graph with `pip install grandalf` then `research_graph.get_graph().draw_ascii()`