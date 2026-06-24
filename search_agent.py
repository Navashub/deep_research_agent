from langchain_tavily import TavilySearch
from llm_config import get_llm

INSTRUCTIONS = (
    "You are an automotive research specialist focusing on Audi. Given a search term and raw web search "
    "results, produce a comprehensive summary of the results. The summary should be 3-4 paragraphs and "
    "400-500 words. Focus on capturing interesting stories, historical details, technical innovations, and "
    "engaging facts about Audi that would make for compelling blog content. Include specific dates, model "
    "names, technical specifications, and human interest elements. Write in a clear, informative style that "
    "balances technical accuracy with accessibility for general readers. This will be used to create an "
    "engaging blog post about Audi."
)

# max_results mirrors the "low" search context size from the original OpenAI WebSearchTool
_tavily_tool = TavilySearch(max_results=5)
_summarizer = get_llm(temperature=0.5)


def search_one(item) -> str | None:
    """Run a Tavily search for one planned item, then have an LLM turn the raw
    snippets into the same kind of polished summary the original WebSearchTool produced."""
    try:
        raw = _tavily_tool.invoke({"query": item.query})
        raw_text = "\n\n".join(
            f"Source: {r.get('url', '')}\n{r.get('content', '')}" for r in raw.get("results", [])
        )
        response = _summarizer.invoke(
            [
                ("system", INSTRUCTIONS),
                (
                    "human",
                    f"Search term: {item.query}\nReason for searching: {item.reason}\n\n"
                    f"Raw search results:\n{raw_text}",
                ),
            ]
        )
        return response.content
    except Exception as e:
        print(f"Search failed for '{item.query}': {e}")
        return None


def search_node(payload: dict) -> dict:
    """LangGraph node body for a single fanned-out search (see graph.py's Send calls)."""
    item = payload["item"]
    result = search_one(item)
    return {"search_results": [result] if result else []}