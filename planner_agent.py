from typing import List
from pydantic import BaseModel, Field
from llm_config import get_llm

HOW_MANY_SEARCHES = 10

INSTRUCTIONS = (
    "You are an automotive research specialist focusing on Audi cars and automotive history. "
    f"Given an Audi-related query, create {HOW_MANY_SEARCHES} strategic web searches that will help create a "
    "comprehensive blog post about Audi. Focus on: brand history, iconic models, technological innovations, "
    "design evolution, racing heritage, company milestones, founders/key figures, manufacturing, and cultural "
    "impact. Prioritize searches that will provide engaging stories and fascinating details for blog readers."
)


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: List[WebSearchItem] = Field(
        description="A list of web searches to perform to best answer the query."
    )


_llm = get_llm(temperature=0.3)
planner_chain = _llm.with_structured_output(WebSearchPlan)


def plan_searches_node(state: dict) -> dict:
    """LangGraph node: turn the user's query into a structured WebSearchPlan."""
    query = state["query"]
    print("Planning searches...")
    plan: WebSearchPlan = planner_chain.invoke(
        [
            ("system", INSTRUCTIONS),
            ("human", f"Query: {query}"),
        ]
    )
    print(f"Will perform {len(plan.searches)} searches")
    return {"search_plan": plan.searches}