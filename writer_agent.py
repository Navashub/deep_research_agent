from typing import List
from pydantic import BaseModel, Field
from llm_config import get_llm

INSTRUCTIONS = (
    "You are a professional automotive blogger specializing in luxury car brands, particularly Audi. "
    "You will be provided with research findings about Audi and should create an engaging, well-structured "
    "blog post. Write in a compelling narrative style that combines historical storytelling with technical "
    "insights. Structure your blog post with: an engaging introduction, multiple themed sections with "
    "descriptive headings, fascinating anecdotes and human stories, technical details explained accessibly, "
    "and a memorable conclusion. Use markdown formatting with proper headers, and aim for 2000-3000 words "
    "that will captivate car enthusiasts and general readers alike. Include specific dates, model names, and "
    "interesting facts throughout."
)


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: List[str] = Field(description="Suggested topics to research further")


_llm = get_llm(temperature=0.7)
writer_chain = _llm.with_structured_output(ReportData)


def write_report_node(state: dict) -> dict:
    """LangGraph node: turn all the gathered search summaries into the final report."""
    print("Thinking about report...")
    query = state["query"]
    search_results = state["search_results"]
    report: ReportData = writer_chain.invoke(
        [
            ("system", INSTRUCTIONS),
            ("human", f"Original query: {query}\nSummarized search results: {search_results}"),
        ]
    )
    print("Finished writing report")
    return {"report": report}