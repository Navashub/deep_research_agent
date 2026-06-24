"""
Quick sanity checks before running the full app — catches config issues in
seconds instead of discovering them after a 2-minute full research run.

Usage: python test_setup.py
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def check_env_vars():
    print("1. Checking environment variables...")
    provider = os.environ.get("LLM_PROVIDER", "groq")
    print(f"   LLM_PROVIDER = {provider}")
    if provider == "ollama":
        print(f"   OLLAMA_MODEL = {os.environ.get('OLLAMA_MODEL')}")
        print(f"   OLLAMA_BASE_URL = {os.environ.get('OLLAMA_BASE_URL')}")
    elif provider == "groq":
        ok = bool(os.environ.get("GROQ_API_KEY"))
        print(f"   GROQ_API_KEY set: {ok}")
    elif provider == "openai":
        ok = bool(os.environ.get("OPENAI_API_KEY"))
        print(f"   OPENAI_API_KEY set: {ok}")
    tavily_ok = bool(os.environ.get("TAVILY_API_KEY"))
    print(f"   TAVILY_API_KEY set: {tavily_ok}")
    if not tavily_ok:
        print("   WARNING: TAVILY_API_KEY is required regardless of LLM_PROVIDER.")
    print()


def check_plain_call():
    print("2. Testing a plain (non-structured) call to the LLM...")
    from llm_config import get_llm

    llm = get_llm(temperature=0.3)
    response = llm.invoke("Reply with exactly the words: connection successful")
    print(f"   Response: {response.content!r}")
    print()


def check_structured_output():
    print("3. Testing structured output (this is what trips up some local models)...")
    from pydantic import BaseModel, Field
    from llm_config import get_llm

    class Fact(BaseModel):
        animal: str = Field(description="name of an animal")
        fact: str = Field(description="one interesting fact about it")

    llm = get_llm(temperature=0.3)
    structured = llm.with_structured_output(Fact)
    result = structured.invoke("Tell me one interesting fact about an octopus.")
    print(f"   Parsed result: {result}")
    print()


def check_tavily():
    print("4. Testing Tavily web search...")
    from langchain_tavily import TavilySearch

    tool = TavilySearch(max_results=2)
    result = tool.invoke({"query": "Audi quattro history"})
    titles = [r.get("title") for r in result.get("results", [])]
    print(f"   Got {len(titles)} results: {titles}")
    print()


if __name__ == "__main__":
    check_env_vars()
    try:
        check_plain_call()
        check_structured_output()
        check_tavily()
        print("All checks passed. You're good to run: python app.py")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("\nCommon fixes:")
        print("- ollama provider: make sure 'ollama serve' is running and you've run "
              "'ollama pull llama3.2'")
        print("- groq/openai provider: double check the API key in .env")
        print("- structured output failing on Ollama: try a newer/larger model, "
              "llama3.2 should work but very small/quantized variants sometimes don't")