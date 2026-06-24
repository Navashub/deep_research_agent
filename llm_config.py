import os

# Which provider to use for every chat call in this project.
# Set LLM_PROVIDER in your .env to one of: "groq", "ollama", "openai"
#
#   groq   -> hosted Llama models, runs in the cloud, no local GPU needed, very fast,
#             free tier available. Good default if you don't have hardware for local inference.
#   ollama -> truly local, runs on your own machine via https://ollama.com
#             (`ollama pull llama3.1` first)
#   openai -> the original provider, kept as a fallback

DEFAULT_PROVIDER = "groq"


def get_llm(temperature: float = 0.5):
    """Returns a LangChain chat model based on LLM_PROVIDER. All three support
    .with_structured_output(), which is all this project needs."""
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'groq', 'ollama', or 'openai'.")