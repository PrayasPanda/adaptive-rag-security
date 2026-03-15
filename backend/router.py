import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY")

ROUTER_PROMPT = """
You are a routing expert for a cybersecurity RAG system.

Given this question decide the best retrieval strategy.

SIMPLE   → straightforward single fact question
           Example: "What is CVE-2024-1234?"

MULTI    → complex question needing multiple searches
           Example: "What are all critical vulnerabilities 
                     and their fixes?"

DECOMPOSE → comparison or multi-part question
            Example: "Compare Apache and OpenSSL vulnerabilities"

Question: {question}

Reply with exactly one word: SIMPLE, MULTI, or DECOMPOSE
"""


def route_question(question: str) -> str:
    """
    Routes the question to the appropriate retrieval strategy.
    Returns: "SIMPLE", "MULTI", or "DECOMPOSE"
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    result_msg = llm.invoke(
        ROUTER_PROMPT.format(question=question)
    )
    result = result_msg.content.strip().upper()

    # Validate response
    valid_strategies = ["SIMPLE", "MULTI", "DECOMPOSE"]
    for strategy in valid_strategies:
        if strategy in result:
            return strategy

    # Default to SIMPLE if unclear
    return "SIMPLE"
