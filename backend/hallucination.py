import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY")

HALLUCINATION_PROMPT = """
You are a fact checker for a cybersecurity RAG system.

Context (source of truth):
{context}

Generated Answer:
{answer}

Is every factual claim in the answer supported by 
the context above?
Check especially: CVE IDs, severity levels, CVSS scores,
affected systems, version numbers.

If the generated answer states that the document does not contain the information, reply with exactly one word: GROUNDED

Otherwise, reply with exactly one word: GROUNDED or HALLUCINATING
"""


def check_hallucination(answer: str, chunks: list) -> bool:
    """
    Checks if the answer is grounded in the retrieved chunks.
    Returns True if grounded, False if hallucinating.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    context = "\n".join([chunk.page_content for chunk in chunks])

    msg = llm.invoke(
        HALLUCINATION_PROMPT.format(
            context=context,
            answer=answer
        )
    )
    result = msg.content.strip().upper()

    return "GROUNDED" in result
