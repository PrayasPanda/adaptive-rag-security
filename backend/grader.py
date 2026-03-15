import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY")

GRADER_PROMPT = """
You are a relevance grader for a cybersecurity RAG system.

Question: {question}

Retrieved chunk:
{chunk}

Is this chunk relevant and useful to answer the question?
Reply with exactly one word: YES or NO
"""

REWRITE_PROMPT = """
The original search query did not find relevant results.
Rewrite this query to be more specific and likely to 
find relevant cybersecurity information.

Original query: {question}

Rewritten query (one line only):
"""


def grade_chunks(question: str, chunks: list) -> list:
    """
    Grades each chunk for relevance to the question.
    Returns only relevant chunks.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    relevant_chunks = []

    for chunk in chunks:
        msg = llm.invoke(
            GRADER_PROMPT.format(
                question=question,
                chunk=chunk.page_content
            )
        )
        result = msg.content.strip().upper()

        if "YES" in result:
            relevant_chunks.append(chunk)

    return relevant_chunks


def rewrite_query(question: str) -> str:
    """
    Rewrites the query when no relevant chunks found.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    msg = llm.invoke(
        REWRITE_PROMPT.format(question=question)
    )
    rewritten = msg.content.strip()

    return rewritten
