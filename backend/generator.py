import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY")

GENERATOR_PROMPT = """
You are a senior cybersecurity expert assistant.

Answer the question using ONLY the context provided below.
For every claim you make cite the page number like (Page X).
If the context does not contain the answer say:
"The uploaded document does not contain information about this."
Never make up CVE IDs, severity scores, or technical details.

Context:
{context}

Question: {question}

Detailed answer with page citations:
"""


def generate_answer(question: str, chunks: list) -> str:
    """
    Generates a cited answer from relevant chunks.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    # Build context from chunks with page numbers
    context = ""
    for chunk in chunks:
        page = chunk.metadata.get("page", "?")
        context += f"\n[Page {page}]:\n{chunk.page_content}\n"

    msg = llm.invoke(
        GENERATOR_PROMPT.format(
            context=context,
            question=question
        )
    )
    answer = msg.content

    return answer


def extract_sources(chunks: list) -> list:
    """
    Extracts page numbers and content previews from chunks.
    """
    sources = []
    seen_pages = set()

    for chunk in chunks:
        page = chunk.metadata.get("page", 0)
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append({
                "page": page + 1,  # Convert to 1-indexed
                "content_preview": chunk.page_content[:150] + "..."
            })

    return sorted(sources, key=lambda x: x["page"])
