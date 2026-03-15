import os
from langchain_openai import ChatOpenAI
from ingest import load_vectorstore
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:120b-cloud")
API_KEY = os.getenv("OLLAMA_API_KEY")
TOP_K = int(os.getenv("TOP_K_CHUNKS", "4"))

MULTI_QUERY_PROMPT = """
Generate exactly 3 different search queries to find 
comprehensive information about: {question}

Each query should approach the topic differently.
Return exactly 3 queries, one per line.
No numbering. No explanation. Just the queries.
"""

DECOMPOSE_PROMPT = """
Break this question into exactly 3 simpler sub-questions
that together answer the original question: {question}

Return exactly 3 sub-questions, one per line.
No numbering. No explanation. Just the questions.
"""


def retrieve_chunks(
    question: str,
    doc_name: str,
    strategy: str
) -> list:
    """
    Retrieves relevant chunks from FAISS based on strategy.
    SIMPLE: one search
    MULTI: three different query searches
    DECOMPOSE: break into sub-questions then search each
    """
    vectorstore = load_vectorstore(doc_name)
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_base=f"{OLLAMA_BASE_URL}/v1",
        openai_api_key=API_KEY,
        temperature=0
    )

    all_chunks = []
    seen_contents = set()

    def search_and_collect(query: str):
        results = vectorstore.similarity_search(query, k=TOP_K)
        for doc in results:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                all_chunks.append(doc)

    if strategy == "SIMPLE":
        search_and_collect(question)

    elif strategy == "MULTI":
        msg = llm.invoke(
            MULTI_QUERY_PROMPT.format(question=question)
        )
        queries_text = msg.content
        queries = [
            q.strip()
            for q in queries_text.strip().split('\n')
            if q.strip()
        ][:3]

        for query in queries:
            search_and_collect(query)

    elif strategy == "DECOMPOSE":
        msg = llm.invoke(
            DECOMPOSE_PROMPT.format(question=question)
        )
        sub_questions_text = msg.content
        sub_questions = [
            q.strip()
            for q in sub_questions_text.strip().split('\n')
            if q.strip()
        ][:3]

        for sub_q in sub_questions:
            search_and_collect(sub_q)

    return all_chunks
