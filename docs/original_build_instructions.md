# 🤖 AGENT BUILD INSTRUCTIONS
# Adaptive RAG Security Assistant
# LangGraph + FAISS + Ollama + FastAPI + Next.js
# 100% Local — GDPR Compliant

---

## AGENT IDENTITY
You are an expert Python and Next.js developer.
Build this project exactly as described step by step.
Do not skip any step.
Do not add any libraries not listed.
Do not ask questions. Just build.
If something fails, fix it and continue.

---

## PROJECT SUMMARY
Build a GDPR-compliant adaptive RAG pipeline for
cybersecurity document analysis.

The system has 4 LLM nodes managed by LangGraph:
1. Router — decides retrieval strategy
2. Grader — filters irrelevant chunks
3. Generator — produces cited answer
4. Hallucination Checker — verifies answer

Everything runs locally via Ollama.
No cloud. No vector cloud. No external APIs.
FAISS handles all vector storage locally.

---

## PREREQUISITES
Before starting make sure these are installed:
- Python 3.11
- Node.js 18+
- Ollama running at http://localhost:11434
- These Ollama models pulled:
  ollama pull llama3.2
  ollama pull nomic-embed-text

---

## COMPLETE FOLDER STRUCTURE TO CREATE

```
adaptive-rag-security/
│
├── backend/
│   ├── main.py
│   ├── ingest.py
│   ├── router.py
│   ├── retriever.py
│   ├── grader.py
│   ├── generator.py
│   ├── hallucination.py
│   ├── graph.py
│   └── models.py
│
├── indexes/
├── data/
│   └── docs/
│
├── frontend/
│   ├── pages/
│   │   ├── index.js
│   │   ├── chat.js
│   │   └── health.js
│   ├── components/
│   │   ├── UploadBox.js
│   │   ├── ChatBox.js
│   │   ├── StrategyBadge.js
│   │   └── SourceCard.js
│   └── styles/
│       └── globals.css
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## STEP 1 — CREATE requirements.txt

```
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.6.0
python-multipart==0.0.9
python-dotenv==1.0.1
langchain==0.2.0
langchain-community==0.2.0
langchain-ollama==0.1.0
langgraph==0.1.0
faiss-cpu==1.8.0
pypdf==4.2.0
httpx==0.27.0
```

---

## STEP 2 — CREATE .env.example

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
MAX_RETRIES=2
TOP_K_CHUNKS=4
```

---

## STEP 3 — CREATE .gitignore

```
.env
__pycache__/
*.pyc
venv/
node_modules/
.next/
indexes/
data/docs/
*.pdf
*.txt
```

---

## STEP 4 — CREATE backend/models.py

Define all Pydantic models used across the project.

```python
from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    message: str
    doc_name: str
    chunks_indexed: int
    gdpr_compliant: bool = True


class QueryRequest(BaseModel):
    question: str
    doc_name: str


class SourceItem(BaseModel):
    page: int
    content_preview: str


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    iterations: int
    is_grounded: bool
    sources: list[SourceItem]
    gdpr_compliant: bool = True


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    gdpr_compliant: bool = True
    local_inference: bool = True
```

---

## STEP 5 — CREATE backend/ingest.py

This file handles PDF ingestion into FAISS.

```python
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def ingest_pdf(pdf_path: str, doc_name: str) -> int:
    """
    Load PDF, split into semantic chunks, embed with Ollama,
    store in FAISS locally.
    Returns number of chunks indexed.
    """
    os.makedirs("indexes", exist_ok=True)

    print(f"[Ingest] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"[Ingest] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    print(f"[Ingest] Creating embeddings for {len(chunks)} chunks...")
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_path = f"indexes/{doc_name}_faiss"
    vectorstore.save_local(index_path)

    print(f"[Ingest] Saved FAISS index to {index_path}")
    return len(chunks)


def load_vectorstore(doc_name: str) -> FAISS:
    """Load existing FAISS index for a document."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    index_path = f"indexes/{doc_name}_faiss"
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
```

---

## STEP 6 — CREATE backend/router.py

This is Node 1 of the pipeline.
It decides the retrieval strategy for the question.

```python
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

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
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    result = llm.invoke(
        ROUTER_PROMPT.format(question=question)
    ).strip().upper()

    # Validate response
    valid_strategies = ["SIMPLE", "MULTI", "DECOMPOSE"]
    for strategy in valid_strategies:
        if strategy in result:
            return strategy

    # Default to SIMPLE if unclear
    return "SIMPLE"
```

---

## STEP 7 — CREATE backend/retriever.py

This is Node 2 of the pipeline.
It retrieves relevant chunks based on the routing strategy.

```python
import os
from langchain_ollama import OllamaLLM
from ingest import load_vectorstore
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
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
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
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
        queries_text = llm.invoke(
            MULTI_QUERY_PROMPT.format(question=question)
        )
        queries = [
            q.strip()
            for q in queries_text.strip().split('\n')
            if q.strip()
        ][:3]

        for query in queries:
            search_and_collect(query)

    elif strategy == "DECOMPOSE":
        sub_questions_text = llm.invoke(
            DECOMPOSE_PROMPT.format(question=question)
        )
        sub_questions = [
            q.strip()
            for q in sub_questions_text.strip().split('\n')
            if q.strip()
        ][:3]

        for sub_q in sub_questions:
            search_and_collect(sub_q)

    return all_chunks
```

---

## STEP 8 — CREATE backend/grader.py

This is Node 3 of the pipeline.
It filters out irrelevant chunks.

```python
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

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
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    relevant_chunks = []

    for chunk in chunks:
        result = llm.invoke(
            GRADER_PROMPT.format(
                question=question,
                chunk=chunk.page_content
            )
        ).strip().upper()

        if "YES" in result:
            relevant_chunks.append(chunk)

    return relevant_chunks


def rewrite_query(question: str) -> str:
    """
    Rewrites the query when no relevant chunks found.
    """
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    rewritten = llm.invoke(
        REWRITE_PROMPT.format(question=question)
    ).strip()

    return rewritten
```

---

## STEP 9 — CREATE backend/generator.py

This is Node 4 of the pipeline.
It generates the final answer from relevant chunks.

```python
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

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
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Build context from chunks with page numbers
    context = ""
    for chunk in chunks:
        page = chunk.metadata.get("page", "?")
        context += f"\n[Page {page}]:\n{chunk.page_content}\n"

    answer = llm.invoke(
        GENERATOR_PROMPT.format(
            context=context,
            question=question
        )
    )

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
```

---

## STEP 10 — CREATE backend/hallucination.py

This is Node 5 of the pipeline.
It verifies the answer is grounded in the retrieved chunks.

```python
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

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

Reply with exactly one word: GROUNDED or HALLUCINATING
"""


def check_hallucination(answer: str, chunks: list) -> bool:
    """
    Checks if the answer is grounded in the retrieved chunks.
    Returns True if grounded, False if hallucinating.
    """
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    context = "\n".join([chunk.page_content for chunk in chunks])

    result = llm.invoke(
        HALLUCINATION_PROMPT.format(
            context=context,
            answer=answer
        )
    ).strip().upper()

    return "GROUNDED" in result
```

---

## STEP 11 — CREATE backend/graph.py

This file connects all nodes using LangGraph.
This is the brain of the entire pipeline.

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from router import route_question
from retriever import retrieve_chunks
from grader import grade_chunks, rewrite_query
from generator import generate_answer, extract_sources
from hallucination import check_hallucination
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


# Define the state that flows through the graph
class RAGState(TypedDict):
    question: str
    doc_name: str
    strategy: str
    chunks: list
    relevant_chunks: list
    answer: str
    is_grounded: bool
    sources: list
    iterations: int
    retrieval_retries: int
    generation_retries: int


# ── NODE FUNCTIONS ──────────────────────────────────────

def router_node(state: RAGState) -> RAGState:
    """Node 1: Decide retrieval strategy"""
    print(f"[Router] Question: {state['question']}")
    strategy = route_question(state["question"])
    print(f"[Router] Strategy: {strategy}")
    return {**state, "strategy": strategy}


def retriever_node(state: RAGState) -> RAGState:
    """Node 2: Retrieve chunks from FAISS"""
    print(f"[Retriever] Strategy: {state['strategy']}")
    chunks = retrieve_chunks(
        state["question"],
        state["doc_name"],
        state["strategy"]
    )
    print(f"[Retriever] Retrieved {len(chunks)} chunks")
    return {**state, "chunks": chunks}


def grader_node(state: RAGState) -> RAGState:
    """Node 3: Grade chunks for relevance"""
    print(f"[Grader] Grading {len(state['chunks'])} chunks...")
    relevant = grade_chunks(state["question"], state["chunks"])
    print(f"[Grader] {len(relevant)} relevant chunks found")

    retries = state.get("retrieval_retries", 0)

    if not relevant and retries < MAX_RETRIES:
        # Rewrite query and try again
        new_question = rewrite_query(state["question"])
        print(f"[Grader] No relevant chunks. Rewriting query: {new_question}")
        return {
            **state,
            "relevant_chunks": [],
            "question": new_question,
            "retrieval_retries": retries + 1
        }

    return {
        **state,
        "relevant_chunks": relevant,
        "retrieval_retries": retries
    }


def generator_node(state: RAGState) -> RAGState:
    """Node 4: Generate answer from relevant chunks"""
    chunks = state["relevant_chunks"] or state["chunks"]
    print(f"[Generator] Generating answer from {len(chunks)} chunks...")

    answer = generate_answer(state["question"], chunks)
    sources = extract_sources(chunks)
    iterations = state.get("iterations", 0) + 1

    print(f"[Generator] Answer generated (iteration {iterations})")
    return {
        **state,
        "answer": answer,
        "sources": sources,
        "iterations": iterations
    }


def hallucination_node(state: RAGState) -> RAGState:
    """Node 5: Check if answer is grounded"""
    chunks = state["relevant_chunks"] or state["chunks"]
    print("[Hallucination Checker] Verifying answer...")

    is_grounded = check_hallucination(state["answer"], chunks)
    gen_retries = state.get("generation_retries", 0)

    print(f"[Hallucination Checker] Grounded: {is_grounded}")

    if not is_grounded and gen_retries < MAX_RETRIES:
        return {
            **state,
            "is_grounded": False,
            "generation_retries": gen_retries + 1
        }

    return {**state, "is_grounded": is_grounded}


# ── CONDITIONAL EDGE FUNCTIONS ──────────────────────────

def should_retrieve_again(state: RAGState) -> str:
    """After grading: go to generator or retrieve again"""
    if state["relevant_chunks"]:
        return "generator"
    if state.get("retrieval_retries", 0) >= MAX_RETRIES:
        return "generator"
    return "retriever"


def should_generate_again(state: RAGState) -> str:
    """After hallucination check: end or regenerate"""
    if state["is_grounded"]:
        return "end"
    if state.get("generation_retries", 0) >= MAX_RETRIES:
        return "end"
    return "generator"


# ── BUILD THE GRAPH ─────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("generator", generator_node)
    graph.add_node("hallucination_checker", hallucination_node)

    # Set entry point
    graph.set_entry_point("router")

    # Define edges
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "grader")

    # Conditional: after grading
    graph.add_conditional_edges(
        "grader",
        should_retrieve_again,
        {
            "generator": "generator",
            "retriever": "retriever"
        }
    )

    # After generating always check hallucination
    graph.add_edge("generator", "hallucination_checker")

    # Conditional: after hallucination check
    graph.add_conditional_edges(
        "hallucination_checker",
        should_generate_again,
        {
            "end": END,
            "generator": "generator"
        }
    )

    return graph.compile()


# Compile once at module level
rag_graph = build_rag_graph()


def run_pipeline(question: str, doc_name: str) -> dict:
    """
    Run the full adaptive RAG pipeline.
    Returns answer, strategy, iterations, grounded status, sources.
    """
    initial_state = RAGState(
        question=question,
        doc_name=doc_name,
        strategy="SIMPLE",
        chunks=[],
        relevant_chunks=[],
        answer="",
        is_grounded=False,
        sources=[],
        iterations=0,
        retrieval_retries=0,
        generation_retries=0
    )

    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "strategy": final_state["strategy"],
        "iterations": final_state["iterations"],
        "is_grounded": final_state["is_grounded"],
        "sources": final_state["sources"]
    }
```

---

## STEP 12 — CREATE backend/main.py

FastAPI application connecting everything.

```python
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx

from models import (
    UploadResponse, QueryRequest,
    QueryResponse, HealthResponse, SourceItem
)
from ingest import ingest_pdf
from graph import run_pipeline

load_dotenv()

app = FastAPI(
    title="Adaptive RAG Security Assistant",
    description="GDPR-compliant adaptive RAG for cybersecurity documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("data/docs", exist_ok=True)
os.makedirs("indexes", exist_ok=True)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a security PDF document."""

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    doc_name = Path(file.filename).stem.replace(" ", "_")
    file_path = f"data/docs/{file.filename}"

    # Save uploaded file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Ingest into FAISS
    try:
        chunks_count = ingest_pdf(file_path, doc_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )

    return UploadResponse(
        message=f"Successfully indexed {file.filename}",
        doc_name=doc_name,
        chunks_indexed=chunks_count,
        gdpr_compliant=True
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document using the adaptive RAG pipeline.
    Router → Retriever → Grader → Generator → Hallucination Checker
    """
    index_path = f"indexes/{request.doc_name}_faiss"
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_name}' not found. Upload it first."
        )

    try:
        result = run_pipeline(request.question, request.doc_name)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )

    sources = [
        SourceItem(
            page=s["page"],
            content_preview=s["content_preview"]
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        strategy=result["strategy"],
        iterations=result["iterations"],
        is_grounded=result["is_grounded"],
        sources=sources,
        gdpr_compliant=True
    )


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    docs = []
    for path in Path("indexes").glob("*_faiss"):
        doc_name = path.name.replace("_faiss", "")
        docs.append({"name": doc_name, "index_path": str(path)})
    return {"documents": docs}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ollama_url}/api/tags",
                timeout=3
            )
        ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        ollama_status = "not running — run: ollama serve"

    return HealthResponse(
        status="running",
        ollama_status=ollama_status,
        gdpr_compliant=True,
        local_inference=True
    )
```

---

## STEP 13 — CREATE frontend/styles/globals.css

```css
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: #060810;
  color: #e2e8f0;
  font-family: 'Courier New', monospace;
  min-height: 100vh;
}

.container {
  max-width: 860px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

.header {
  text-align: center;
  margin-bottom: 40px;
}

.header h1 {
  font-size: 26px;
  font-weight: 700;
  color: #00d4ff;
  margin-bottom: 6px;
}

.header p { font-size: 13px; color: #475569; }

.gdpr-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 14px;
  background: rgba(16,185,129,0.08);
  border: 1px solid rgba(16,185,129,0.2);
  border-radius: 100px;
  font-size: 11px;
  color: #10b981;
  margin-top: 10px;
}

.nav {
  display: flex;
  gap: 20px;
  margin-bottom: 28px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  padding-bottom: 14px;
}

.nav a {
  color: #475569;
  text-decoration: none;
  font-size: 13px;
  transition: color 0.2s;
}

.nav a:hover, .nav a.active { color: #00d4ff; }

.card {
  background: #0d1117;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 20px;
}

.upload-zone {
  border: 2px dashed rgba(0,212,255,0.25);
  border-radius: 10px;
  padding: 44px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.upload-zone:hover {
  border-color: rgba(0,212,255,0.5);
  background: rgba(0,212,255,0.02);
}

.upload-zone input { display: none; }

.btn {
  padding: 11px 22px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.2s;
  font-family: inherit;
}

.btn-primary { background: #00d4ff; color: #000; }
.btn-primary:hover { filter: brightness(1.1); transform: translateY(-1px); }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

.input-row {
  display: flex;
  gap: 10px;
  margin-bottom: 16px;
}

.input-row input {
  flex: 1;
  padding: 11px 14px;
  background: #111827;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  color: #e2e8f0;
  font-family: inherit;
  font-size: 13px;
  outline: none;
}

.input-row input:focus {
  border-color: rgba(0,212,255,0.35);
}

.answer-box {
  background: #111827;
  border: 1px solid rgba(0,212,255,0.12);
  border-radius: 8px;
  padding: 18px;
  font-size: 13px;
  line-height: 1.8;
  color: #cbd5e1;
  white-space: pre-wrap;
}

.meta-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}

.strategy-badge {
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
}

.strategy-simple {
  background: rgba(0,212,255,0.1);
  color: #00d4ff;
  border: 1px solid rgba(0,212,255,0.2);
}

.strategy-multi {
  background: rgba(124,58,237,0.1);
  color: #a78bfa;
  border: 1px solid rgba(124,58,237,0.2);
}

.strategy-decompose {
  background: rgba(245,158,11,0.1);
  color: #fbbf24;
  border: 1px solid rgba(245,158,11,0.2);
}

.grounded-badge {
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 11px;
  font-weight: 600;
}

.grounded-yes {
  background: rgba(16,185,129,0.1);
  color: #34d399;
  border: 1px solid rgba(16,185,129,0.2);
}

.grounded-no {
  background: rgba(239,68,68,0.1);
  color: #f87171;
  border: 1px solid rgba(239,68,68,0.2);
}

.source-card {
  background: #111827;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
}

.source-page {
  font-size: 11px;
  color: #a78bfa;
  margin-bottom: 4px;
  font-weight: 600;
}

.source-preview {
  font-size: 12px;
  color: #475569;
  line-height: 1.5;
}

.section-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: #374151;
  margin-bottom: 10px;
  margin-top: 18px;
}

.success { color: #34d399; font-size: 13px; margin-top: 14px; }
.error { color: #f87171; font-size: 13px; margin-top: 14px; }
.loading { color: #00d4ff; font-size: 13px; margin-top: 12px; }

.health-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-size: 13px;
}

.health-row:last-child { border-bottom: none; }
.status-ok { color: #34d399; }
.status-err { color: #f87171; }
```

---

## STEP 14 — CREATE frontend/pages/index.js

Upload page.

```javascript
import { useState, useRef } from 'react'
import Link from 'next/link'
import axios from 'axios'
import '../styles/globals.css'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileRef = useRef()

  const handleFile = (e) => {
    setFile(e.target.files[0])
    setResult(null)
    setError(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await axios.post(`${API}/upload`, form)
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="header">
        <h1>🔒 Adaptive RAG Security Assistant</h1>
        <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
        <div className="gdpr-badge">● Zero data leaves your machine</div>
      </div>

      <nav className="nav">
        <Link href="/" className="active">Upload</Link>
        <Link href="/chat">Ask Questions</Link>
        <Link href="/health">System Health</Link>
      </nav>

      <div className="card">
        <div className="upload-zone" onClick={() => fileRef.current.click()}>
          <input ref={fileRef} type="file" accept=".pdf" onChange={handleFile} />
          {file
            ? <p style={{ color: '#00d4ff' }}>📄 {file.name}</p>
            : <>
                <p style={{ fontSize: '28px', marginBottom: '10px' }}>📑</p>
                <p style={{ color: '#475569' }}>Click to upload security PDF</p>
                <p style={{ color: '#1f2937', fontSize: '11px', marginTop: '6px' }}>
                  CVE reports · OWASP docs · Pentest reports · Security advisories
                </p>
              </>
          }
        </div>

        {file && (
          <div style={{ textAlign: 'center', marginTop: '16px' }}>
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? 'Indexing...' : 'Upload & Index Document'}
            </button>
          </div>
        )}

        {loading && (
          <div className="loading">
            <p>⚙️ Creating semantic chunks...</p>
            <p>⚙️ Generating Ollama embeddings...</p>
            <p>⚙️ Building FAISS index...</p>
          </div>
        )}

        {result && (
          <div className="success">
            <p>✅ <strong>{result.doc_name}</strong> indexed successfully</p>
            <p style={{ marginTop: '6px' }}>
              📦 {result.chunks_indexed} chunks indexed in FAISS
            </p>
            <p style={{ marginTop: '10px' }}>
              <Link href="/chat" style={{ color: '#00d4ff' }}>
                → Start asking questions
              </Link>
            </p>
          </div>
        )}

        {error && <p className="error">❌ {error}</p>}
      </div>
    </div>
  )
}
```

---

## STEP 15 — CREATE frontend/pages/chat.js

Q&A page showing pipeline metadata.

```javascript
import { useState } from 'react'
import Link from 'next/link'
import axios from 'axios'
import '../styles/globals.css'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function ChatPage() {
  const [question, setQuestion] = useState('')
  const [docName, setDocName] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleQuery = async () => {
    if (!question.trim() || !docName.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await axios.post(`${API}/query`, {
        question,
        doc_name: docName
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  const strategyClass = (s) => {
    if (!s) return ''
    if (s === 'SIMPLE') return 'strategy-simple'
    if (s === 'MULTI') return 'strategy-multi'
    return 'strategy-decompose'
  }

  return (
    <div className="container">
      <div className="header">
        <h1>🔒 Adaptive RAG Security Assistant</h1>
        <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
        <div className="gdpr-badge">● Zero data leaves your machine</div>
      </div>

      <nav className="nav">
        <Link href="/">Upload</Link>
        <Link href="/chat" className="active">Ask Questions</Link>
        <Link href="/health">System Health</Link>
      </nav>

      <div className="card">
        <div className="input-row">
          <input
            type="text"
            placeholder="Document name (e.g. owasp_top_10)"
            value={docName}
            onChange={e => setDocName(e.target.value)}
            style={{ maxWidth: '240px' }}
          />
          <input
            type="text"
            placeholder="Ask a security question..."
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleQuery()}
          />
          <button
            className="btn btn-primary"
            onClick={handleQuery}
            disabled={loading || !question || !docName}
          >
            {loading ? '...' : 'Ask'}
          </button>
        </div>

        <p style={{ fontSize: '11px', color: '#1f2937' }}>
          Try: "What are the critical vulnerabilities?" ·
          "What fixes are recommended?" ·
          "Compare injection and XSS risks"
        </p>

        {loading && (
          <div className="loading">
            🧠 Running adaptive RAG pipeline...
          </div>
        )}

        {error && <p className="error">❌ {error}</p>}

        {result && (
          <>
            <div className="meta-row" style={{ marginTop: '20px' }}>
              <span className={`strategy-badge ${strategyClass(result.strategy)}`}>
                Strategy: {result.strategy}
              </span>
              <span className={`grounded-badge ${result.is_grounded ? 'grounded-yes' : 'grounded-no'}`}>
                {result.is_grounded ? '✓ Grounded' : '⚠ Unverified'}
              </span>
              <span style={{ fontSize: '11px', color: '#374151', paddingTop: '4px' }}>
                {result.iterations} iteration{result.iterations !== 1 ? 's' : ''}
              </span>
            </div>

            <div className="section-label">Answer</div>
            <div className="answer-box">{result.answer}</div>

            {result.sources?.length > 0 && (
              <>
                <div className="section-label">
                  Sources ({result.sources.length})
                </div>
                {result.sources.map((src, i) => (
                  <div key={i} className="source-card">
                    <div className="source-page">📄 Page {src.page}</div>
                    <div className="source-preview">{src.content_preview}</div>
                  </div>
                ))}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
```

---

## STEP 16 — CREATE frontend/pages/health.js

System health page.

```javascript
import { useState, useEffect } from 'react'
import Link from 'next/link'
import axios from 'axios'
import '../styles/globals.css'

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function HealthPage() {
  const [health, setHealth] = useState(null)
  const [docs, setDocs] = useState([])

  useEffect(() => {
    axios.get(`${API}/health`).then(r => setHealth(r.data)).catch(() => {})
    axios.get(`${API}/documents`).then(r => setDocs(r.data.documents)).catch(() => {})
  }, [])

  return (
    <div className="container">
      <div className="header">
        <h1>🔒 Adaptive RAG Security Assistant</h1>
        <p>GDPR-Compliant · 100% Local · LangGraph Pipeline</p>
        <div className="gdpr-badge">● Zero data leaves your machine</div>
      </div>

      <nav className="nav">
        <Link href="/">Upload</Link>
        <Link href="/chat">Ask Questions</Link>
        <Link href="/health" className="active">System Health</Link>
      </nav>

      <div className="card">
        <div className="section-label">System Status</div>

        {health ? (
          <>
            <div className="health-row">
              <span>Backend Status</span>
              <span className="status-ok">● {health.status}</span>
            </div>
            <div className="health-row">
              <span>Ollama LLM</span>
              <span className={health.ollama_status === 'connected'
                ? 'status-ok' : 'status-err'}>
                ● {health.ollama_status}
              </span>
            </div>
            <div className="health-row">
              <span>GDPR Compliant</span>
              <span className="status-ok">● Yes — 100% local</span>
            </div>
            <div className="health-row">
              <span>External API Calls</span>
              <span className="status-ok">● None</span>
            </div>
            <div className="health-row">
              <span>Vector Store</span>
              <span className="status-ok">● FAISS (local)</span>
            </div>
            <div className="health-row">
              <span>Pipeline</span>
              <span className="status-ok">● LangGraph Adaptive RAG</span>
            </div>
          </>
        ) : (
          <p className="loading">Checking system status...</p>
        )}
      </div>

      {docs.length > 0 && (
        <div className="card">
          <div className="section-label">Indexed Documents ({docs.length})</div>
          {docs.map((doc, i) => (
            <div key={i} className="health-row">
              <span>📄 {doc.name}</span>
              <span className="status-ok">● Ready</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

---

## STEP 17 — CREATE frontend/.env.local

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## STEP 18 — INSTALL AND RUN

### Backend
```bash
cd adaptive-rag-security
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cd backend
uvicorn main:app --reload --port 8000
```

### Ollama (separate terminal)
```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Frontend (separate terminal)
```bash
cd frontend
npx create-next-app@14 . --no-typescript --no-tailwind --no-eslint --app=false
npm install axios
npm run dev
```

---

## STEP 19 — TEST THE FULL PIPELINE

```
1. Go to http://localhost:3000
2. Upload OWASP Top 10 PDF
3. Wait for indexing (1-2 minutes)
4. Go to Ask Questions page
5. Document name: owasp_top_10
6. Ask: "What are the critical security risks?"
7. See:
   - Strategy used (SIMPLE/MULTI/DECOMPOSE)
   - Grounded status (✓ Grounded)
   - Iterations count
   - Answer with page citations
   - Source cards
```

---

## DO NOT

- Do not use LangExtract
- Do not use PageIndex
- Do not use ChromaDB
- Do not use any cloud LLM API
- Do not add authentication
- Do not change the folder structure
- Do not add libraries not in requirements.txt

---

## WHAT EACH FILE DOES

```
ingest.py        → PDF to FAISS chunks
router.py        → Decides retrieval strategy
retriever.py     → Searches FAISS with strategy
grader.py        → Filters irrelevant chunks
generator.py     → Generates cited answer
hallucination.py → Verifies answer is grounded
graph.py         → LangGraph connects all nodes
main.py          → FastAPI exposes endpoints
index.js         → Upload page
chat.js          → Q&A page with pipeline metadata
health.js        → System status page
```

---

## BUILD COMPLETE ✅
```
