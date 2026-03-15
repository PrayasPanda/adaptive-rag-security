# 🧠 System Architecture
# Adaptive RAG Security Assistant

The Adaptive RAG Security Assistant leverages an advanced localized architecture. The core innovation of this system is the implementation of an **Agentic Workflow** using **LangGraph**. This ensures high data quality and reduces the common risks of LLM hallucination and irrelevant data retrieval.

---

## The 5 Nodes of LangGraph

The application is structured around a StateGraph that dictates how the input query flows through the system. Each state modification is handled by an independent localized node powered by **Ollama (Llama 3.2)**.

### Node 1: Router (`backend/router.py`)
- **Purpose**: Classifies user questions and determines the optimal retrieval strategy.
- **Strategies**:
  - `SIMPLE`: A single-fact query.
  - `MULTI`: A complex question requiring comprehensive, multi-angle search.
  - `DECOMPOSE`: A query that must be broken down into simpler, sequential sub-questions.
- **Logic**: Evaluates the incoming prompt against the LLM to output a precise categorical strategy.

### Node 2: Retriever (`backend/retriever.py`)
- **Purpose**: Executes semantic search against the FAISS vector database.
- **Logic**: Uses the strategy designated by the **Router** to pull chunks. If the strategy is `MULTI`, it expands the query into three entirely different parallel searches.

### Node 3: Grader (`backend/grader.py`)
- **Purpose**: Analyzes the raw chunks retrieved by the **Retriever** before passing them back to the user.
- **Logic**: The grader LLM checks every single retrieved chunk against the original question. If the document chunks are irrelevant, the system triggers a **Rewrite Loop** (rewriting the original question to be more specific) and sends it back to Node 2 for re-retrieval.

### Node 4: Generator (`backend/generator.py`)
- **Purpose**: Synthesizes a cohesive final answer using *only* the chunks validated by the **Grader**.
- **Logic**: It strict-formats the answer to include direct citations (e.g., "[Page X]").

### Node 5: Hallucination Checker (`backend/hallucination.py`)
- **Purpose**: Acts as a final "Fact-Checking" barrier before displaying the output to the frontend.
- **Logic**: It compares the generated answer directly against the context provided by FAISS to ensure no out-of-bounds claims (like fabricated CVEs or CVSS scores) were made. If it detects a hallucination, it forces the **Generator** to try again.

---

## Data Privacy & Flow

Because this tool is built for Cybersecurity professionals, **GDPR Compliance and Data Sovereignty** are the highest priority.

### Vector Ingestion Flow (`backend/ingest.py`)
1. **Upload**: A PDF (up to hundreds of pages) is passed from the Next.js frontend to the FastAPI backend locally via Multipart Form Data.
2. **Chunking**: The document is split using a `RecursiveCharacterTextSplitter` (800 char sizes, 100 char overlap).
3. **Embedding**: Chunks are processed by **Nomic-Embed-Text** running exclusively via the local Ollama instance on `localhost:11434`.
4. **Storage**: The vectors are saved natively to the file system in the `backend/indexes` directory using **FAISS (Facebook AI Similarity Search)**.

At no point does the data leave the host machine.
