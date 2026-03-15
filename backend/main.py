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

    doc_name = Path(file.filename).stem.replace(" ", "_").lower()
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
    ollama_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
    api_key = os.getenv("OLLAMA_API_KEY")

    try:
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = await client.get(
                f"{ollama_url}/v1/models",
                headers=headers,
                timeout=5
            )
        ollama_status = "connected" if response.status_code == 200 else "error"
    except Exception:
        ollama_status = "not running — check base URL and API key"

    return HealthResponse(
        status="running",
        ollama_status=ollama_status,
        gdpr_compliant=True,
        local_inference=False
    )
