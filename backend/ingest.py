import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.getenv("OLLAMA_API_KEY")

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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_path = f"indexes/{doc_name}_faiss"
    vectorstore.save_local(index_path)

    print(f"[Ingest] Saved FAISS index to {index_path}")
    return len(chunks)


def load_vectorstore(doc_name: str) -> FAISS:
    """Load existing FAISS index for a document."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )
    index_path = f"indexes/{doc_name}_faiss"
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
