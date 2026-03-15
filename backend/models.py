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
