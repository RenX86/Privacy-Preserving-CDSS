from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    source: str
    reference: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str
    safe_failure: bool

class ChunkDetail(BaseModel):
    text: str
    source: str
    reference: str
    score: float
    grade: str

class NodeTrace(BaseModel):
    node: str
    duration_ms: int
    summary: str
    data: dict

class InstrumentedResponse(BaseModel):
    answer: str
    draft_answer: str
    citations: List[Citation]
    confidence: str
    safe_failure: bool
    gene: str | None
    total_duration_ms: int
    trace: List[NodeTrace]