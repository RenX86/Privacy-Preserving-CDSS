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