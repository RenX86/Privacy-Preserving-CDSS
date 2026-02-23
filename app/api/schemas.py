from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None

class citations(BaseModel):
    answer: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[citations]
    confidence: str
    safe_failure: bool