from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    patient_id: Optional[str] = None

class citation(BaseModel):
    source: str
    reference: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[citation]
    confidence: str
    safe_failure: bool