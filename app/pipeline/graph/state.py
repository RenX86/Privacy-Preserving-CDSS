from typing import TypedDict, Optional
from app.pipeline.retrieval.reranker import RetrievedChunk
from app.api.schemas import Citation

class CDSSGraphState(TypedDict):

    query: str
    gene: Optional[str]
    sub_queries: list


    trusted_chunks: list[RetrievedChunk]
    candidate_chunks: list[RetrievedChunk]


    verified_chunks: list[RetrievedChunk]
    draft_answer: str


    final_answer: str
    citations: list[Citation]
    confidence: str