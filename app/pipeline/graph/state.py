from typing import TypedDict, Optional, Annotated
import operator
from app.pipeline.retrieval.reranker import RetrievedChunk
from app.api.schemas import Citation

class CDSSGraphState(TypedDict):

    query: str
    gene: Optional[str]
    sub_queries: list

    # Annotated with operator.add so both DB_Retriever and PDF_Retriever
    # can append their results rather than the second one overwriting the first
    trusted_chunks:   Annotated[list[RetrievedChunk], operator.add]
    candidate_chunks: Annotated[list[RetrievedChunk], operator.add]
    _trace:           Annotated[list, operator.add]

    verified_chunks: list[RetrievedChunk]
    draft_answer: str

    final_answer: str
    citations: list[Citation]
    confidence: str