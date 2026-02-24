from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse, citation
from app.pipeline.decomposition import decompose_query

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}

@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Handles incoming clinical queries by decomposing them into sub-queries and routing them to appropriate data sources.
    """
    sub_queries = decompose_query(request.query)

    routing_summary = []

    for sq in sub_queries:
        routing_summary.append(f"[{sq.target.upper()}] {sq.text}")

    return QueryResponse(
        answer="Query decomposed succesfully. Pipeline not yet fully connected.",
        citations=[
            citation(source="Decomposer", reference=summary)
            for summary in routing_summary
        ],
        confidence="low",
        safe_failure=False
    )
