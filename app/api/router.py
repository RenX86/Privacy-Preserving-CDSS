from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse
from app.pipeline.generation.guardrails import SAFE_FAILURE_MESSAGE
from app.pipeline.graph.workflow import cdss_app

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}


@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    query = request.query
    
    # ── 1. INVOKE LANGGRAPH ────────────────────────────────────
    print(f"\n[STARTING LANGGRAPH] Query: {query}")
    initial_state = {"query": query}
    
    # This single line runs your entire flowchart (all 7 nodes!)
    final_state = cdss_app.invoke(initial_state)
    
    # ── 2. RETURN THE OUTPUT ───────────────────────────────────
    print(f"[LANGGRAPH FINISHED] Confidence: {final_state.get('confidence', 'low')}")
    return QueryResponse(
        answer=final_state.get("final_answer", SAFE_FAILURE_MESSAGE),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", "low"),
        safe_failure=False if final_state.get("final_answer") else True
    )
