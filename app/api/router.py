from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse, citations

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}

@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    return QueryResponse(
        answer=f"Recieved your Query: '{request.query}'.Pipeline not yet Connected.",
        citations=[
            citation(source="system", reference="Placeholder response")
        ],
        confidence="low",
        safe_failure=False
    )