from app.pipeline.retrieval.reranker import RetrievedChunk

CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

def grade_chunk(query: str, chunk: RetrievedChunk) -> str:

    if chunk.score >= 3.0:
        return CORRECT
    elif chunk.score >= 0.0:
        return AMBIGUOUS
    else: 
        return INCORRECT

def evaluate(query: str, chunk: list[RetrievedChunk]) -> dict:

    results = {
        CORRECT: [],
        AMBIGUOUS: [],
        INCORRECT: []
    }

    for chunk in chunks:
        grade = grade_chunk(query, chunk)
        results[grade].append(chunk)

    return results

def has_sufficient_context(evaluation: dict) -> bool:
    return len(evaluation[CORRECT]) > 0
