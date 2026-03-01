from app.pipeline.retrieval.reranker import RetrievedChunk

CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

def grade_chunk(query: str, chunk: RetrievedChunk) -> str:

    if chunk.score >= 2.0:       # high confidence — clearly relevant
        return CORRECT
    elif chunk.score >= -6.0:     # low confidence but not clearly irrelevant
        return AMBIGUOUS
    else: 
        return INCORRECT

def evaluate_chunks(query: str, chunks: list[RetrievedChunk]) -> dict:

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
