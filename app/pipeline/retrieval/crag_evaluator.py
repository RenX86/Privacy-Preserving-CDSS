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

    print(f"\n[CRAG] Grading {len(chunks)} PDF chunks:")
    print(f"       Thresholds: CORRECT >= 2.0 | AMBIGUOUS >= -6.0 | INCORRECT < -6.0")
    print(f"       {'score':>8}  {'grade':>12}  source / text preview")
    print(f"       {'-'*70}")

    for chunk in chunks:
        grade = grade_chunk(query, chunk)
        results[grade].append(chunk)
        grade_symbol = "[CORRECT]" if grade == CORRECT else ("[AMBIGUOUS]" if grade == AMBIGUOUS else "[INCORRECT]")
        preview = chunk.text.replace("\n", " ")[:120]
        print(f"       {chunk.score:>8.3f}  {grade_symbol:>12}  [{chunk.source}] {preview}")

    return results

def has_sufficient_context(evaluation: dict) -> bool:
    return len(evaluation[CORRECT]) > 0
