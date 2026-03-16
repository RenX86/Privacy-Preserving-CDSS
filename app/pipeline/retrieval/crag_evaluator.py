import sys
from app.pipeline.retrieval.reranker import RetrievedChunk

CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

CORRECT_THRESHOLD  = 0.0    # BGE score > 0 means "chunk IS relevant" to this query
AMBIGUOUS_THRESHOLD = -3.0  # BGE score -3 to 0: uncertain — keep but deprioritise
# score < -3.0 → INCORRECT: BGE is confident this chunk is NOT relevant → dropped

def grade_chunk(query: str, chunk: RetrievedChunk) -> str:
    if chunk.score >= CORRECT_THRESHOLD:
        return CORRECT
    elif chunk.score >= AMBIGUOUS_THRESHOLD:
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
    print(f"       Thresholds: CORRECT >= {CORRECT_THRESHOLD} | AMBIGUOUS >= {AMBIGUOUS_THRESHOLD} | INCORRECT < {AMBIGUOUS_THRESHOLD}")
    print(f"       {'score':>8}  {'grade':>12}  source / text preview")
    print(f"       {'-'*70}")

    for chunk in chunks:
        grade = grade_chunk(query, chunk)
        results[grade].append(chunk)
        grade_symbol = "[CORRECT]" if grade == CORRECT else ("[AMBIGUOUS]" if grade == AMBIGUOUS else "[INCORRECT]")
        preview = chunk.text.replace("\n", " ")[:120]
        # Avoid UnicodeEncodeError on Windows terminals
        safe_preview = preview.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding) if sys.stdout.encoding else preview
        print(f"       {chunk.score:>8.3f}  {grade_symbol:>12}  [{chunk.source}] {safe_preview}")

    return results

def has_sufficient_context(evaluation: dict) -> bool:
    return len(evaluation[CORRECT]) > 0
