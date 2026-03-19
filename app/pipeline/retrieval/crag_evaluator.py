import sys
from app.pipeline.retrieval.reranker import RetrievedChunk

CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

# Tuned thresholds based on observed BGE score distribution.
# In pilot queries all chunks scored 0.001–0.195 with CORRECT_THRESHOLD=0.0,
# meaning nothing was ever filtered. The values below enforce a meaningful floor.
#
# CORRECT   >= 0.05 : chunk is clearly relevant to the query
# AMBIGUOUS  0.01–0.05: borderline — keep but place after CORRECT chunks
# INCORRECT < 0.01 : BGE considers this chunk unrelated — dropped
CORRECT_THRESHOLD  = 0.05
AMBIGUOUS_THRESHOLD = 0.01

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

    n_dropped = len(results[INCORRECT])
    n_kept    = len(results[CORRECT]) + len(results[AMBIGUOUS])
    print(f"\n[CRAG] Kept {n_kept} chunks | Dropped {n_dropped} INCORRECT chunks\n")

    return results

def has_sufficient_context(evaluation: dict) -> bool:
    return len(evaluation[CORRECT]) > 0
