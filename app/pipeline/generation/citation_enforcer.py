import re
from app.pipeline.retrieval.reranker import RetrievedChunk

def extract_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> list[dict]:

    citations = []
    seen = set()

    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*([^\]]+)\]')
    for match in inline_pattern.finditer(answer_text):
        source = match.group(1).strip()
        reference = match.group(2).strip()
        key = (source, reference)
        if key not in seen:
            citations.append({"source": source, "reference": reference})
            seen.add(key)

    for chunk in used_chunks:
        key = (chunk.source, chunk.reference)
        if key not in seen:
            citations.append({"source": chunk.source, "reference": chunk.reference})
            seen.add(key)

    return citations
