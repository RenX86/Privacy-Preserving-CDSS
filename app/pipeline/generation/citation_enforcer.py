import re
from app.pipeline.retrieval.reranker import RetrievedChunk

def fix_hallucinated_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> str:

    source_to_ref = {}
    for chunk in used_chunks:
        source_to_ref[chunk.source.lower()] = chunk.reference
    
    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)

    def replace_match(match):
        source = match.group(1).strip()
        ref = match.group(2).strip()

        correct_ref = source_to_ref.get(source.lower())
        if correct_ref and correct_ref != ref:
            return f"[Source: {source}, Reference: {correct_ref}]"
        return match.group(0)

    answer_text = inline_pattern.sub(replace_match, answer_text)

    db_keywords = {
        "Clinvar":  [r'\brs\d+\b', r'\bpathogenic\b', r'\bbenign\b', r'\bclinvar\b', r'\bclassified as\b'],
        "ClinGen":  [r'\bclingen\b', r'\bexpert panel\b', r'\bgene-disease validity\b', r'\bactionability\b'],
        "gnomAD":   [r'\bgnomad\b', r'\bba1\b', r'\ballele.?frequency\b', r'\bpopulation database\b'],
    }

    for match in list(inline_pattern.finditer(answer_text)):
        cited_source = match.group(1).strip()
        start = max(0, match.start() - 200)
        context_before = answer_text[start:match.start()].lower()

        if cited_source.lower() not in ("clinvar", "clingen", "gnomad"):
            for correct_source, patterns in db_keywords.items():
                hits = sum(1 for p in patterns if re.search(p, context_before, re.IGNORECASE))
                if hits >= 2:
                    correct_ref = source_to_ref.get(correct_source.lower(), match.group(2).strip())
                    old_citation = match.group(0)
                    new_citation = f"[Source: {correct_source}, Reference: {correct_ref}]"
                    answer_text = answer_text.replace(old_citation, new_citation, 1)
                    break

    return answer_text

def extract_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> list[dict]:

    citations = []
    seen = set()

    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)
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
