import re
from app.pipeline.retrieval.reranker import RetrievedChunk

def fix_hallucinated_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> str:
    # Keywords that UNIQUELY identify each source category
    category_keywords = {
        "Clinvar": [r'\bclinvar\b', r'\bclassified as\b', r'\brs\d+\b', r'\bclinical significance\b'],
        "ClinGen": [r'\bclingen\b', r'\bexpert panel\b', r'\bgene-disease validity\b', r'\bactionability\b'],
        "gnomAD":  [r'\bgnomad\b', r'\ballele.?frequency\b', r'\bpopulation frequency\b', r'\babsent from controls\b', r'\brare in population\b'],
        "Treatment": [r'\bchemotherapy\b', r'\bradiotherapy\b', r'\bneoadjuvant\b', r'\bsurgery\b', r'\bstaging\b', r'\bregimen\b'],
        "Detection": [r'\bnccn\b', r'\bscreening\b', r'\bmastectomy\b', r'\bsurveillance\b', r'\bmri\b', r'\bmammography\b', r'\bprotocol\b', r'\brrso\b', r'\bsalpingo\b'],
    }


    # ── Build: source → SET of all valid references (not just the first one) ──
    # Old code used first-wins: if source already in dict, skip all subsequent chunks.
    # This collapsed all page refs to a single page regardless of what was actually cited.
    source_to_refs: dict[str, set] = {}
    source_first_ref: dict[str, str] = {}  # fallback: first ref seen per source
    for chunk in used_chunks:
        source_to_refs.setdefault(chunk.source, set()).add(chunk.reference)
        if chunk.source not in source_first_ref:
            source_first_ref[chunk.source] = chunk.reference

    # Build a set of known real source names (lowercase) for quick lookup
    known_sources_lower = {s.lower(): s for s in source_to_refs.keys()}

    inline_pattern = re.compile(r'\[Source:\s*(.+?),\s*Reference:\s*([^\]]+)\]', re.IGNORECASE)

    matches = list(inline_pattern.finditer(answer_text))
    for match in reversed(matches):
        cited_source = match.group(1).strip()
        cited_ref    = match.group(2).strip()

        # ── GUARD: If the LLM cited a source that actually EXISTS in our chunks, ──
        # trust the source identity and only fix the reference if it's wrong.
        cited_source_lower = cited_source.lower()
        actual_source_match = None
        for known_lower, known_real in known_sources_lower.items():
            if cited_source_lower in known_lower or known_lower in cited_source_lower:
                actual_source_match = known_real
                break

        if actual_source_match:
            valid_refs = source_to_refs[actual_source_match]
            if cited_ref in valid_refs:
                # Citation is valid — leave it completely alone
                continue
            else:
                # Source is real but ref is wrong — remap to the first ref as fallback
                fallback_ref = source_first_ref[actual_source_match]
                new_citation = f"[Source: {actual_source_match}, Reference: {fallback_ref}]"
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]
            continue

        # ── HALLUCINATION: The cited source doesn't match any real chunk source ──
        # Now score the context to find the correct source
        start_pos = max(0, match.start() - 200)
        context_before = answer_text[start_pos:match.start()].lower()

        best_category = None
        max_hits = 0
        for cat, patterns in category_keywords.items():
            hits = sum(1 for p in patterns if re.search(p, context_before, re.IGNORECASE))
            if hits > max_hits:
                max_hits = hits
                best_category = cat

        # Need at least 2 keyword hits to confidently remap the citation
        if best_category and max_hits >= 2:
            actual_source = None
            for s in source_to_refs.keys():
                if best_category.lower() in s.lower():
                    actual_source = s
                    break
            if actual_source:
                fallback_ref = source_first_ref[actual_source]
                new_citation = f"[Source: {actual_source}, Reference: {fallback_ref}]"
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]

    return answer_text


def extract_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> list[dict]:
    citations = []
    seen = set()

    inline_pattern = re.compile(r'\[Source:\s*(.+?),\s*Reference:\s*([^\]]+)\]', re.IGNORECASE)
    for match in inline_pattern.finditer(answer_text):
        source = match.group(1).strip()
        reference = match.group(2).strip()
        key = (source.lower(), str(reference).lower())
        if key not in seen:
            citations.append({"source": source, "reference": reference})
            seen.add(key)

    # We NO LONGER blindly append ALL used_chunks. 
    # Only chunks actually cited by the LLM in the text will be included!
    
    return citations
