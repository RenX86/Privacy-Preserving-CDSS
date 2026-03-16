import re
from app.pipeline.retrieval.reranker import RetrievedChunk

def fix_hallucinated_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> str:
    # Keywords that UNIQUELY identify each source category
    # IMPORTANT: Do NOT include generic clinical words like 'pathogenic'/'benign' in Clinvar
    # Because those words appear everywhere and cause ACMG citations to be misclassified
    category_keywords = {
        "Clinvar": [r'\bclinvar\b', r'\bclassified as\b', r'\brs\d+\b', r'\bclinical significance\b'],
        #           ↑ removed 'pathogenic' and 'benign' — they appear in ALL clinical text
        #           ↑ kept rsID pattern (very specific to ClinVar)
        "ClinGen": [r'\bclingen\b', r'\bexpert panel\b', r'\bgene-disease validity\b', r'\bactionability\b'],
        "gnomAD":  [r'\bgnomad\b', r'\bba1\b', r'\ballele.?frequency\b', r'\bpopulation frequency\b', r'\babsent from controls\b'],
        "ACMG":    [r'\bPM\d\b', r'\bPS\d\b', r'\bPVS\d\b', r'\bPP\d\b', r'\bBS\d\b', r'\bBA\d\b', r'\bBP\d\b', r'\bcriteria\b', r'\bacmg\b'],
        #           ↑ ACMG criteria codes are very specific — PM1, PS3, PVS1 etc. kept
        "NCCN":    [r'\bnccn\b', r'\bscreening\b', r'\bmastectomy\b', r'\bsurveillance\b', r'\bmri\b', r'\bmammography\b', r'\bprotocol\b']
    }

    # Build a map: source → reference, from actual retrieved chunks
    source_to_ref = {}
    for chunk in used_chunks:
        if chunk.source not in source_to_ref:
            source_to_ref[chunk.source] = chunk.reference

    # Build a set of known real source names (lowercase) for quick lookup
    known_sources_lower = {s.lower(): s for s in source_to_ref.keys()}

    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)

    matches = list(inline_pattern.finditer(answer_text))
    for match in reversed(matches):
        cited_source = match.group(1).strip()
        cited_ref    = match.group(2).strip()

        # ── GUARD: If the LLM cited a source that actually EXISTS in our chunks, ──
        # trust the source identity and only fix the reference if it's wrong.
        # This prevents the enforcer from "correcting" ACMG → ClinVar when
        # the text around the citation says "pathogenic" (a generic clinical word).
        cited_source_lower = cited_source.lower()
        actual_source_match = None
        for known_lower, known_real in known_sources_lower.items():
            if cited_source_lower in known_lower or known_lower in cited_source_lower:
                actual_source_match = known_real
                break

        if actual_source_match:
            # Source exists in our chunks — only fix the reference if it's wrong
            correct_ref = source_to_ref[actual_source_match]
            if cited_ref != correct_ref:
                new_citation = f"[Source: {actual_source_match}, Reference: {correct_ref}]"
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]
            # Source and ref are correct — leave it alone
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
            for s in source_to_ref.keys():
                if best_category.lower() in s.lower():
                    actual_source = s
                    break
            if actual_source:
                correct_ref = source_to_ref[actual_source]
                new_citation = f"[Source: {actual_source}, Reference: {correct_ref}]"
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]

    return answer_text


def extract_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> list[dict]:
    citations = []
    seen = set()

    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)
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
