import re
from app.pipeline.retrieval.reranker import RetrievedChunk

def fix_hallucinated_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> str:
    # Provide a more robust dictionary of keywords mapped to exact source strings
    category_keywords = {
        "Clinvar": [r'\bpathogenic\b', r'\bbenign\b', r'\bclinvar\b', r'\bclassified as\b', r'\brs\d+\b'],
        "ClinGen": [r'\bclingen\b', r'\bexpert panel\b', r'\bgene-disease validity\b', r'\bactionability\b'],
        "gnomAD":  [r'\bgnomad\b', r'\bba1\b', r'\ballele.?frequency\b', r'\bpopulation\b', r'\babsent from controls\b'],
        "ACMG":    [r'\bPM\d\b', r'\bPS\d\b', r'\bPVS\d\b', r'\bPP\d\b', r'\bBS\d\b', r'\bBA\d\b', r'\bBP\d\b', r'\bcriteria\b', r'\bacmg\b'],
        "NCCN":    [r'\bnccn\b', r'\bscreening\b', r'\bmastectomy\b', r'\bsurveillance\b', r'\bmri\b', r'\bmammography\b', r'\bprotocol\b']
    }

    # First, gather exact chunk references into a map so we don't invent anything
    source_to_ref = {}
    for chunk in used_chunks:
        if chunk.source not in source_to_ref:
            source_to_ref[chunk.source] = chunk.reference

    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)
    
    # Process from back to front to avoid index shifting when replacing
    matches = list(inline_pattern.finditer(answer_text))
    for match in reversed(matches):
        cited_source = match.group(1).strip()
        
        start_pos = max(0, match.start() - 200)
        context_before = answer_text[start_pos:match.start()].lower()

        # Score which category this citation actually belongs to
        best_category = None
        max_hits = 0
        for cat, patterns in category_keywords.items():
            hits = sum(1 for p in patterns if re.search(p, context_before, re.IGNORECASE))
            if hits > max_hits:
                max_hits = hits
                best_category = cat

        # If we scored at least 2 points on a category, let's enforce it
        if best_category and max_hits >= 2:
            # Find the actual chunk source name that corresponds to our category
            # (e.g., matching "ACMG" category to "ACMG_2015" actual source)
            actual_source = None
            for s in source_to_ref.keys():
                if best_category.lower() in s.lower():
                    actual_source = s
                    break
            
            if actual_source:
                correct_ref = source_to_ref[actual_source]
                new_citation = f"[Source: {actual_source}, Reference: {correct_ref}]"
                # Replace exactly at the indices of THIS specific match
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]
        else:
            # If we don't have enough context to change it, just make sure the ref matches the cited source
            for s, ref in source_to_ref.items():
                if cited_source.lower() in s.lower() and match.group(2).strip() != ref:
                    new_citation = f"[Source: {s}, Reference: {ref}]"
                    answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]
                    break

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
