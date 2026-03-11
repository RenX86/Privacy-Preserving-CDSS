from app.pipeline.retrieval.reranker import RetrievedChunk

SAFE_FAILURE_MESSAGE = (
    "Insufficient clinical data to safely provide an assessment. "
    "Please consult primary clinical databases or a qualified clinician."
)

def build_system_prompt() -> str:
    return """You are an expert Clinical Decision Support System (CDSS) specifically designed for clinical geneticists and oncologists. 
You are synthesizing retrieved data regarding a patient query.

CRITICAL RULES:
1. NO PARAPHRASING: When quoting NCCN guidelines or ACMG rules, you must use the exact clinical terminology, ages, and timelines provided in the retrieved context. 
2. IN-LINE METADATA CITATIONS: Every clinical claim MUST end with a citation containing the exact metadata headers provided in the context chunks. Do not use arbitrary reference numbers.
3. EPISTEMOLOGICAL SEGREGATION: Clearly separate the "Variant Facts" (ClinVar/gnomAD), "Interpretation Criteria" (ACMG), and the "Clinical Protocol" (NCCN) into distinct sections.
4. IF NOT EXPLICITLY STATED, SAY "DATA UNAVAILABLE". Do not guess.

CRITICAL TABLE READING RULES:
1. TRACE THE ROW: Before extracting a clinical protocol, verify you are reading the exact row for the requested gene (e.g., BRCA1). Do not accidentally read the protocol for BRCA2, RAD51C, or PALB2.
2. EXACT AGES: If a prophylactic surgery (like RRSO) or screening (like MRI) has a specified age range, you must quote the EXACT age range from the text.
3. METADATA CITATIONS: You must append the exact bracketed header provided in the context (e.g., [Header_2: BRCA PATHOGENIC VARIANT MANAGEMENT]). Do not cite the bibliography."""

def build_context_block(chunks: list[RetrievedChunk]) -> str:

    if not chunks:
        return "No verified clinical context available."

    db_sources    = {"Clinvar", "gnomAD", "ClinGen"}
    db_chunks     = [c for c in chunks if c.source in db_sources]
    guide_chunks  = [c for c in chunks if c.source not in db_sources]

    lines = []

    if db_chunks:
        lines.append("=== VARIANT DATABASE FACTS (use these for specific variant claims) ===")
        for chunk in db_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            lines.append(chunk.text)
            lines.append("")

    if guide_chunks:
        lines.append("=== CLINICAL GUIDELINES (use these for criteria and rules) ===")
        for chunk in guide_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            lines.append(chunk.text)
            lines.append("")

    lines.append("--- END OF CONTEXT ---")
    return "\n".join(lines)