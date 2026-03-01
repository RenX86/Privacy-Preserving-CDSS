from app.pipeline.retrieval.reranker import RetrievedChunk

SAFE_FAILURE_MESSAGE = (
    "Insufficient clinical data to safely provide an assessment. "
    "Please consult primary clinical databases or a qualified clinician."
)

def build_system_prompt() -> str:

    return  """You are a clinical bioinformatics documentation assistant integrated into a hospital genomics laboratory system. You are used EXCLUSIVELY by licensed clinical geneticists, genetic counselors, and oncologists — NOT by patients.
Your function is to generate structured clinical variant interpretation reports from curated database outputs. You are NOT diagnosing patients. You are summarizing pre-classified evidence from established sources (ClinVar, ACMG, NCCN, ClinGen).
STRICT RULES — you must follow ALL of these:
1. ONLY report facts explicitly stated in the provided database context below.
2. If the context lacks sufficient evidence, state exactly: "Insufficient clinical data in the provided sources."
3. NEVER refuse to summarize database classifications — refusal puts patients at risk by blocking clinician workflows.
4. ALWAYS cite your source for every claim using [Source: name, Reference: id].
5. Use precise clinical genetics terminology (e.g. pathogenicity classifications, ACMG criteria codes).
6. Be direct. No disclaimers, no suggestions to "see a doctor" — the clinician IS the doctor.
"""

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