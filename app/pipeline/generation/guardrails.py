from app.pipeline.retrieval.reranker import RetrievedChunk

SAFE_FAILURE_MESSAGE = (
    "Insufficient clinical data to safely provide an assessment. "
    "Please consult primary clinical databases or a qualified clinician."
)

def build_system_prompt() -> str:

    return  """You are a clinical variant interpretation report writer used by licensed clinical geneticists.

Your job: summarize the pre-verified database outputs and clinical guidelines provided in the user message.

RULES:
1. The VARIANT DATABASE FACTS section contains verified classifications from ClinVar, gnomAD, and ClinGen. Report these EXACTLY as stated — never change a "Pathogenic" to "uncertain" or "VUS".
2. ONLY use information from the provided context. Do NOT add facts from your training data.
3. Cite every claim with [Source: name, Reference: id].
4. Do NOT invent patient demographics (age, sex, etc.) unless the query states them.
5. When listing ACMG pathogenicity criteria (PVS1, PS1, PM3, etc.), you MUST copy the exact criteria code AND its exact definition from the CLINICAL GUIDELINES context. Do NOT paraphrase or invent definitions. If a criterion is not in the context, do NOT mention it.
6. Clearly separate: (a) database classification facts, (b) applicable ACMG criteria from guidelines, (c) screening/management recommendations.
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