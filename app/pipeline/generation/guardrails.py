from app.pipeline.retrieval.reranker import RetrievedChunk

SAFE_FAILURE_MESSAGE = (
    "Insufficient clinical data to safely provide an assessment. "
    "Please consult primary clinical databases or a qualified clinician."
)


def build_system_prompt() -> str:
    return """\
You are an expert Clinical Decision Support System (CDSS) for clinical geneticists and oncologists.
You synthesize retrieved database records and guideline chunks into a structured clinical summary.

══════════════════════════════════════════════════════
RULE 1 — READ THE VARIANT FACTS FIRST (non-negotiable)
══════════════════════════════════════════════════════
The prompt begins with a ⚑ VARIANT FACTS block. This block contains verified structured records
from ClinVar, gnomAD, and ClinGen. You MUST read and report these BEFORE writing anything else.
Do NOT say a variant is "not classified" or "not listed" if it appears in the ⚑ VARIANT FACTS block.
Report the clinical_significance field EXACTLY as written — do not rephrase, soften, or upgrade it.

══════════════════════════════════════════════════════
RULE 2 — CITATIONS MUST BE COPIED FROM THE MANIFEST
══════════════════════════════════════════════════════
The prompt ends with a numbered CITATION MANIFEST. Every [Source: X, Reference: Y] string you write
MUST be copied verbatim from that manifest. Do NOT invent page numbers, accession IDs, or guideline
titles. If a reference does not appear in the manifest, do not cite it.

══════════════════════════════════════════════════════
RULE 3 — EXACT AGES AND TERMINOLOGY FROM GUIDELINES
══════════════════════════════════════════════════════
When writing screening protocols from NCCN chunks, copy exact age ranges and procedure names from
the text. Do not substitute ages from memory (e.g. do not write "age 25-29" unless that exact range
appears in the retrieved NCCN chunk text).

══════════════════════════════════════════════════════
RULE 4 — IF NOT IN CONTEXT, SAY DATA UNAVAILABLE
══════════════════════════════════════════════════════
Do not fill gaps with general medical knowledge. If a specific data point (e.g. gnomAD frequency,
ClinGen curated date) is not in the context, write "Data unavailable in retrieved context."
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No verified clinical context available."

    db_sources   = {"Clinvar", "gnomAD", "ClinGen"}
    db_chunks    = [c for c in chunks if c.source in db_sources]
    guide_chunks = [c for c in chunks if c.source not in db_sources]

    lines = []

    # ── DB facts come FIRST and are visually unmissable ───────────────────────
    # Ministral/Qwen ignore the DB section when it comes after the long guideline
    # block. Placing it first — with a bold flag — forces the model to read it
    # before it processes any guideline text.
    if db_chunks:
        lines.append("⚑ ═══════════════════════════════════════════════════════════")
        lines.append("⚑  VARIANT DATABASE FACTS  —  READ THIS SECTION FIRST")
        lines.append("⚑  These are verified structured records. Report them exactly.")
        lines.append("⚑ ═══════════════════════════════════════════════════════════")
        lines.append("")
        for chunk in db_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            lines.append(chunk.text)
            lines.append("")
        lines.append("⚑ ═══ END OF VARIANT DATABASE FACTS ═══")
        lines.append("")

    if guide_chunks:
        lines.append("── CLINICAL GUIDELINES (use for criteria, rules, and protocols) ──")
        lines.append("")
        for chunk in guide_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            lines.append(chunk.text)
            lines.append("")

    lines.append("── END OF CONTEXT ──")
    return "\n".join(lines)