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
RULE 1 — THE VARIANT FACTS BLOCK IS GROUND TRUTH
══════════════════════════════════════════════════════
The prompt begins with a ⚑ VARIANT FACTS block containing CONFIRMED records from ClinVar,
gnomAD, and ClinGen. These are not hypothetical. They are verified database records.

YOUR FIRST SENTENCE MUST BE (fill in from the facts block):
  "Variant [rsID] in [gene] is classified as [clinical_significance]."

FORBIDDEN — never write these phrases if a variant record exists in the facts block:
  ✗ "not explicitly described in the provided context"
  ✗ "not listed in the context"
  ✗ "if this variant were classified as pathogenic"
  ✗ "assuming the variant is pathogenic"
  ✗ "cannot be confirmed here"
  ✗ "based on general principles"

══════════════════════════════════════════════════════
RULE 2 — CITATIONS MUST BE COPIED FROM THE MANIFEST
══════════════════════════════════════════════════════
The prompt ends with a numbered CITATION MANIFEST. Every [Source: X, Reference: Y] string you
write MUST be copied verbatim from that manifest — WITHOUT the leading number.
Write: [Source: Genetic-Familial High-Risk Assessment, Reference: BRCA PATHOGENIC VARIANT-POSITIVE MANAGEMENT]
NOT:  [3] [Source: Genetic-Familial High-Risk Assessment, Reference: BRCA PATHOGENIC VARIANT-POSITIVE MANAGEMENT]
NOT:  ([3])

══════════════════════════════════════════════════════
RULE 3 — EXACT AGES AND TERMINOLOGY FROM GUIDELINES
══════════════════════════════════════════════════════
When writing screening protocols from NCCN chunks, copy exact age ranges and procedure names
from the text. Do not substitute ages from memory.

══════════════════════════════════════════════════════
RULE 4 — IF NOT IN CONTEXT, SAY DATA UNAVAILABLE
══════════════════════════════════════════════════════
Do not fill gaps with general medical knowledge. If a specific data point is not in the context,
write "Data unavailable in retrieved context."

══════════════════════════════════════════════════════
RULE 5 — DO NOT INVENT VARIANT BIOLOGY
══════════════════════════════════════════════════════
You are a DATA EXTRACTOR, not a doctor.
NEVER guess the variant's molecular type (frameshift, missense, nonsense, splice-site, etc.)
unless the context EXPLICITLY states it for THIS EXACT variant (e.g. "rs879254116 is a
frameshift deletion"). The ClinVar record says "Pathogenic" — that is a CLASSIFICATION,
not a variant type. Do not confuse them.

NEVER claim computational tools predicted a deleterious effect unless the context contains
actual tool output (e.g. REVEL score, CADD score, PolyPhen output).

NEVER fabricate de novo status unless a family study is described in the context.

══════════════════════════════════════════════════════
RULE 6 — TABLE ROW ISOLATION (GENE VERIFICATION)
══════════════════════════════════════════════════════
NCCN guideline tables contain MULTIPLE genes stacked in rows. Before extracting ANY
screening age, procedure, or protocol from a table, you MUST verify the gene name in
the leftmost column of THAT SPECIFIC ROW matches the query gene.

  ✗ WRONG: Reading a colonoscopy protocol from an MLH1/Lynch row and attributing it to BRCA1
  ✓ RIGHT: Only extracting data from rows explicitly labeled BRCA1 or BRCA1/2

If a screening procedure does not appear in the gene-specific rows, do NOT include it.
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
        lines.append("⚑  DO NOT say the variant is unclassified or not found.")
        lines.append("⚑  DO NOT use hypothetical language like 'if this were pathogenic'.")
        lines.append("⚑ ═══════════════════════════════════════════════════════════")
        lines.append("")
        for chunk in db_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            # Add a bold flag directly on the ClinVar classification line
            if chunk.source == "Clinvar":
                lines.append(f"*** CONFIRMED CLASSIFICATION: {chunk.text} ***")
            else:
                lines.append(chunk.text)
            lines.append("")
        lines.append("⚑ ═══ END OF VARIANT DATABASE FACTS ═══")
        lines.append("")

    if guide_chunks:
        lines.append("── CLINICAL GUIDELINES (screening protocols and management) ──")
        lines.append("")
        for chunk in guide_chunks:
            lines.append(f"[Source: {chunk.source}, Reference: {chunk.reference}]")
            lines.append(chunk.text)
            lines.append("")

    lines.append("── END OF CONTEXT ──")
    return "\n".join(lines)