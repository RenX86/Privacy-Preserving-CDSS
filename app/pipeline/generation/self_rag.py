import re
import json
import logging
import ollama
from pydantic import BaseModel, Field
from app.config import settings

log = logging.getLogger("cdss.self_rag")


class ClinicalClaim(BaseModel):
    text: str = Field(
        description="The clinical text or rule. Must be a complete sentence, not just a code like 'PM1'."
    )
    citations: list[str] = Field(
        description=(
            "Citations for this claim. Each MUST be an exact copy of one bullet from the "
            "CITATION MANIFEST at the end of the prompt. "
            "Copy the FULL string exactly as written: '[Source: X, Reference: Y]'. "
            "Do NOT use numbers like '[1]' or '[3]'. "
            "Do NOT write '[Source: X]' without a Reference. "
            "Do NOT invent page numbers or source names. "
            "If no manifest entry covers this claim, use the closest matching bullet."
        )
    )


class ClinicalResponse(BaseModel):
    summary: ClinicalClaim = Field(
        description=(
            "Report the variant's clinical_significance EXACTLY as it appears in the "
            "⚑ VARIANT DATABASE FACTS block (e.g. 'Pathogenic', 'Likely pathogenic', 'VUS'). "
            "Do NOT say the variant is unclassified if it appears in the VARIANT DATABASE FACTS. "
            "Include gnomAD allele frequency if present. "
            "Do NOT add disease associations or cancer risk statements from general knowledge. "
            "Only state facts that appear verbatim in the context. "
            "Cite only from the CITATION MANIFEST."
        )
    )
    clingen_validity: list[ClinicalClaim] = Field(
        description=(
            "ClinGen expert panel summary from the ⚑ VARIANT DATABASE FACTS block. "
            "USE THIS EXACT FORMAT: '[GENE]: gene-disease validity: [True/False], "
            "actionability: [True/False], dosage sensitivity: [value or NA], "
            "last curated: [date]'. "
            "Copy each field value EXACTLY from the ClinGen record. Do NOT paraphrase or interpret. "
            "Do NOT infer variant-level actionability from gene-level ClinGen data. "
            "Leave as [] if no ClinGen entry is in the VARIANT DATABASE FACTS."
        )
    )
    screening_protocol: list[ClinicalClaim] = Field(
        description=(
            "Cancer screening protocol from NCCN chunks in CLINICAL GUIDELINES. "
            "Copy exact ages and procedure names from the retrieved text. "
            "Each item MUST be a specific screening action (e.g. 'Annual mammography starting at age 25'). "
            "Do NOT include introductory sentences like 'The following protocol is recommended'. "
            "CRITICAL: Before extracting any row from an NCCN table, verify the gene "
            "name in the leftmost column matches the query gene (e.g. BRCA1). "
            "Do NOT include protocols from other genes' rows (e.g. colonoscopy from "
            "MLH1/Lynch syndrome rows). "
            "Cite from the CITATION MANIFEST. "
            "Leave as [] if no NCCN screening text is in the context."
        )
    )



from app.pipeline.generation.guardrails import (
    build_system_prompt,
    build_context_block,
    SAFE_FAILURE_MESSAGE,
)
from app.pipeline.retrieval.reranker import RetrievedChunk


def _build_reference_manifest(verified_chunks: list[RetrievedChunk]) -> str:
    seen = set()
    entries = []
    for chunk in verified_chunks:
        tag = f"[Source: {chunk.source}, Reference: {chunk.reference}]"
        if tag not in seen:
            seen.add(tag)
            entries.append(tag)

    lines = [
        "══════════════════════════════════════════════════════════",
        "  CITATION MANIFEST — copy these EXACT strings into citations[]",
        "  Do NOT use numbers. Write the full [Source: X, Reference: Y] string.",
        "══════════════════════════════════════════════════════════",
    ]
    for tag in entries:
        lines.append(f"  • {tag}")
    lines.append("══════════════════════════════════════════════════════════")
    return "\n".join(lines)


def _render_citations(raw_citations: list[str]) -> str:
    rendered = []
    for c in raw_citations:
        c = c.strip()
        if not c:
            continue
        # Strip leading manifest number like "[3] " or "[12] "
        c = re.sub(r'^\[\d+\]\s*', '', c)
        if c.startswith("[Source:") and c.endswith("]"):
            rendered.append(c)
        elif c.lower().startswith("source:"):
            rendered.append(f"[{c}]")
        else:
            log.debug(f"Skipping malformed citation: {c!r}")
    return " ".join(rendered)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks if the model's thinking mode fires."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate_answer(query: str, verified_chunks: list[RetrievedChunk]) -> str:

    if not verified_chunks:
        log.warning("No verified chunks — returning safe failure message")
        return SAFE_FAILURE_MESSAGE

    context_block      = build_context_block(verified_chunks)
    reference_manifest = _build_reference_manifest(verified_chunks)

    # Manifest goes at the END — recency bias means the model reads it last
    # and it's fresh in context when it starts filling the JSON fields.
    user_message = (
        f"{context_block}\n\n"
        f"Answer this clinical query using ONLY the context above:\n"
        f"{query}\n\n"
        f"MANDATORY RULES:\n"
        f"1. Your summary MUST begin: 'Variant [rsID] in [gene] is classified as [significance].'\n"
        f"   The classification is confirmed in the ⚑ VARIANT DATABASE FACTS block above.\n"
        f"   NEVER write 'not described', 'not found', or 'if this were pathogenic'.\n"
        f"2. CITATIONS: copy the FULL '[Source: X, Reference: Y]' string from the manifest below.\n"
        f"   Do NOT use '[1]', '[3]', or any number as a citation. Numbers are not citations.\n"
        f"   Every bullet in screening_protocol MUST have at least one citation.\n"
        f"3. SCREENING TABLES: Before extracting any protocol row, verify the gene name\n"
        f"   in the leftmost column matches the query gene. Do NOT include screening from\n"
        f"   other genes' rows (e.g. colonoscopy from MLH1 rows for a BRCA1 query).\n"
        f"4. SUMMARY SCOPE: The summary field must ONLY report facts from the ⚑ VARIANT DATABASE\n"
        f"   FACTS block (ClinVar classification, gnomAD frequency). Do NOT include cancer risk\n"
        f"   statements or disease associations in the summary — those belong in screening_protocol.\n"
        f"5. CLINGEN SCOPE: ClinGen data is GENE-LEVEL (e.g. 'BRCA1 validity: True').\n"
        f"   Do NOT write 'variants like rs879254116 are clinically actionable' — that infers\n"
        f"   variant-level actionability from gene-level data. Report ClinGen facts as-is.\n\n"
        f"{reference_manifest}"
    )

    DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}
    db_count  = sum(1 for c in verified_chunks if c.source in DB_SOURCES)
    pdf_count = len(verified_chunks) - db_count

    log.info(f"Calling Ollama [{settings.LOCAL_LLM_MODEL}] ({db_count} DB + {pdf_count} PDF chunks)")
    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user",   "content": user_message}
            ],
            format=ClinicalResponse.model_json_schema()
        )
        raw_json = _strip_thinking(response.message.content)
        log.info(f"Draft JSON generated ({len(raw_json)} chars)")
        print(f"\n--- RAW JSON ---\n{raw_json}\n----------------\n")

        try:
            data = json.loads(raw_json)
            lines = []

            if data.get("summary"):
                s     = data["summary"]
                text  = s.get("text", "").rstrip()
                cites = _render_citations(s.get("citations", []))
                if cites and cites not in text:
                    text = f"{text} {cites}"
                lines.append(f"**Clinical Summary**\n{text}\n")

            if data.get("clingen_validity"):
                lines.append("**ClinGen Expert Panel Validity**")
                for item in data["clingen_validity"]:
                    text  = item.get("text", "").rstrip()
                    # Strip leading list prefixes the LLM may add inside the JSON text field
                    text = re.sub(r'^[-•*]\s+', '', text)
                    cites = _render_citations(item.get("citations", []))
                    if cites and cites not in text:
                        text = f"{text} {cites}"
                    lines.append(f"* {text}")
                lines.append("")


            if data.get("screening_protocol"):
                lines.append("**Cancer Screening Protocol**")
                for prot in data["screening_protocol"]:
                    text  = prot.get("text", "").rstrip()
                    # Strip leading list prefixes the LLM may add inside the JSON text field
                    text = re.sub(r'^[-•*]\s+', '', text)
                    cites = _render_citations(prot.get("citations", []))
                    if cites and cites not in text:
                        text = f"{text} {cites}"
                    lines.append(f"* {text}")

            answer = "\n".join(lines).strip()

        except json.JSONDecodeError:
            log.warning("Failed to decode JSON from Ollama. Falling back to raw response.")
            answer = raw_json

        print(f"\n--- PRE-ENFORCER ANSWER ---\n{answer}\n----------------\n")
        return answer

    except Exception as e:
        log.error(f"Ollama error during generation: {e}")
        return SAFE_FAILURE_MESSAGE

