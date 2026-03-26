import re
import json
import logging
import ollama
from pydantic import BaseModel, Field
from app.config import settings
from app.pipeline.generation.citation_enforcer import fix_hallucinated_citations

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
            "Include gnomAD allele frequency if present. Cite only from the CITATION MANIFEST."
        )
    )
    clingen_validity: list[ClinicalClaim] = Field(
        description=(
            "ClinGen expert panel summary from the ⚑ VARIANT DATABASE FACTS block. "
            "Report: gene-disease validity (True/False), actionability, dosage sensitivity, "
            "and date_last_curated. Leave as [] if no ClinGen entry is in the VARIANT DATABASE FACTS."
        )
    )
    acmg_rules: list[ClinicalClaim] = Field(
        description=(
            "ACMG criteria DEFINITIONS from the guideline text in context. "
            "For each criterion, report: (1) the criterion code, (2) its DEFINITION "
            "copied from the ACMG guideline text, and (3) which database fact is relevant. "
            "Example: 'PM2 is defined as: absent from controls in population databases "
            "(ACMG 2015 Table 3). The gnomAD record shows this variant was not found, "
            "which is relevant to this criterion.' "
            "Do NOT conclude whether criteria apply — that requires clinical review. "
            "Do NOT invent the variant type (frameshift, missense) unless the context "
            "explicitly states it for this exact variant. "
            "Do NOT claim computational predictions exist unless scores are in context."
        )
    )
    screening_protocol: list[ClinicalClaim] = Field(
        description=(
            "Cancer screening protocol from NCCN chunks in CLINICAL GUIDELINES. "
            "Copy exact ages and procedure names from the retrieved text. "
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
        f"   Every bullet in acmg_rules and screening_protocol MUST have at least one citation.\n"
        f"3. ACMG rules: report the DEFINITION of each criterion from the guideline text.\n"
        f"   Then state which database fact is RELEVANT — but do NOT conclude it applies.\n"
        f"   Do NOT invent the variant's molecular type (frameshift, missense, splice).\n"
        f"   Do NOT claim computational predictions exist unless actual scores are in context.\n"
        f"   Do NOT confuse criteria codes: PM2 = absent from controls, PS2 = de novo.\n"
        f"4. SCREENING TABLES: Before extracting any protocol row, verify the gene name\n"
        f"   in the leftmost column matches the query gene. Do NOT include screening from\n"
        f"   other genes' rows (e.g. colonoscopy from MLH1 rows for a BRCA1 query).\n\n"
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
                    cites = _render_citations(item.get("citations", []))
                    if cites and cites not in text:
                        text = f"{text} {cites}"
                    lines.append(f"* {text}")
                lines.append("")

            if data.get("acmg_rules"):
                lines.append("**ACMG Pathogenicity Criteria**")
                for rule in data["acmg_rules"]:
                    text  = rule.get("text", "").rstrip()
                    cites = _render_citations(rule.get("citations", []))
                    if cites and cites not in text:
                        text = f"{text} {cites}"
                    # Skip bare criterion codes with no explanation
                    if len(text.split()) <= 2:
                        log.debug(f"Skipping bare ACMG code: {text!r}")
                        continue
                    lines.append(f"* {text}")
                lines.append("")

            if data.get("screening_protocol"):
                lines.append("**Cancer Screening Protocol**")
                for prot in data["screening_protocol"]:
                    text  = prot.get("text", "").rstrip()
                    cites = _render_citations(prot.get("citations", []))
                    if cites and cites not in text:
                        text = f"{text} {cites}"
                    lines.append(f"* {text}")

            answer = "\n".join(lines).strip()

        except json.JSONDecodeError:
            log.warning("Failed to decode JSON from Ollama. Falling back to raw response.")
            answer = raw_json

        print(f"\n--- PRE-ENFORCER ANSWER ---\n{answer}\n----------------\n")
        answer = fix_hallucinated_citations(answer, verified_chunks)
        print(f"\n--- POST-ENFORCER ANSWER ---\n{answer}\n----------------\n")
        return answer

    except Exception as e:
        log.error(f"Ollama error during generation: {e}")
        return SAFE_FAILURE_MESSAGE


def self_rag_critic(query: str, draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """Single-pass LLM critic. Reverts to draft if citations are dropped."""
    log.info("Running LLM-based critic...")

    if not verified_chunks:
        return draft_answer

    context_block = build_context_block(verified_chunks)

    critic_prompt = (
        f"You are a strict Clinical Auditor verifying a generated clinical summary.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"DRAFT SUMMARY TO AUDIT:\n{draft_answer}\n\n"
        f"AUDIT RULES:\n"
        f"1. WRONG GENE DATA: Did the draft assign the wrong age, surgery, or protocol for this gene?\n"
        f"2. HALLUCINATIONS: Remove any claim not supported by the context.\n"
        f"3. CITATIONS: Every sentence must end with [Source: X, Reference: Y] from the context.\n\n"
        f"INSTRUCTIONS:\n"
        f"- Rewrite only what is factually wrong. Do not shorten unnecessarily.\n"
        f"- Preserve ALL [Source: X, Reference: Y] citation tags. Do NOT remove them.\n"
        f"- Output ONLY the corrected clinical summary. No audit notes, no preamble."
    )

    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical auditor. Output only the corrected summary text."},
                {"role": "user",   "content": critic_prompt}
            ]
        )
        final_answer = _strip_thinking(response.message.content)
        log.info(f"LLM Critic finished. Length: {len(draft_answer)} -> {len(final_answer)}")

        # Safety net: revert if the critic dropped citations
        draft_cite_count = draft_answer.count("[Source:")
        final_cite_count = final_answer.count("[Source:")
        if draft_cite_count > 0 and final_cite_count < draft_cite_count:
            log.warning(
                f"Critic dropped citations ({draft_cite_count} -> {final_cite_count}). Reverting to draft."
            )
            return draft_answer

        return final_answer

    except Exception as e:
        log.error(f"Ollama error during critic: {e}")
        return draft_answer
