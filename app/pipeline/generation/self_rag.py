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
            "Citations supporting this claim. Each MUST be an EXACT copy of one line from the "
            "CITATION MANIFEST at the end of the prompt — format: '[Source: X, Reference: Y]'. "
            "DO NOT invent page numbers, accession IDs, or guideline names. "
            "DO NOT write 'page 363' or any page number unless that exact string appears in the manifest. "
            "If no manifest entry covers this claim, use the closest matching manifest line."
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
            "ACMG criteria that apply to this variant based on the CLINICAL GUIDELINES text. "
            "Each item: one full sentence explaining WHY the criterion applies to THIS variant. "
            "Example: 'PVS1 applies because rs879254116 introduces a frameshift in BRCA1, "
            "a gene where loss-of-function is a known disease mechanism per ACMG 2015.' "
            "Only include criteria the guideline text explicitly supports. No bare codes."
        )
    )
    screening_protocol: list[ClinicalClaim] = Field(
        description=(
            "Cancer screening protocol from NCCN chunks in CLINICAL GUIDELINES. "
            "Copy exact ages and procedure names from the retrieved text. "
            "Cite from the CITATION MANIFEST — use the Header_2 reference, not invented page numbers. "
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
    """
    Numbered manifest of every valid citation in the context.
    Numbered format makes it harder for the model to paraphrase or mutate.
    Placed at the END of the prompt (recency bias helps compliance).
    """
    seen = set()
    entries = []
    for chunk in verified_chunks:
        tag = f"[Source: {chunk.source}, Reference: {chunk.reference}]"
        if tag not in seen:
            seen.add(tag)
            entries.append(tag)

    lines = [
        "══════════════════════════════════════════════════════════",
        "  CITATION MANIFEST — ONLY these citations are permitted",
        "  Copy the EXACT string including brackets. No other citations allowed.",
        "══════════════════════════════════════════════════════════",
    ]
    for i, tag in enumerate(entries, 1):
        lines.append(f"  [{i}] {tag}")
    lines.append("══════════════════════════════════════════════════════════")
    return "\n".join(lines)


def _render_citations(raw_citations: list[str]) -> str:
    """
    Render citations list into inline text.
    Wraps bare 'Source: X, Reference: Y' strings that lost their brackets.
    Skips clearly malformed strings.
    """
    rendered = []
    for c in raw_citations:
        c = c.strip()
        if not c:
            continue
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
        f"RULES BEFORE YOU WRITE:\n"
        f"1. Read the ⚑ VARIANT DATABASE FACTS block first. Report clinical_significance exactly.\n"
        f"2. Do NOT say a variant is unclassified if it appears in the VARIANT DATABASE FACTS.\n"
        f"3. Every citation must be an exact copy of a line from the CITATION MANIFEST below.\n"
        f"4. Do NOT write page numbers (e.g. 'page 363') — they are not in the manifest.\n"
        f"5. ACMG rules: full sentences linking criterion to THIS variant. No bare codes.\n"
        f"6. Screening ages: copy exactly from retrieved NCCN text. Do not use memorised ages.\n\n"
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
                if cites and not text.endswith("]"):
                    text = f"{text} {cites}"
                lines.append(f"**Clinical Summary**\n{text}\n")

            if data.get("clingen_validity"):
                lines.append("**ClinGen Expert Panel Validity**")
                for item in data["clingen_validity"]:
                    text  = item.get("text", "").rstrip()
                    cites = _render_citations(item.get("citations", []))
                    if cites and not text.endswith("]"):
                        text = f"{text} {cites}"
                    lines.append(f"* {text}")
                lines.append("")

            if data.get("acmg_rules"):
                lines.append("**ACMG Pathogenicity Criteria**")
                for rule in data["acmg_rules"]:
                    text  = rule.get("text", "").rstrip()
                    cites = _render_citations(rule.get("citations", []))
                    if cites and not text.endswith("]"):
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
                    if cites and not text.endswith("]"):
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
