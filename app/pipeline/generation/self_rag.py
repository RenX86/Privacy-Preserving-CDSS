import re
import json
import logging
import ollama
from pydantic import BaseModel, Field
from app.config import settings
from app.pipeline.generation.citation_enforcer import fix_hallucinated_citations

log = logging.getLogger("cdss.self_rag")

class ClinicalClaim(BaseModel):
    text: str = Field(description="The clinical text or rule.")
    citations: list[str] = Field(description="List of exact metadata citations used to support this text, formatted as '[Source: X, Reference: Y]'. MUST NOT BE EMPTY.")

class ClinicalResponse(BaseModel):
    summary: ClinicalClaim = Field(description="The main clinical summary answering the query, including variant classification from ClinVar and gnomAD population frequency.")
    clingen_validity: list[ClinicalClaim] = Field(description="ClinGen expert panel gene-disease validity. If ClinGen data is in context, summarise gene validity, actionability, dosage, and last curated date. Leave empty if no ClinGen data in context.")
    acmg_rules: list[ClinicalClaim] = Field(description="ACMG criteria that SPECIFICALLY apply to this variant based on the ACMG guideline sections in context. Only cite criteria explicitly supported by the retrieved text.")
    screening_protocol: list[ClinicalClaim] = Field(description="Cancer screening and surveillance protocol from NCCN for this specific gene, including exact ages, screening types (MRI, mammography), and risk-reducing surgery timing.")
from app.pipeline.generation.guardrails import (
    build_system_prompt,
    build_context_block,
    SAFE_FAILURE_MESSAGE,
)
from app.pipeline.retrieval.reranker import RetrievedChunk

def generate_answer(query: str, verified_chunks: list[RetrievedChunk]) -> str:

    if not verified_chunks:
        log.warning("No verified chunks — returning safe failure message")
        return SAFE_FAILURE_MESSAGE

    # ── Build the context block using the shared function ─────────────────────
    context_block = build_context_block(verified_chunks)

    # ── Final user message: structured context → question ────────────────────
    user_message = (
        f"{context_block}\n\n"
        f"Using ONLY the database facts and guidelines above, write a clinical summary answering:\n"
        f"{query}\n\n"
        f"CRITICAL RULES:\n"
        f"1. The VARIANT DATABASE FACTS above are verified. Report their classifications EXACTLY as stated — do NOT change Pathogenic to uncertain or VUS.\n"
        f"2. Every claim MUST include the exact `[Source: X, Reference: Y]` strings in its citations array.\n"
        f"3. Do not add information not present in the context above.\n"
        f"4. Do not invent patient demographics (age, sex) unless stated in the query.\n"
        f"5. Do not suggest external resources."
    )

    DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}
    db_count = sum(1 for c in verified_chunks if c.source in DB_SOURCES)
    pdf_count = len(verified_chunks) - db_count

    log.info(f"Calling Ollama [{settings.LOCAL_LLM_MODEL}] "
             f"({db_count} DB + {pdf_count} PDF chunks)")
    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user",   "content": user_message}
            ],
            format=ClinicalResponse.model_json_schema()
        )
        raw_json = response.message.content
        log.info(f"Draft JSON generated ({len(raw_json)} chars)")
        print(f"\n--- RAW JSON ---\n{raw_json}\n----------------\n")
        try:
            data = json.loads(raw_json)
            lines = []
            
            if "summary" in data and data["summary"]:
                summary_data = data["summary"]
                text = summary_data.get("text", "")
                cites = " ".join(summary_data.get("citations", []))
                lines.append(f"**Clinical Summary**\n{text} {cites}\n")

            # ── ClinGen section — new field ───────────────────────────────────
            if "clingen_validity" in data and data["clingen_validity"]:
                lines.append("**ClinGen Expert Panel Validity**")
                for item in data["clingen_validity"]:
                    text = item.get("text", "")
                    cites = " ".join(item.get("citations", []))
                    lines.append(f"* {text} {cites}")
                lines.append("")

            if "acmg_rules" in data and data["acmg_rules"]:
                lines.append("**ACMG Pathogenicity Criteria**")
                for rule in data["acmg_rules"]:
                    text = rule.get("text", "")
                    cites = " ".join(rule.get("citations", []))
                    lines.append(f"* {text} {cites}")
                lines.append("")

            if "screening_protocol" in data and data["screening_protocol"]:
                lines.append("**Cancer Screening Protocol**")
                for prot in data["screening_protocol"]:
                    text = prot.get("text", "")
                    cites = " ".join(prot.get("citations", []))
                    lines.append(f"* {text} {cites}")
            
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
    """LLM-based Critic to verify clinical accuracy, especially ages and exact genes."""
    log.info("Running LLM-based critic...")
    
    if not verified_chunks:
        return draft_answer

    context_block = build_context_block(verified_chunks)
    
    critic_prompt = f"""You are a strict Clinical Auditor. Your job is to verify a generated clinical summary against the retrieved medical context.
    
CONTEXT:
{context_block}

DRAFT SUMMARY TO AUDIT:
{draft_answer}

CRITICAL AUDIT RULES:
1. FATAL ERRORS: Did the draft assign the wrong age or the wrong surgery for a gene? (e.g., Draft says BRCA1 RRSO is 45-50, but context row for BRCA1 says 35-40).
2. HALLUCINATIONS: Are there claims in the draft not supported by the context?
3. CITATIONS: Does every sentence end with a proper metadata citation [Source: X, Reference: Y] found exactly in the context?

INSTRUCTIONS:
You MUST rewrite the draft to accurately reflect the text. Strip out any hallucinated protocols or fake citations not present in the CONTEXT.
CRITICAL MANDATE: You MUST ensure EVERY sentence ends with an inline citation structured EXACTLY as `[Source: X, Reference: Y]`. DO NOT strip these out!
If the draft is already perfectly accurate, return the exact original draft but ENSURE it has the citations.
CRITICAL: DO NOT add any explanatory text. DO NOT output your audit checklist. Output ONLY the raw, safe clinical summary.
"""
    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical auditor. Output only the final corrected text."},
                {"role": "user", "content": critic_prompt}
            ]
        )
        final_answer = response.message.content.strip()
        log.info(f"LLM Critic finished. Length changed from {len(draft_answer)} to {len(final_answer)}.")
        return final_answer
    except Exception as e:
        log.error(f"Ollama error during critic: {e}")
        return draft_answer
