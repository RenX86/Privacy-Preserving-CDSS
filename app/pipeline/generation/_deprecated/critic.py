"""
DEPRECATED: Self-RAG Critic (Single-Pass LLM Auditor)
=====================================================
Removed from the pipeline on 2026-03-29.

Reason: The critic was demonstrably net-negative. In the latest run, it expanded
text from 3083->3257 chars, adding unverified disease-association claims while
failing to catch the summary hallucination (BRCA1 cancer risk from LLM memory)
and ClinGen inference error (gene-level -> variant-level actionability).

The citation safety net (revert if citations drop) only catches citation removal,
not content addition - which is the more dangerous failure mode.

This file is kept for reference in case a multi-pass critic with stricter
constraints is explored in the future.
"""

import re
import logging
import ollama
from app.config import settings
from app.pipeline.generation.guardrails import build_context_block
from app.pipeline.retrieval.reranker import RetrievedChunk


log = logging.getLogger("cdss.self_rag")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks if the model's thinking mode fires."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
