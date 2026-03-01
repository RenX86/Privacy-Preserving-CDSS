import re
import logging
import ollama
from app.config import settings

log = logging.getLogger("cdss.self_rag")
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

    system_prompt = build_system_prompt()
    context_block = build_context_block(verified_chunks)

    user_message = f"""{context_block}

CLINICAL QUERY: {query}

INSTRUCTIONS — follow these steps in order:
1. Look at VARIANT DATABASE FACTS: state the specific variant's classification, gene, and associated conditions directly from that section.
2. Look at CLINICAL GUIDELINES: identify any criteria codes (PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4) or classification rules mentioned that are relevant to this variant type.
3. Write a concise clinical answer that directly addresses the query.
4. Cite every claim: [Source: name, Reference: id].
5. If a specific criterion is not mentioned in the CLINICAL GUIDELINES section, say "criteria details not available in retrieved guidelines" — do NOT invent criteria names.
6. Do NOT summarize the entire guideline document. Answer ONLY what was asked."""

    log.info(f"Calling Ollama [{settings.LOCAL_LLM_MODEL}] to generate answer ({len(verified_chunks)} verified chunks)")
    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        answer = response.message.content
        log.info(f"Draft answer generated ({len(answer)} chars)")
        return answer

    except Exception as e:
        log.error(f"Ollama error during generation: {e}")
        return SAFE_FAILURE_MESSAGE

def _extract_key_terms(text: str) -> set[str]:
    """
    Extract clinically meaningful terms from a sentence.
    These are the terms we check against the source chunks.
    """
    terms = set()

    # rsIDs: rs123456789
    terms.update(re.findall(r'\brs\d+\b', text, re.IGNORECASE))

    # Gene names: BRCA1, TP53, MLH1 (2-6 uppercase letters + optional digit)
    terms.update(re.findall(r'\b[A-Z]{2,6}\d?\b', text))

    # ACMG criteria codes: PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-7
    terms.update(re.findall(r'\b(?:PVS|PS|PM|PP|BA|BS|BP)\d\b', text))

    # Key clinical classification words
    clinical_words = {
        "pathogenic", "likely pathogenic", "benign", "likely benign",
        "uncertain significance", "vus", "pathogenicity", "classification",
        "loss of function", "frameshift", "nonsense", "splice", "missense",
        "hereditary", "breast", "ovarian", "cancer", "syndrome"
    }
    lower = text.lower()
    for word in clinical_words:
        if word in lower:
            terms.add(word)

    # Numbers / percentages (allele frequency, etc.)
    terms.update(re.findall(r'\b\d+\.?\d*%?\b', text))

    return terms


def programmatic_critic(draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Deterministic sentence-level critic — no LLM needed.

    For each sentence in the draft:
    - Extract its clinical key terms (rsIDs, gene names, ACMG codes, etc.)
    - Check if any of those terms appear in the combined chunk corpus
    - If yes → keep the sentence
    - If no key terms at all → keep it (likely a connecting/intro sentence)
    - If key terms exist but none match the corpus → drop the sentence

    This replaces the LLM critic which was rewriting answers from training data.
    """
    # Build a single searchable corpus from all chunk texts
    corpus = " ".join(c.text.lower() for c in verified_chunks)

    sentences = re.split(r'(?<=[.!?])\s+', draft_answer.strip())
    kept = []
    dropped = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        key_terms = _extract_key_terms(sentence)

        if not key_terms:
            # No clinical terms — likely a transition sentence, keep it
            kept.append(sentence)
            continue

        # Check if at least one key term appears in the corpus
        matched = any(term.lower() in corpus for term in key_terms)
        if matched:
            kept.append(sentence)
        else:
            dropped.append(sentence)

    if dropped:
        log.info(f"Programmatic critic: kept {len(kept)}, dropped {len(dropped)} sentences")
        for d in dropped:
            log.info(f"  DROPPED: {d[:80]}")
    else:
        log.info("Programmatic critic: all sentences supported — no changes")

    return " ".join(kept) if kept else draft_answer


def self_rag_critic(query: str, draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """Entry point for the critic — now uses the programmatic critic."""
    log.info("Running programmatic critic (no LLM)...")
    return programmatic_critic(draft_answer, verified_chunks)

