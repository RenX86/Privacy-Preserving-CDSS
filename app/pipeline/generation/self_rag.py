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

    # ── Build the context block using the shared function ─────────────────────
    context_block = build_context_block(verified_chunks)

    # ── Final user message: structured context → question ────────────────────
    user_message = (
        f"{context_block}\n\n"
        f"Using ONLY the database facts and guidelines above, write a clinical summary answering:\n"
        f"{query}\n\n"
        f"CRITICAL RULES:\n"
        f"1. The VARIANT DATABASE FACTS above are verified. Report their classifications EXACTLY as stated — do NOT change Pathogenic to uncertain or VUS.\n"
        f"2. Cite every fact with [Source: name, Reference: id].\n"
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
        "synonymous", "deletion", "insertion", "duplication", "null variant",
        "hereditary", "breast", "ovarian", "cancer", "syndrome",
        "screening", "surveillance", "mastectomy", "prophylactic",
    }
    lower = text.lower()
    for word in clinical_words:
        if word in lower:
            terms.add(word)

    # Numbers / percentages (allele frequency, etc.)
    terms.update(re.findall(r'\b\d+\.?\d*%?\b', text))

    return terms


# ── Classification terms grouped by meaning ──────────────────────────────────
_PATHOGENIC_TERMS   = {"pathogenic", "likely pathogenic"}
_BENIGN_TERMS       = {"benign", "likely benign"}
_UNCERTAIN_TERMS    = {"uncertain significance", "vus", "uncertain"}

def _get_db_classifications(db_chunks: list[RetrievedChunk]) -> dict[str, str]:
    """
    Extract the actual classification stated for each rsID in the DB chunks.
    Returns e.g. {"rs879254116": "pathogenic"}
    """
    classifications = {}
    for chunk in db_chunks:
        rsids = re.findall(r'\brs\d+\b', chunk.text, re.IGNORECASE)
        text_lower = chunk.text.lower()
        for rsid in rsids:
            if "pathogenic" in text_lower:
                classifications[rsid.lower()] = "pathogenic"
            elif "benign" in text_lower:
                classifications[rsid.lower()] = "benign"
            elif "uncertain" in text_lower or "vus" in text_lower:
                classifications[rsid.lower()] = "uncertain"
    return classifications


def programmatic_critic(draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Deterministic sentence-level critic — no LLM needed.

    Two-pass check:
    1. Factual contradiction check: If a sentence says "uncertain/VUS" but the DB
       says "Pathogenic", drop or correct it. This catches the main hallucination.
    2. Key-term grounding check: For each sentence, check that at least one
       clinical key term appears in the combined chunk corpus. If key terms exist
       but none match → drop the sentence (likely hallucinated).
    """
    DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}
    db_chunks = [c for c in verified_chunks if c.source in DB_SOURCES]
    db_classifications = _get_db_classifications(db_chunks)

    # Build a single searchable corpus from all chunk texts
    corpus = " ".join(c.text.lower() for c in verified_chunks)

    sentences = re.split(r'(?<=[.!?])\s+', draft_answer.strip())
    kept = []
    dropped = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_lower = sentence.lower()

        # ── Pass 1: Factual contradiction check ──────────────────────────────
        # If the sentence mentions a specific rsID that we have DB data for,
        # check whether it contradicts the known classification
        contradiction = False
        for rsid, db_class in db_classifications.items():
            if rsid in sentence_lower:
                # DB says pathogenic, but sentence says uncertain/VUS
                if db_class in ("pathogenic", "likely pathogenic"):
                    if any(t in sentence_lower for t in ("uncertain", "vus", "not specified", "not found", "unable to find")):
                        log.info(f"  DROP (contradicts DB: {rsid} is {db_class} but sentence says uncertain): {sentence[:80]}")
                        dropped.append(sentence)
                        contradiction = True
                        break
                # DB says benign, but sentence says pathogenic
                elif db_class in ("benign", "likely benign"):
                    if "pathogenic" in sentence_lower and "benign" not in sentence_lower:
                        log.info(f"  DROP (contradicts DB: {rsid} is {db_class} but sentence says pathogenic): {sentence[:80]}")
                        dropped.append(sentence)
                        contradiction = True
                        break

        if contradiction:
            continue

        # ── Pass 1b: Drop hallucinated patient demographics ──────────────────
        # If the sentence invents patient age/sex not present in any chunk
        age_pattern = re.search(r'\b\d{1,3}-year-old\b', sentence_lower)
        if age_pattern and age_pattern.group() not in corpus:
            log.info(f"  DROP (hallucinated demographics): {sentence[:80]}")
            dropped.append(sentence)
            continue

        # ── Pass 2: Key-term grounding check ─────────────────────────────────
        key_terms = _extract_key_terms(sentence)

        if not key_terms:
            kept.append(sentence)
            log.info(f"  KEEP (no clinical terms — transition sentence): {sentence[:80]}")
            continue

        # Check if at least one key term appears in the corpus
        matched_terms = [t for t in key_terms if t.lower() in corpus]
        if matched_terms:
            kept.append(sentence)
            log.info(f"  KEEP (terms matched: {list(matched_terms)[:3]}): {sentence[:80]}")
        else:
            dropped.append(sentence)
            log.info(f"  DROP (no corpus match for terms {list(key_terms)[:5]}): {sentence[:80]}")

    if dropped:
        log.info(f"Critic result: kept {len(kept)} / dropped {len(dropped)} sentences")
    else:
        log.info(f"Critic result: all {len(kept)} sentences supported — no changes")

    return " ".join(kept) if kept else draft_answer


def self_rag_critic(query: str, draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """Entry point for the critic — now uses the programmatic critic."""
    log.info("Running programmatic critic (no LLM)...")
    return programmatic_critic(draft_answer, verified_chunks)
