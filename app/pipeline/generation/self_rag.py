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


def _enforce_gnomad_accuracy(answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Programmatic guard: If gnomAD returned 'not found' or no gnomAD chunk
    exists at all, strip any hallucinated frequency values from the answer.
    The LLM can ignore prompt rules. It cannot ignore re.sub().
    """
    gnomad_chunks = [c for c in verified_chunks if c.source == "gnomAD"]

    # Pattern matches phrases like:
    #   "with a gnomAD frequency of 0.00000817 (1/122,362)."
    #   "gnomAD allele frequency of **0.000008224**"
    #   "frequency of 0.0001"
    freq_pattern = re.compile(
        r'(?:with\s+)?(?:a\s+)?(?:gnomAD\s+)?(?:allele\s+)?frequency\s+(?:of\s+)?\**[\d.eE\-]+\**(?:\s*\([^)]+\))?[.,]?\s*',
        re.IGNORECASE
    )
    gnomad_citation_pattern = re.compile(
        r'\[Source:\s*gnomAD[^\]]*\]\s*',
        re.IGNORECASE
    )

    if not gnomad_chunks:
        # No gnomAD data at all — remove any AF mentions and gnomAD citations
        answer = freq_pattern.sub('', answer)
        answer = gnomad_citation_pattern.sub('', answer)
        log.info("[Guardrail] No gnomAD chunk — stripped any hallucinated frequency")
        return answer

    gnomad_text = gnomad_chunks[0].text.lower()

    if "not found" in gnomad_text or "absent" in gnomad_text:
        # gnomAD explicitly said not found — remove invented frequencies AND citations
        answer = freq_pattern.sub('', answer)
        answer = gnomad_citation_pattern.sub('', answer)
        log.info("[Guardrail] gnomAD='not found' — stripped hallucinated frequency + citation")

    return answer


def _enforce_no_fabricated_acmg(answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Programmatic guard: Strip ACMG evidence code lists (PS4, PM2, PP3, etc.)
    that the LLM fabricates. ClinVar stores 'Pathogenic' as a classification,
    NOT the underlying evidence codes. Unless the chunks explicitly contain
    these codes, they are hallucinated.
    """
    # Pattern matches parenthesized lists of ACMG codes like:
    #   "(PS4, PM2, PP3, PP4, PS3_strong, PS3_moderate)"
    #   "(PVS1, PM2)"
    acmg_pattern = re.compile(
        r'\s*\((?:P[SVP]\d|P[MV]S?\d|B[SP]\d|BA1)[\w_,\s]*\)',
        re.IGNORECASE
    )

    all_chunk_text = " ".join(c.text for c in verified_chunks)

    matches = list(acmg_pattern.finditer(answer))
    for match in reversed(matches):
        acmg_text = match.group(0).strip()
        # Check if these codes appear in any chunk
        if acmg_text in all_chunk_text:
            continue  # Grounded in evidence
        answer = answer[:match.start()] + answer[match.end():]
        log.info(f"[Guardrail] Stripped fabricated ACMG codes: '{acmg_text}'")

    return answer


def _cleanup_orphaned_text(answer: str) -> str:
    """
    Clean up grammatical fragments left behind after guardrail stripping.
    E.g. 'classified as **Pathogenic** and ClinVar classification of **Pathogenic**'
    becomes 'classified as **Pathogenic**'.
    """
    # Remove duplicate classification statements created by frequency stripping
    answer = re.sub(
        r'(is classified as \**\w[\w/\s]*\**)\.?\s*(?:and|with)?\s*(?:ClinVar\s+)?classification\s+(?:of\s+)?\**\w[\w/\s]*\**',
        r'\1',
        answer, flags=re.IGNORECASE
    )
    # Remove orphaned "and" at start/end of phrases
    answer = re.sub(r'\band\s+\.', '.', answer)
    answer = re.sub(r'\band\s+ClinGen', 'ClinGen', answer)
    # Remove double spaces left by stripping
    answer = re.sub(r'  +', ' ', answer)
    # Remove empty bold markers
    answer = re.sub(r'\*\*\s*\*\*', '', answer)
    return answer.strip()


def _enforce_no_fabricated_biology(answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Programmatic guard: Strip variant biology claims (frameshift, missense, etc.)
    that do not appear in any verified chunk. ClinVar says 'Pathogenic' — that is
    a CLASSIFICATION, not a variant type. The LLM must not confuse them.
    """
    # Molecular type terms the LLM might fabricate
    biology_terms = [
        r'\bframeshift\b', r'\bmissense\b', r'\bnonsense\b',
        r'\bsplice[- ]?site\b', r'\btruncating\b', r'\bin[- ]?frame\s+deletion\b',
        r'\bstop[- ]?gain\b', r'\bloss[- ]?of[- ]?function\b',
        r'\bgain[- ]?of[- ]?function\b', r'\bde\s+novo\b',
        r'\bdominant[- ]?negative\b', r'\bhaploinsufficiency\b',
    ]

    # Build a single text blob from all chunks for searching
    all_chunk_text = " ".join(c.text.lower() for c in verified_chunks)

    for term_pattern in biology_terms:
        # Check if this term appears in the answer
        matches = list(re.finditer(term_pattern, answer, re.IGNORECASE))
        if not matches:
            continue

        # Check if this term appears in ANY verified chunk
        if re.search(term_pattern, all_chunk_text, re.IGNORECASE):
            continue  # Term is grounded in evidence — keep it

        # Term is fabricated — remove the sentence containing it
        for match in reversed(matches):
            # Find the sentence boundaries around the match
            sent_start = answer.rfind('.', 0, match.start())
            sent_start = sent_start + 1 if sent_start != -1 else max(0, answer.rfind('\n', 0, match.start()))
            sent_end = answer.find('.', match.end())
            sent_end = sent_end + 1 if sent_end != -1 else len(answer)

            removed = answer[sent_start:sent_end].strip()
            answer = answer[:sent_start] + answer[sent_end:]
            log.info(f"[Guardrail] Stripped fabricated biology: '{removed}'")

    return answer.strip()


def _enforce_clinvar_classification(answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Programmatic guard: Verify the LLM reported the ClinVar classification
    exactly as it appears in the ClinVar chunk. If the classification was
    altered (e.g. 'Pathogenic' → 'Likely Pathogenic'), fix it.
    """
    clinvar_chunks = [c for c in verified_chunks if c.source == "Clinvar"]
    if not clinvar_chunks:
        return answer

    # Extract the actual classification from the ClinVar chunk text
    # ClinVar chunks contain "clinical_significance: Pathogenic" or similar
    clinvar_text = clinvar_chunks[0].text
    sig_match = re.search(
        r'clinical_significance:\s*([^|\n,]+)',
        clinvar_text, re.IGNORECASE
    )
    if not sig_match:
        return answer

    actual_classification = sig_match.group(1).strip()

    # Known classification values to check for drift
    classification_variants = [
        'Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic',
        'Uncertain significance', 'Likely benign', 'Benign',
        'Conflicting classifications of pathogenicity',
    ]

    for wrong_class in classification_variants:
        if wrong_class.lower() == actual_classification.lower():
            continue  # This IS the correct classification
        # Check if the LLM used this wrong classification
        pattern = re.compile(
            r'is\s+classified\s+as\s+\**' + re.escape(wrong_class) + r'\**',
            re.IGNORECASE
        )
        if pattern.search(answer):
            # Replace with the correct classification
            answer = pattern.sub(
                f'is classified as **{actual_classification}**',
                answer
            )
            log.info(
                f"[Guardrail] Fixed ClinVar drift: '{wrong_class}' → '{actual_classification}'"
            )

    return answer


def _enforce_no_fabricated_predictions(answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """
    Programmatic guard: Strip computational prediction claims (REVEL, CADD,
    PolyPhen, SIFT, etc.) that do not appear in any verified chunk.
    LLMs love to invent plausible-looking tool scores.
    """
    prediction_patterns = [
        r'\bREVEL\s+(?:score\s+(?:of\s+)?)?\d+\.\d+',
        r'\bCADD\s+(?:score\s+(?:of\s+)?)?\d+\.?\d*',
        r'\bPolyPhen[\s-]*2?\s+(?:score\s+(?:of\s+)?)?\d+\.\d+',
        r'\bSIFT\s+(?:score\s+(?:of\s+)?)?\d+\.\d+',
        r'\bAlphaMissense\s+(?:score\s+(?:of\s+)?)?\d+\.\d+',
        r'\bSpliceAI\s+(?:score\s+(?:of\s+)?)?\d+\.\d+',
        r'\bpredicted\s+(?:to\s+be\s+)?deleterious\b',
        r'\bcomputationally\s+predicted\b',
        r'\bin\s+silico\s+(?:tools?\s+)?predict',
    ]

    all_chunk_text = " ".join(c.text.lower() for c in verified_chunks)

    for pattern in prediction_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            # Only strip if the term is NOT in any chunk
            if not re.search(pattern, all_chunk_text, re.IGNORECASE):
                answer = re.sub(
                    # Remove the full sentence containing the fabricated prediction
                    r'[^.\n]*' + pattern + r'[^.\n]*[.]?\s*',
                    '', answer, flags=re.IGNORECASE
                )
                log.info(f"[Guardrail] Stripped fabricated prediction matching: {pattern}")

    return answer


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

            # ── Programmatic Hallucination Guards ─────────────────
            # These run AFTER generation. The LLM can ignore prompt
            # rules, but it cannot bypass re.sub().
            answer = _enforce_gnomad_accuracy(answer, verified_chunks)
            answer = _enforce_no_fabricated_biology(answer, verified_chunks)
            answer = _enforce_clinvar_classification(answer, verified_chunks)
            answer = _enforce_no_fabricated_predictions(answer, verified_chunks)
            answer = _enforce_no_fabricated_acmg(answer, verified_chunks)
            answer = _cleanup_orphaned_text(answer)

        except json.JSONDecodeError:
            log.warning("Failed to decode JSON from Ollama. Falling back to raw response.")
            answer = raw_json

        print(f"\n--- PRE-ENFORCER ANSWER ---\n{answer}\n----------------\n")
        return answer

    except Exception as e:
        log.error(f"Ollama error during generation: {e}")
        return SAFE_FAILURE_MESSAGE

