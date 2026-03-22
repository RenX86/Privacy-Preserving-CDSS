"""Tests for generation components: safe failure, citation enforcer, and citation extraction."""
import pytest
from app.pipeline.retrieval.reranker import RetrievedChunk
from app.pipeline.generation.guardrails import SAFE_FAILURE_MESSAGE
from app.pipeline.generation.citation_enforcer import fix_hallucinated_citations, extract_citations


def _make_chunk(text: str, source: str, reference: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, reference=reference)


def test_generate_answer_safe_failure_on_empty_chunks():
    """Empty verified_chunks must return SAFE_FAILURE_MESSAGE without calling Ollama."""
    from app.pipeline.generation.self_rag import generate_answer
    result = generate_answer("any query", [])
    assert result == SAFE_FAILURE_MESSAGE


def test_fix_citations_leaves_valid_citation_unchanged():
    """A valid citation whose source+reference both exist in chunks should not be modified."""
    chunks = [_make_chunk("PVS1 criteria text", source="ACMG_2015", reference="page 33")]
    answer = "The variant meets PVS1. [Source: ACMG_2015, Reference: page 33]"
    result = fix_hallucinated_citations(answer, chunks)
    assert "[Source: ACMG_2015, Reference: page 33]" in result


def test_fix_citations_remaps_hallucinated_source():
    """A citation with a source not in used_chunks should be remapped based on context keywords."""
    chunks = [_make_chunk("gnomAD AF=0.000001 population frequency allele frequency", source="gnomAD", reference="rs12345")]
    answer = "The gnomAD population frequency allele frequency shows the variant is rare. [Source: SomeRandomDB, Reference: unknown]"
    result = fix_hallucinated_citations(answer, chunks)
    assert "SomeRandomDB" not in result


def test_fix_citations_preserves_multiple_refs_per_source():
    """Multiple valid references for the same source should not be collapsed (H-2 fix)."""
    chunks = [
        _make_chunk("Page 3 content about ACMG", source="ACMG_2015", reference="page 3"),
        _make_chunk("Page 33 content about classification", source="ACMG_2015", reference="page 33"),
    ]
    answer = (
        "Claim one. [Source: ACMG_2015, Reference: page 3] "
        "Claim two. [Source: ACMG_2015, Reference: page 33]"
    )
    result = fix_hallucinated_citations(answer, chunks)
    assert "[Source: ACMG_2015, Reference: page 3]" in result
    assert "[Source: ACMG_2015, Reference: page 33]" in result


def test_extract_citations_deduplicates():
    """Duplicate citations in the answer text should appear only once in the output."""
    chunks = [_make_chunk("text", source="ACMG_2015", reference="page 3")]
    answer = (
        "Claim one. [Source: ACMG_2015, Reference: page 3] "
        "Claim two. [Source: ACMG_2015, Reference: page 3]"
    )
    citations = extract_citations(answer, chunks)
    assert len(citations) == 1


def test_extract_citations_empty_answer():
    """Empty answer should return an empty citation list."""
    result = extract_citations("", [])
    assert result == []


def test_strip_thinking_removes_think_blocks():
    """_strip_thinking() should remove <think>...</think> blocks (Qwen/thinking model safety)."""
    from app.pipeline.generation.self_rag import _strip_thinking
    raw = '<think>internal reasoning here</think>{"summary": {"text": "result"}}'
    cleaned = _strip_thinking(raw)
    assert "<think>" not in cleaned
    assert '{"summary"' in cleaned
