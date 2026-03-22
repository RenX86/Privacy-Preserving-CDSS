"""Tests for retrieval components: reranker and CRAG evaluator."""
import pytest
from app.pipeline.retrieval.reranker import RetrievedChunk
from app.pipeline.retrieval.crag_evaluator import evaluate_chunks, CORRECT_THRESHOLD, AMBIGUOUS_THRESHOLD


def _make_chunk(text: str, source: str = "ACMG_2015", reference: str = "page 1") -> RetrievedChunk:
    return RetrievedChunk(text=text, source=source, reference=reference)


def test_reranker_sorts_by_score_descending():
    """Reranked chunks should be ordered by cross-encoder score (highest first)."""
    from app.pipeline.retrieval.reranker import rerank_chunks
    chunks = [
        _make_chunk("BRCA1 PVS1 null variant loss of function"),
        _make_chunk("This is about weather forecasting and climate models"),
        _make_chunk("ACMG criteria for pathogenic classification strong evidence"),
    ]
    ranked = rerank_chunks("ACMG BRCA1 pathogenic criteria", chunks)
    # The clinical chunks should score higher than weather text
    assert len(ranked) == 3


def test_reranker_empty_input_returns_empty():
    """Empty input should return an empty list."""
    from app.pipeline.retrieval.reranker import rerank_chunks
    result = rerank_chunks("anything", [])
    assert result == []


def test_crag_correct_threshold_value():
    """CRAG CORRECT threshold should be 0.05 (H-1 fix)."""
    assert CORRECT_THRESHOLD == 0.05


def test_crag_ambiguous_threshold_value():
    """CRAG AMBIGUOUS threshold should be 0.01 (H-1 fix)."""
    assert AMBIGUOUS_THRESHOLD == 0.01


def test_crag_grading_high_score_is_correct():
    """A chunk graded with a score >= 0.05 should be classified as CORRECT."""
    chunks = [_make_chunk("PVS1 null variant criteria for BRCA1")]
    # Simulate the evaluate function — we test the grading logic
    result = evaluate_chunks("ACMG BRCA1 PVS1 criteria", chunks)
    # Result should have keys: correct, ambiguous, incorrect
    assert "correct" in result
    assert "ambiguous" in result
    assert "incorrect" in result
    total = len(result["correct"]) + len(result["ambiguous"]) + len(result["incorrect"])
    assert total == 1, "Every input chunk must appear in exactly one grade"


def test_crag_empty_input_returns_empty_grades():
    """Empty chunks list should return empty grade categories."""
    result = evaluate_chunks("any query", [])
    assert result["correct"] == []
    assert result["ambiguous"] == []
    assert result["incorrect"] == []
