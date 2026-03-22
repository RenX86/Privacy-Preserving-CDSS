"""Tests for query decomposition and routing logic."""
import pytest
from app.pipeline.decomposition import decompose_query, SubQuery


def test_rsid_produces_postgres_subquery():
    """An rsID pattern should route to postgres for data extraction."""
    subs = decompose_query("What is the significance of rs80357713?")
    targets = [sq.target for sq in subs]
    types   = [sq.query_type for sq in subs]
    assert "postgres" in targets
    assert "data_extraction" in types


def test_acmg_keyword_produces_rule_retrieval():
    """ACMG keyword should trigger a vector_db rule_retrieval sub-query."""
    subs = decompose_query("Which ACMG criteria classify BRCA1 as pathogenic?")
    matches = [sq for sq in subs if sq.target == "vector_db" and sq.query_type == "rule_retrieval"]
    assert len(matches) >= 1


def test_screening_keyword_produces_screening_retrieval():
    """Screening-related keywords should route to screening_retrieval."""
    subs = decompose_query("What is the NCCN screening protocol for BRCA1 carriers?")
    matches = [sq for sq in subs if sq.query_type == "screening_retrieval"]
    assert len(matches) >= 1


def test_protocol_keywords_without_screening():
    """Treatment protocol keywords without screening context should route to protocol_retrieval."""
    subs = decompose_query("What chemotherapy regimen is used for HER2 positive breast cancer?")
    matches = [sq for sq in subs if sq.query_type == "protocol_retrieval"]
    assert len(matches) >= 1


def test_protocol_keywords_suppressed_when_screening_present():
    """Protocol keywords should NOT fire when screening keywords are also present."""
    subs = decompose_query("What is the NCCN screening protocol for hereditary breast cancer?")
    proto = [sq for sq in subs if sq.query_type == "protocol_retrieval"]
    assert len(proto) == 0, "protocol_retrieval should be suppressed when screening keywords are present"


def test_clingen_keyword_routes_to_clingen():
    """ClinGen-specific keywords should produce a clingen sub-query."""
    subs = decompose_query("Confirm the ClinGen expert panel validity for BRCA1")
    matches = [sq for sq in subs if sq.target == "clingen"]
    assert len(matches) >= 1


def test_clinvar_mention_does_not_trigger_clingen():
    """ClinVar mentions should NOT trigger a ClinGen API call (H-4 fix)."""
    subs = decompose_query("What does ClinVar say about BRCA1 variants?")
    clingen_subs = [sq for sq in subs if sq.target == "clingen"]
    assert len(clingen_subs) == 0, "ClinVar query incorrectly routed to ClinGen"


def test_empty_query_returns_fallback():
    """An empty or unrecognised query should produce a default vector_db fallback."""
    subs = decompose_query("tell me something interesting")
    assert len(subs) >= 1
    assert subs[0].target == "vector_db"


def test_multiple_rsids_produce_multiple_subqueries():
    """Multiple rsIDs in one query should each get their own postgres sub-query."""
    subs = decompose_query("Compare rs80357713 and rs879254116 in BRCA1")
    postgres_subs = [sq for sq in subs if sq.target == "postgres"]
    assert len(postgres_subs) >= 2
