import re
from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse, Citation
from app.pipeline.decomposition import decompose_query

#data sources
from app.pipeline.sources.postgres_client import get_variant_by_rsid, get_variant_by_gene
from app.pipeline.sources.vector_client import search_documents
from app.pipeline.retrieval.multi_query import multi_query_search
from app.pipeline.sources.clingen_client import get_gene_validity, extract_gene_from_query

#retrival and verification
from app.pipeline.retrieval.reranker import (
    rerank_chunks, from_postgres_result, from_vector_result, from_clingen_result, RetrievedChunk
)
from app.pipeline.retrieval.crag_evaluator import evaluate_chunks, has_sufficient_context

#generation
from app.pipeline.generation.guardrails import SAFE_FAILURE_MESSAGE
from app.pipeline.generation.self_rag import generate_answer, self_rag_critic
from app.pipeline.generation.citation_enforcer import extract_citations

from app.pipeline.sources.gnomad_client import get_allele_frequency

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}

@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):

    query = request.query
    print(f"\n{'='*60}")
    print(f"[PHASE 1] QUERY RECEIVED: {query}")
    print(f"{'='*60}")

    sub_queries = decompose_query(query)
    print(f"\n[PHASE 1] DECOMPOSED INTO {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  [{i}] target={sq.target} | type={sq.query_type} | text={sq.text[:60]}...")

    # ── TWO LANES ─────────────────────────────────────────────────────────────
    # trusted_chunks   : curated databases (ClinVar, gnomAD, ClinGen)
    #                    → ALWAYS included, never scored by cross-encoder
    # candidate_chunks : PDF vector results (ACMG, NCCN guidelines)
    #                    → CRAG filtered for relevance before use
    # ──────────────────────────────────────────────────────────────────────────
    trusted_chunks   = []
    candidate_chunks = []

    gene = extract_gene_from_query(query)
    print(f"\n[PHASE 2] Detected gene: {gene}")

    for sq in sub_queries:

        if sq.target == "postgres":
            print(f"\n[PHASE 2] -> PostgreSQL (ClinVar) [TRUSTED]")
            rsids = re.findall(r'rs\d+', sq.text)
            for rsid in rsids:
                result = get_variant_by_rsid(rsid)
                if result:
                    print(f"  [DB] ClinVar: {rsid} -> {result.get('clinical_significance')} in {result.get('gene_symbol')}")
                    freq_data = get_allele_frequency(rsid)
                    if freq_data:
                        print(f"  [DB] gnomAD: AF={freq_data['allele_frequency']:.6f} | BA1={freq_data['ba1_applicable']}")
                        freq_text = (
                            f"Population frequency for {rsid}: "
                            f"AF={freq_data['allele_frequency']:.6f}. "
                            f"{'BA1 rule APPLIES (common variant -> Benign).' if freq_data['ba1_applicable'] else 'Variant is rare in population.'}"
                        )
                        trusted_chunks.append(RetrievedChunk(
                            text=freq_text,
                            source="gnomAD",
                            reference=rsid
                        ))
                    else:
                        print(f"  [gnomAD] {rsid} absent from gnomAD r4 — BA1 rule does NOT apply (supports rare/pathogenic)")
                    trusted_chunks.append(from_postgres_result(result))
                else:
                    print(f"  [WARN] ClinVar: {rsid} not found in local DB")

            if gene and not rsids:
                results = get_variant_by_gene(gene)
                print(f"  [DB] ClinVar gene search: {gene} -> {len(results)} variants")
                for r in results:
                    trusted_chunks.append(from_postgres_result(r))

        elif sq.target == "vector_db":
            print(f"\n[PHASE 2] -> Vector DB (ACMG/NCCN) [CRAG-FILTERED]")
            if sq.query_type == "rule_retrieval":
                results = multi_query_search(sq.text, top_k=5, category_filter="guideline")
            elif sq.query_type == "protocol_retrieval":
                results = multi_query_search(sq.text, top_k=5, category_filter="protocol")
            else:
                results = multi_query_search(sq.text, top_k=5)
            print(f"  Multi-Query returned {len(results)} unique chunks")
            for r in results:
                print(f"     source={r.get('source')} | similarity={r.get('similarity', 0):.3f}")
            for r in results:
                candidate_chunks.append(from_vector_result(r))

        elif sq.target == "clingen":
            print(f"\n[PHASE 2] -> ClinGen API [TRUSTED]")
            if gene:
                results = get_gene_validity(gene)
                print(f"  [DB] ClinGen: {len(results)} expert panel results for {gene}")
                for r in results:
                    trusted_chunks.append(from_clingen_result(r, gene))
            else:
                print(f"  [WARN] ClinGen: no gene detected — skipped")

    print(f"\n[PHASE 3] trusted={len(trusted_chunks)} | candidates={len(candidate_chunks)}")

    # ── PHASE 4: Rerank and CRAG-filter only the PDF candidates ───────────────
    filtered_candidates = []
    if candidate_chunks:
        ranked = rerank_chunks(query, candidate_chunks)
        print(f"\n[PHASE 4] RE-RANKING PDF candidates (cross-encoder scores):")
        for c in ranked:
            print(f"  score={c.score:.3f} | source={c.source} | text={c.text[:60]}...")

        evaluation = evaluate_chunks(query, ranked)
        filtered_candidates = evaluation["correct"] + evaluation["ambiguous"]
        print(f"\n[PHASE 5] CRAG (PDF only): correct={len(evaluation['correct'])} | "
              f"ambiguous={len(evaluation['ambiguous'])} | incorrect={len(evaluation['incorrect'])}")
    else:
        print("\n[PHASE 4] No PDF candidates to rank")

    # ── FINAL VERIFIED POOL ───────────────────────────────────────────────────
    # Trusted DB facts always first, filtered PDF sections appended after
    verified = trusted_chunks + filtered_candidates

    print(f"\n[PHASE 5] FINAL CONTEXT: {len(trusted_chunks)} trusted DB + "
          f"{len(filtered_candidates)} filtered PDF = {len(verified)} total")

    if not verified:
        print("  [FAIL] No context — returning safe failure")
        return QueryResponse(
            answer=SAFE_FAILURE_MESSAGE,
            citations=[],
            confidence="low",
            safe_failure=True
        )

    print(f"\n[PHASE 6] GENERATION")
    draft = generate_answer(query, verified)
    print(f"  Draft: {draft[:120]}...")
    answer = self_rag_critic(query, draft, verified)
    print(f"  Final: {answer[:120]}...")

    raw_citations = extract_citations(answer, verified)
    citations = [
        Citation(source=c["source"], reference=c["reference"])
        for c in raw_citations
    ]

    # Confidence based on trusted DB sources (not PDF chunks)
    if len(trusted_chunks) >= 2:
        confidence = "high"
    elif len(trusted_chunks) >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    print(f"\n[DONE] confidence={confidence} | citations={len(citations)}")
    print(f"{'='*60}\n")

    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        safe_failure=False
    )
