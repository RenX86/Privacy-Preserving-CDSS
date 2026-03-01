import re
from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse, Citation
from app.pipeline.decomposition import decompose_query

#data sources
from app.pipeline.sources.postgres_client import get_variant_by_rsid, get_variant_by_gene
from app.pipeline.sources.vector_client import search_documents
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
    sub_queries = decompose_query(query)

    all_chunks = []

    gene = extract_gene_from_query(query)

    for sq in sub_queries:

        if sq.target == "postgres":

            rsids = re.findall(r'rs\d+', sq.text)
            for rsid in rsids:
                result = get_variant_by_rsid(rsid)
                if result:
                    # Also fetch population frequency from gnomAD
                    freq_data = get_allele_frequency(rsid)
                    if freq_data:
                        freq_text = (
                            f"Population frequency for {rsid}: "
                            f"AF={freq_data['allele_frequency']:.6f}. "
                            f"{'BA1 rule APPLIES (common variant → Benign).' if freq_data['ba1_applicable'] else 'Variant is rare in population.'}"
                        )
                        all_chunks.append(RetrievedChunk(
                            text=freq_text,
                            source="gnomAD",
                            reference=rsid
                        ))
                    all_chunks.append(from_postgres_result(result))

            if gene and not rsids:
                results = get_variant_by_gene(gene)
                for r in results:
                    all_chunks.append(from_postgres_result(r))
        
        elif sq.target == "vector_db":

            if sq.query_type == "rule_retrieval":
                results = search_documents(sq.text, top_k=5, category_filter="guideline")
            elif sq.query_type == "protocol_retrieval":
                results = search_documents(sq.text, top_k=5, category_filter="protocol")
            else:
                results = search_documents(sq.text, top_k=5)
            for r in results:
                all_chunks.append(from_vector_result(r))
            
        elif sq.target == "clingen":
            if gene:
                results = get_gene_validity(gene)
                for r in results:
                    all_chunks.append(from_clingen_result(r,gene))

    ranked_chunks = rerank_chunks(query, all_chunks)
    evaluation    = evaluate_chunks(query, ranked_chunks)
    verified      = evaluation["correct"]

    if not has_sufficient_context(evaluation):
        return QueryResponse(
            answer=SAFE_FAILURE_MESSAGE,
            citations=[],
            confidence="low",
            safe_failure=True
        )
    
    draft = generate_answer(query, verified)
    answer = self_rag_critic(query, draft, verified)

    raw_citations = extract_citations(answer, verified)
    citations = [
        Citation(source=c["source"], reference=c["reference"])
        for c in raw_citations
    ]

    if len(verified) >= 3:
        confidence = "high"
    elif len(verified) >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        safe_failure=False
    )

