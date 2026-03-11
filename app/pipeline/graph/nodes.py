import re
from app.pipeline.graph.state import CDSSGraphState

from app.pipeline.decomposition import decompose_query
from app.pipeline.sources.postgres_client import get_variant_by_gene, get_variant_by_rsid
from app.pipeline.sources.gnomad_client import get_allele_frequency
from app.pipeline.sources.clingen_client import extract_gene_from_query, get_gene_validity
from app.pipeline.retrieval.reranker import from_postgres_result, from_clingen_result, from_vector_result, RetrievedChunk
from app.pipeline.retrieval.multi_query import multi_query_search
from app.pipeline.retrieval.crag_evaluator import evaluate_chunks
from app.pipeline.retrieval.reranker import rerank_chunks
from app.api.schemas import Citation
from app.pipeline.generation.self_rag import generate_answer, self_rag_critic
from app.pipeline.generation.citation_enforcer import extract_citations, fix_hallucinated_citations

def decompose_node(state: CDSSGraphState) -> dict:

    query = state["query"]
    gene = extract_gene_from_query(query)
    sub_queries = decompose_query(query)

    return {"gene": gene, "sub_queries": sub_queries}

def retrieve_node(state: CDSSGraphState) -> dict:

    trusted_chunks = []
    gene = state.get("gene")

    for sq in state["sub_queries"]:
        if sq.target == "postgres":
            rsids = re.findall(r'rs\d+', sq.text)
            for rsid in rsids:
                result = get_variant_by_rsid(rsid)
                if result:
                    freq_data = get_allele_frequency(rsid)
                    if freq_data:
                        freq_text = f"Population frequency for {rsid}: AF={freq_data['allele_frequency']:.6f}. {'BA1 rule APPLIES (common variant -> Benign).' if freq_data['ba1_applicable'] else 'Variant is rare in population.'}"
                        trusted_chunks.append(RetrievedChunk(text=freq_text, source="gnomAD", reference=rsid))
                    else:
                        trusted_chunks.append(RetrievedChunk(text=f"gnomAD lookup for {rsid}: Variant not found in gnomAD r4 population database. ACMG BA1 benign stand-alone rule does NOT apply.", source="gnomAD", reference=rsid))

                    trusted_chunks.append(from_postgres_result(result))

            if gene and not rsids:
                results = get_variant_by_gene(gene)
                for r in results:
                    trusted_chunks.append(from_postgres_result(r))

        elif sq.target == "clingen" and gene:
            results =get_gene_validity(gene)
            for r in results: 
                trusted_chunks.append(from_clingen_result(r, gene))

    return {"trusted_chunks": trusted_chunks}

def retrieve_pdf_node(state: CDSSGraphState) -> dict:

    candidate_chunks = []

    for sq in state["sub_queries"]:
        if sq.target == "vector_db":
            if sq.query_type == "rule_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="guideline")
            elif sq.query_type == "protocol_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="protocol")
            elif sq.query_type == "screening_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="screening_protocol")
            else: 
                results = multi_query_search(sq.text, top_k=15)

            for r in results:
                candidate_chunks.append(from_vector_result(r))

    return {"candidate_chunks": candidate_chunks}


def evaluate_node(state: CDSSGraphState) -> dict:

    candidate_chunks = state.get("candidate_chunks", [])
    trusted_chunks = state.get("trusted_chunks", [])
    query = state["query"]

    TOP_PDF_CHUNKS = 5
    filtered_candidates = []

    if candidate_chunks:
        
        seen_parents = set()
        unique_candidates = []
        for c in candidate_chunks:
            key = c.text[:200]
            if key not in seen_parents:
                seen_parents.add(key)
                unique_candidates.append(c)

        ranked = rerank_chunks(query, unique_candidates)
        evaluation = evaluate_chunks(query, ranked)

        all_passing = evaluation["correct"] + evaluation["ambiguous"]
        
        gene = state.get("gene", "")
        gene_lower = gene.lower() if gene else ""
        
        # Keep top 6 chunks per unique source, but strongly filter NCCN for the target gene 
        # to prevent irrelevant tumor protocols (STK11, CDH1) from crowding out BRCA1 tables via CRAG
        source_counts = {}
        for chunk in all_passing:
            source = chunk.source
            is_nccn = "nccn" in source.lower() or "genetic" in source.lower()
            mentions_gene = gene_lower and gene_lower in chunk.text.lower()
            
            # Drop NCCN chunks that don't mention the target gene (ACMG rules don't mention genes, so keep them)
            if is_nccn and gene_lower and not mentions_gene:
                continue

            if source_counts.get(source, 0) < 6:
                filtered_candidates.append(chunk)
                source_counts[source] = source_counts.get(source, 0) + 1

    verified = trusted_chunks + filtered_candidates
    return {"verified_chunks": verified}

def generate_node(state: CDSSGraphState) -> dict:

    query = state["query"]
    verified = state.get("verified_chunks", [])

    draft = generate_answer(query, verified)
    return {"draft_answer": draft}

def critic_node(state: CDSSGraphState) -> dict:

    query = state["query"]
    draft = state["draft_answer"]
    verified = state.get("verified_chunks", [])

    final = self_rag_critic(query, draft, verified)
    return {"final_answer": final}

def citation_node(state: CDSSGraphState) -> dict:

    final = state["final_answer"]
    verified = state.get("verified_chunks", [])
    trusted_chunks = state.get("trusted_chunks", [])

    # Fix any inline hallucinated formatting before we extract
    fixed_final_answer = fix_hallucinated_citations(final, verified)

    raw_citations = extract_citations(fixed_final_answer, verified)
    citations = [Citation(source=c["source"], reference=c["reference"]) for c in raw_citations]

    if len(trusted_chunks) >= 2:
        confidence = "high"
    elif len(trusted_chunks) >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # Return the updated final_answer along with citations
    return {"final_answer": fixed_final_answer, "citations": citations, "confidence": confidence}