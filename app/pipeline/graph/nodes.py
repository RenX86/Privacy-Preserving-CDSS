import re
from app.config import settings
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
    sub_queries = decompose_query(query, gene=gene)   # pass gene for focused sub-query text

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
                    # Only call gnomAD if the feature flag is enabled (H-5 privacy fix)
                    if settings.ENABLE_GNOMAD_LOOKUP:
                        freq_data = get_allele_frequency(rsid)
                        if freq_data:
                            freq_text = f"Population frequency for {rsid}: AF={freq_data['allele_frequency']:.6f}. {'BA1 rule APPLIES (common variant -> Benign).' if freq_data['ba1_applicable'] else 'Variant is rare in population.'}"
                            trusted_chunks.append(RetrievedChunk(text=freq_text, source="gnomAD", reference=rsid))
                        else:
                            trusted_chunks.append(RetrievedChunk(
                                text=f"gnomAD lookup for {rsid}: Variant not found in gnomAD r4 population database. BA1 benign stand-alone rule does NOT apply (variant is not common). Absence from population databases is noted (relevant to PM2 criterion — requires clinical review).",
                                source="gnomAD",
                                reference=rsid
                            ))

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
            if sq.query_type == "protocol_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="protocol")
            elif sq.query_type == "screening_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="screening_protocol")
            else:
                results = multi_query_search(sq.text, top_k=15)

            batch = [from_vector_result(r) for r in results]

            # ── Per-subquery reranking ────────────────────────────────────────────
            # Rerank THIS batch against its own focused sub-query text.
            # Each batch is scored against its own topic to avoid cross-topic
            # relevance dilution.
            if batch:
                batch = rerank_chunks(sq.text, batch)
                print(f"  [PDF] Reranked {len(batch)} chunks against: {sq.text[:60]}...")

            candidate_chunks.extend(batch)

    return {"candidate_chunks": candidate_chunks}


def evaluate_node(state: CDSSGraphState) -> dict:
    candidate_chunks = state.get("candidate_chunks", [])
    trusted_chunks   = state.get("trusted_chunks", [])
    query            = state["query"]
    gene             = state.get("gene", "")
    gene_lower       = gene.lower() if gene else ""

    filtered_candidates = []

    if candidate_chunks:
        # Deduplicate (same chunk found by multiple sub-queries → keep first)
        seen_keys = set()
        unique_candidates = []
        for c in candidate_chunks:
            key = c.text[:200]
            if key not in seen_keys:
                seen_keys.add(key)
                unique_candidates.append(c)

        # Sort by score (set during per-subquery reranking in retrieve_pdf_node)
        # so the CRAG log output is easy to read (highest scores first)
        unique_candidates.sort(key=lambda c: c.score, reverse=True)

        # CRAG grading — chunks already have scores from per-subquery reranking.
        # No re-reranking against the full query here.
        evaluation = evaluate_chunks(query, unique_candidates)
        all_passing = evaluation["correct"] + evaluation["ambiguous"]

        # Gene filter + source cap
        source_counts = {}
        for chunk in all_passing:
            source   = chunk.source
            is_nccn  = "nccn" in source.lower() or "genetic" in source.lower()
            mentions_gene = gene_lower and gene_lower in chunk.text.lower()
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
    verified_chunks = state.get("verified_chunks", [])
    trusted_chunks = state.get("trusted_chunks", [])

    # Fix any inline hallucinated formatting before we extract
    fixed_final_answer = fix_hallucinated_citations(final, verified_chunks)

    raw_citations = extract_citations(fixed_final_answer, verified_chunks)
    citations = [Citation(source=c["source"], reference=c["reference"]) for c in raw_citations]

    # ── Confidence: use POST-FILTER counts from verified_chunks (H-7 fix) ────
    # Old code used len(candidate_chunks) — the raw PDF_Retriever output BEFORE
    # CRAG filtering. This inflated confidence: many irrelevant chunks retrieved
    # → confidence "high" even if CRAG dropped all of them.
    # Now we count only the chunks that actually reached the LLM.
    DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}
    db_count  = sum(1 for c in verified_chunks if c.source in DB_SOURCES)
    pdf_count = sum(1 for c in verified_chunks if c.source not in DB_SOURCES)

    has_db       = db_count >= 2
    has_guidelines = pdf_count > 0

    if has_db and has_guidelines:
        confidence = "high"
    elif has_db or has_guidelines:
        confidence = "medium"
    else:
        confidence = "low"

    print(f"[Citation] Verified DB chunks: {db_count} | Verified PDF chunks: {pdf_count} → confidence: {confidence}")

    # Return the updated final_answer along with citations
    return {"final_answer": fixed_final_answer, "citations": citations, "confidence": confidence}