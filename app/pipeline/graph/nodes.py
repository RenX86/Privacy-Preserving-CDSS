import re
import time
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
from app.pipeline.generation.self_rag import generate_answer
from app.pipeline.generation.citation_enforcer import extract_citations, fix_hallucinated_citations

def _gene_mentioned(gene_lower: str, text_lower: str) -> bool:
    """
    Check if the target gene is referenced in the chunk text.
    Handles combined notations like BRCA1/2, BRCA1/BRCA2, BRCA 1/2.
    """
    if gene_lower in text_lower:
        return True

    # Handle BRCA1/2 combined notation for BRCA2 queries
    if gene_lower == "brca2":
        return any(p in text_lower for p in ["brca1/2", "brca1/brca2", "brca 1/2"])
    if gene_lower == "brca1":
        return any(p in text_lower for p in ["brca1/2", "brca1/brca2", "brca 1/2"])

    return False


def decompose_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
    query = state["query"]
    gene = extract_gene_from_query(query)
    sub_queries = decompose_query(query, gene=gene)   # pass gene for focused sub-query text

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "gene": gene, 
        "sub_queries": sub_queries,
        "_trace": [{
            "node": "Decomposer",
            "duration_ms": elapsed,
            "summary": f"Extracted gene '{gene}'. Generated {len(sub_queries)} sub-queries.",
            "data": {"gene": gene, "sub_queries": [{"text": sq.text, "target": sq.target, "type": sq.query_type} for sq in sub_queries]}
        }]
    }

def retrieve_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
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
                        if freq_data and freq_data.get("found"):
                            # Variant exists in gnomAD — report its frequency
                            freq_text = (
                                f"Population frequency for {rsid}: "
                                f"AF={freq_data['allele_frequency']:.8f}. "
                                f"{'Variant is common in population (AF >= 5%).' if freq_data['is_common'] else 'Variant is rare in population.'}"
                            )
                            trusted_chunks.append(RetrievedChunk(
                                text=freq_text, source="gnomAD", reference=rsid
                            ))
                        elif freq_data and not freq_data.get("found"):
                            # Variant is NOT in gnomAD — clinically different from AF=0
                            trusted_chunks.append(RetrievedChunk(
                                text=f"gnomAD lookup for {rsid}: Variant NOT FOUND in gnomAD r4 population database. No allele frequency data exists. DO NOT report any allele frequency value for this variant.",
                                source="gnomAD",
                                reference=rsid
                            ))
                            # If freq_data is None entirely → gnomAD API call failed, don't add a chunk
                    trusted_chunks.append(from_postgres_result(result))

            if gene and not rsids:
                results = get_variant_by_gene(gene)
                for r in results:
                    trusted_chunks.append(from_postgres_result(r))

        elif sq.target == "clingen" and gene:
            results =get_gene_validity(gene)
            for r in results: 
                trusted_chunks.append(from_clingen_result(r, gene))

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "trusted_chunks": trusted_chunks,
        "_trace": [{
            "node": "DB_Retriever",
            "duration_ms": elapsed,
            "summary": f"Retrieved {len(trusted_chunks)} chunks from structured DBs.",
            "data": {"chunks_retrieved": len(trusted_chunks)}
        }]
    }


def retrieve_pdf_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
    candidate_chunks = []

    for sq in state["sub_queries"]:
        if sq.target == "vector_db":
            if sq.query_type == "protocol_retrieval":
                results = multi_query_search(sq.text, top_k=15, category_filter="treatment_protocol")
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

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "candidate_chunks": candidate_chunks,
        "_trace": [{
            "node": "PDF_Retriever",
            "duration_ms": elapsed,
            "summary": f"Retrieved {len(candidate_chunks)} candidate chunks from PDF vector index.",
            "data": {"chunks_retrieved": len(candidate_chunks)}
        }]
    }


def evaluate_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
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
            DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}
            is_nccn = source not in DB_SOURCES
            mentions_gene = gene_lower and _gene_mentioned(gene_lower, chunk.text.lower())
            if is_nccn and gene_lower and not mentions_gene:
                continue
            if source_counts.get(source, 0) < 6:
                filtered_candidates.append(chunk)
                source_counts[source] = source_counts.get(source, 0) + 1

    verified = trusted_chunks + filtered_candidates
    elapsed = int((time.perf_counter() - t0) * 1000)

    # Expose FULL chunk texts so the evaluation runner has the same evidence
    # the LLM used. Do NOT truncate here — truncated contexts cause unfair
    # Faithfulness scores in Ragas. The Ragas runner handles its own
    # token-budget truncation separately (_trim_contexts in run_ragas.py).
    chunk_texts = [
        {"source": c.source, "text": c.text}
        for c in verified[:15]          # cap at 15 chunks for response size
    ]

    return {
        "verified_chunks": verified,
        "_trace": [{
            "node": "Evaluator",
            "duration_ms": elapsed,
            "summary": f"CRAG passed {len(filtered_candidates)} PDF chunks. Total verified: {len(verified)}.",
            "data": {
                "verified_chunks_count": len(verified),
                "pdf_chunks_passed": len(filtered_candidates),
                "chunk_texts": chunk_texts,         # used by run_ragas.py
            }
        }]
    }

def generate_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
    query = state["query"]
    verified = state.get("verified_chunks", [])

    draft = generate_answer(query, verified)
    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "draft_answer": draft,
        "_trace": [{
            "node": "Generator",
            "duration_ms": elapsed,
            "summary": "Drafted initial response via LLM.",
            "data": {"answer_length": len(draft)}
        }]
    }

def citation_node(state: CDSSGraphState) -> dict:
    t0 = time.perf_counter()
    final = state["draft_answer"]
    verified_chunks = state.get("verified_chunks", [])

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

    elapsed = int((time.perf_counter() - t0) * 1000)
    # Return the updated final_answer along with citations
    return {
        "final_answer": fixed_final_answer, 
        "citations": citations, 
        "confidence": confidence,
        "_trace": [{
            "node": "Enforcer",
            "duration_ms": elapsed,
            "summary": f"Extracted {len(citations)} citations. Confidence: {confidence}.",
            "data": {"confidence": confidence, "citations_extracted": len(citations)}
        }]
    }