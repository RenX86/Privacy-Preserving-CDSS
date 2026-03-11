import ollama
from app.config import settings
from app.pipeline.sources.vector_client import search_documents

# ─── STEP 1: GENERATE QUERY VARIANTS ──────────────────────────────────────────

def expand_queries(original_query: str, n: int = 3, query_type: str = "general") -> list[str]:
    """
    Generate targeted clinical sub-queries to retrieve specific guideline sections.
    Focuses on extracting ACMG criteria codes and classification rules.
    """
    if query_type in ["protocol_retrieval", "screening_retrieval"]:
        prompt = f"""You are a clinical guidelines search assistant.
        Generate {n} short, specific search queries to retrieve clinical management and screening sections.
        Focus on: screening protocols, surveillance schedules, risk-reducing surgery,
        mammography, MRI, genetic testing recommendations, mutation carrier management.
        DO NOT generate ACMG criteria codes. Generate clinical management keyword queries.
        Return ONLY the queries, one per line, no numbering.

        Original query: {original_query}

        Example good queries:
        BRCA1 carrier annual screening mammography MRI schedule
        risk-reducing salpingo-oophorectomy RRSO age recommendation
        hereditary breast cancer surveillance protocol screening"""

    else:
        prompt = f"""You are a clinical guidelines search assistant.
        Generate {n} short, specific search queries to retrieve ACMG/clinical guideline sections.
        Focus on: criteria codes (PVS1, PS1-4, PM1-6, PP1-5, BA1, BS1-4, BP1-7),
        classification rules, evidence weights, and pathogenicity thresholds.
        DO NOT rephrase the full question. Generate short targeted keyword queries.
        Return ONLY the queries, one per line, no numbering.

        Original query: {original_query}

        Example good queries:
        ACMG PVS1 null variant loss of function criteria
        pathogenic classification criteria strong evidence PS1 PS2
        benign variant population frequency BA1 BS1 criteria"""

    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.message.content.strip()
        variants = [line.strip() for line in raw.splitlines() if line.strip()][:n]
        all_queries = [original_query] + variants
        print(f"  [MultiQuery] Generated {len(variants)} targeted queries:")
        for q in variants:
            print(f"    • {q[:80]}")
        return all_queries

    except Exception as e:
        print(f"  [MultiQuery] LLM expansion failed: {e} -- using original only")
        return [original_query]


# ─── STEP 2: SEARCH + DEDUPLICATE ─────────────────────────────────────────────

def multi_query_search(
    query: str,
    top_k: int = 5,
    category_filter: str = None,
    n_variants: int = 3
) -> list[dict]:
    """
    1. Expand the query into N variants
    2. Run vector search for each variant
    3. Deduplicate results by chunk text (same chunk found multiple times = keep once)
    4. Return the full unique pool — let the reranker pick the best ones
    """
    queries = expand_queries(query, n=n_variants, query_type="protocol_retrieval" if category_filter in ["protocol", "screening_protocol"] else "rule_retrieval")

    seen_texts = set()
    unique_results = []

    for q in queries:
        results = search_documents(q, top_k=top_k, category_filter=category_filter)
        for r in results:
            # Deduplicate by chunk_text (small chunks - more variety than parent_text)
            # parent_text deduplication was too aggressive: all 20 results had same 2 parents
            key = (r.get("chunk_text", "") or r.get("parent_text", ""))[:160]
            if key not in seen_texts:
                seen_texts.add(key)
                unique_results.append(r)

    print(f"  [MultiQuery] {len(queries)} queries × up to {top_k} results = {len(unique_results)} unique chunks")
    return unique_results
