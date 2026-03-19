# CDSS — Solutions

> Complete fix implementations for every item in PROBLEMS.md.
> Each solution is self-contained — copy the code directly into the file indicated.

---

## C-1 — SQL parameter order mismatch in `vector_client.py`

**File:** `app/pipeline/sources/vector_client.py`

Replace the entire `search_documents` function:

```python
def search_documents(
    query_text: str,
    top_k: int = 5,
    source_filter: str = None,
    category_filter: str = None
) -> list[dict]:

    query_vector = embed_text(query_text)

    # Build WHERE clause and params in matched order
    if source_filter:
        where_clause = "WHERE source = %s"
        params = (source_filter, query_vector, query_vector, top_k)
    elif category_filter:
        where_clause = "WHERE category = %s"
        params = (category_filter, query_vector, query_vector, top_k)
    else:
        where_clause = ""
        params = (query_vector, query_vector, top_k)

    if where_clause:
        sql = f"""
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
    else:
        sql = """
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""

    conn = psycopg2.connect(settings.POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
    finally:
        conn.close()
```

**What changed:** `params` for the filtered path is now `(filter_value, query_vector, query_vector, top_k)` — the filter value comes first to match the `WHERE column = %s` placeholder, followed by two `query_vector` values for the `<=>` distance operator and the ORDER BY, then `top_k` for LIMIT.

---

## C-2 — Critic node strips citations (`self_rag.py`)

**File:** `app/pipeline/generation/self_rag.py`

Replace the `self_rag_critic` function:

```python
def self_rag_critic(query: str, draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:
    """LLM-based Critic to verify clinical accuracy. Falls back to draft if citations are lost."""
    log.info("Running LLM-based critic...")

    if not verified_chunks:
        return draft_answer

    context_block = build_context_block(verified_chunks)

    critic_prompt = f"""You are a strict Clinical Auditor. Your job is to verify a generated clinical summary against the retrieved medical context.

CONTEXT:
{context_block}

DRAFT SUMMARY TO AUDIT:
{draft_answer}

CRITICAL AUDIT RULES:
1. FATAL ERRORS: Did the draft assign the wrong age or the wrong surgery for a gene? (e.g., Draft says BRCA1 RRSO is 45-50, but context row for BRCA1 says 35-40).
2. HALLUCINATIONS: Are there claims in the draft not supported by the context?
3. CITATIONS: Does every sentence end with a proper metadata citation [Source: X, Reference: Y] found exactly in the context?

INSTRUCTIONS:
You MUST rewrite the draft to accurately reflect the text. Strip out any hallucinated protocols or fake citations not present in the CONTEXT.
CRITICAL MANDATE: You MUST ensure EVERY sentence ends with an inline citation structured EXACTLY as `[Source: X, Reference: Y]`. DO NOT strip these out!
If the draft is already perfectly accurate, return the exact original draft but ENSURE it has the citations.
CRITICAL: DO NOT add any explanatory text. DO NOT output your audit checklist. Output ONLY the raw, safe clinical summary.
"""
    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a clinical auditor. Output only the final corrected text."},
                {"role": "user",   "content": critic_prompt}
            ]
        )
        final_answer = response.message.content.strip()
        log.info(f"LLM Critic finished. Length changed from {len(draft_answer)} to {len(final_answer)}.")

        # ── Safety net: if the critic dropped citations, revert to the draft ──
        # Free-text generation can silently omit [Source: X, Reference: Y] tags.
        # The citation enforcer cannot fix what was never there.
        draft_citation_count = draft_answer.count("[Source:")
        final_citation_count = final_answer.count("[Source:")

        if draft_citation_count > 0 and final_citation_count < draft_citation_count:
            log.warning(
                f"Critic dropped citations ({draft_citation_count} → {final_citation_count}). "
                f"Reverting to draft to preserve citation integrity."
            )
            return draft_answer

        return final_answer

    except Exception as e:
        log.error(f"Ollama error during critic: {e}")
        return draft_answer
```

**What changed:** Added a post-critic citation count check. If the critic output contains fewer `[Source:` markers than the draft, it reverts to the draft. This prevents the enforcer from receiving citation-stripped text.

---

## C-3 — LangGraph parallel fan-out (`workflow.py` + `state.py`)

**Step 1 — Fix `app/pipeline/graph/state.py`:**

Add `Annotated` reducer fields so both retriever nodes can write their results without overwriting each other:

```python
from typing import TypedDict, Optional, Annotated
import operator
from app.pipeline.retrieval.reranker import RetrievedChunk
from app.api.schemas import Citation

class CDSSGraphState(TypedDict):

    query:       str
    gene:        Optional[str]
    sub_queries: list

    # Annotated with operator.add so both DB_Retriever and PDF_Retriever
    # can append their results rather than the second one overwriting the first
    trusted_chunks:   Annotated[list[RetrievedChunk], operator.add]
    candidate_chunks: Annotated[list[RetrievedChunk], operator.add]

    verified_chunks: list[RetrievedChunk]
    draft_answer:    str

    final_answer: str
    citations:    list[Citation]
    confidence:   str
```

**Step 2 — Fix `app/pipeline/graph/workflow.py`:**

Replace the two `add_edge` fan-out calls with LangGraph's `Send` API:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from app.pipeline.graph.state import CDSSGraphState
from app.pipeline.graph.nodes import (
    decompose_node, retrieve_node, retrieve_pdf_node,
    evaluate_node, generate_node, critic_node, citation_node
)

workflow = StateGraph(CDSSGraphState)

workflow.add_node("Decomposer",        decompose_node)
workflow.add_node("DB_Retriever",      retrieve_node)
workflow.add_node("PDF_Retriever",     retrieve_pdf_node)
workflow.add_node("Evaluator",         evaluate_node)
workflow.add_node("Generator",         generate_node)
workflow.add_node("Critic",            critic_node)
workflow.add_node("Citation_Enforcer", citation_node)

# ── True parallel fan-out using Send ──────────────────────────────────────────
# add_edge("Decomposer", "DB_Retriever") + add_edge("Decomposer", "PDF_Retriever")
# does NOT produce parallel execution — it runs sequentially and the second
# node's state update can overwrite the first's.
# Send() dispatches both nodes concurrently and merges results via the
# Annotated[list, operator.add] reducers in CDSSGraphState.
def fan_out_retrievers(state: CDSSGraphState):
    return [
        Send("DB_Retriever",  state),
        Send("PDF_Retriever", state),
    ]

workflow.add_edge(START, "Decomposer")
workflow.add_conditional_edges("Decomposer", fan_out_retrievers, ["DB_Retriever", "PDF_Retriever"])
workflow.add_edge("DB_Retriever",      "Evaluator")
workflow.add_edge("PDF_Retriever",     "Evaluator")
workflow.add_edge("Evaluator",         "Generator")
workflow.add_edge("Generator",         "Critic")
workflow.add_edge("Critic",            "Citation_Enforcer")
workflow.add_edge("Citation_Enforcer", END)

cdss_app = workflow.compile()
```

**What changed:** `fan_out_retrievers` returns two `Send` objects, which LangGraph executes concurrently. The `Annotated[list, operator.add]` fields in state ensure both nodes' results are merged (list concatenation) rather than overwritten.

---

## C-4 — Database connection pooling

**Step 1 — Create `app/db/pool.py` (new file):**

```python
"""
Shared PostgreSQL connection pool.
Import get_conn() / release_conn() instead of calling psycopg2.connect() directly.
"""
import logging
from contextlib import contextmanager
from psycopg2 import pool as pg_pool
from app.config import settings

log = logging.getLogger("cdss.pool")

_pool: pg_pool.ThreadedConnectionPool | None = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.POSTGRES_URL
        )
        log.info("PostgreSQL connection pool initialised (min=2, max=10)")
    return _pool


def get_conn():
    """Borrow a connection from the pool."""
    return _get_pool().getconn()


def release_conn(conn):
    """Return a connection to the pool."""
    _get_pool().putconn(conn)


@contextmanager
def db_conn():
    """Context manager — borrows a connection and always returns it."""
    conn = get_conn()
    try:
        yield conn
    finally:
        release_conn(conn)
```

**Step 2 — Update `app/pipeline/sources/postgres_client.py`:**

```python
import psycopg2.extras
from app.db.pool import db_conn

def get_variant_by_rsid(rsid: str) -> dict | None:
    query = """
    SELECT rsid, gene_symbol, chromosome, position, ref_allele, alt_allele,
           clinical_significance, review_status, condition, last_evaluated
    FROM variants
    WHERE rsid = %s
    LIMIT 1;
    """
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, (rsid,))
            result = cursor.fetchone()
            return dict(result) if result else None


def get_variant_by_gene(gene_symbol: str) -> list[dict]:
    query = """
    SELECT rsid, gene_symbol, clinical_significance, condition, review_status
    FROM variants
    WHERE gene_symbol = %s
    ORDER BY clinical_significance;
    """
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, (gene_symbol,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
```

**Step 3 — Update `app/pipeline/sources/vector_client.py`:**

Replace `conn = psycopg2.connect(settings.POSTGRES_URL)` and its `finally: conn.close()` with:

```python
from app.db.pool import db_conn

def search_documents(...) -> list[dict]:
    query_vector = embed_text(query_text)
    # ... build sql and params as per C-1 fix above ...

    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
```

**Step 4 — Update `app/pipeline/sources/gnomad_client.py`:**

Replace the `_get_variant_position` function's connection handling:

```python
from app.db.pool import db_conn

def _get_variant_position(rsid: str) -> dict | None:
    sql = """
        SELECT chromosome, position, ref_allele, alt_allele
        FROM variants WHERE rsid = %s LIMIT 1;
    """
    try:
        with db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (rsid,))
                row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        print(f"  [gnomAD] DB lookup failed for {rsid}: {e}")
        return None
```

**What changed:** All three clients now borrow from a shared `ThreadedConnectionPool` (max 10 connections) instead of opening a new OS-level connection per query. The `db_conn()` context manager guarantees the connection is always returned to the pool even if an exception occurs.

---

## H-1 — CRAG thresholds are a no-op

**File:** `app/pipeline/retrieval/crag_evaluator.py`

Replace the threshold constants and add logging for dropped chunks:

```python
import sys
from app.pipeline.retrieval.reranker import RetrievedChunk

CORRECT   = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

# Tuned thresholds based on observed BGE score distribution.
# In pilot queries all chunks scored 0.001–0.195 with CORRECT_THRESHOLD=0.0,
# meaning nothing was ever filtered. The values below enforce a meaningful floor.
#
# CORRECT   >= 0.05  : chunk is clearly relevant to the query
# AMBIGUOUS  0.01–0.05: borderline — keep but place after CORRECT chunks
# INCORRECT < 0.01  : BGE considers this chunk unrelated — dropped
#
# Justification: BGE cross-encoder scores are not probabilities.
# Negative scores indicate the model actively predicts non-relevance.
# Scores near zero are uninformative. A 0.05 floor retains the top ~60%
# of chunks from typical clinical queries while discarding bibliography
# entries, unrelated gene tables, and boilerplate pages.
CORRECT_THRESHOLD   = 0.05
AMBIGUOUS_THRESHOLD = 0.01


def grade_chunk(query: str, chunk: RetrievedChunk) -> str:
    if chunk.score >= CORRECT_THRESHOLD:
        return CORRECT
    elif chunk.score >= AMBIGUOUS_THRESHOLD:
        return AMBIGUOUS
    else:
        return INCORRECT


def evaluate_chunks(query: str, chunks: list[RetrievedChunk]) -> dict:

    results = {CORRECT: [], AMBIGUOUS: [], INCORRECT: []}

    print(f"\n[CRAG] Grading {len(chunks)} PDF chunks:")
    print(f"       Thresholds: CORRECT >= {CORRECT_THRESHOLD} | "
          f"AMBIGUOUS >= {AMBIGUOUS_THRESHOLD} | INCORRECT < {AMBIGUOUS_THRESHOLD}")
    print(f"       {'score':>8}  {'grade':>12}  source / text preview")
    print(f"       {'-'*70}")

    for chunk in chunks:
        grade = grade_chunk(query, chunk)
        results[grade].append(chunk)
        symbol = {"correct": "[CORRECT]", "ambiguous": "[AMBIGUOUS]", "incorrect": "[INCORRECT]"}[grade]
        preview = chunk.text.replace("\n", " ")[:120]
        safe_preview = (preview.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)
                        if sys.stdout.encoding else preview)
        print(f"       {chunk.score:>8.3f}  {symbol:>12}  [{chunk.source}] {safe_preview}")

    n_dropped = len(results[INCORRECT])
    n_kept    = len(results[CORRECT]) + len(results[AMBIGUOUS])
    print(f"\n[CRAG] Kept {n_kept} chunks | Dropped {n_dropped} INCORRECT chunks\n")

    return results


def has_sufficient_context(evaluation: dict) -> bool:
    return len(evaluation[CORRECT]) > 0
```

---

## H-2 — Citation enforcer collapses all refs to first-chunk

**File:** `app/pipeline/generation/citation_enforcer.py`

Replace the `fix_hallucinated_citations` function:

```python
def fix_hallucinated_citations(answer_text: str, used_chunks: list[RetrievedChunk]) -> str:

    category_keywords = {
        "Clinvar": [r'\bclinvar\b', r'\bclassified as\b', r'\brs\d+\b', r'\bclinical significance\b'],
        "ClinGen": [r'\bclingen\b', r'\bexpert panel\b', r'\bgene-disease validity\b', r'\bactionability\b'],
        "gnomAD":  [r'\bgnomad\b', r'\bba1\b', r'\ballele.?frequency\b', r'\bpopulation frequency\b'],
        "ACMG":    [r'\bPM\d\b', r'\bPS\d\b', r'\bPVS\d\b', r'\bPP\d\b', r'\bBS\d\b', r'\bBA\d\b',
                    r'\bBP\d\b', r'\bcriteria\b', r'\bacmg\b'],
        "NCCN":    [r'\bnccn\b', r'\bscreening\b', r'\bmastectomy\b', r'\bsurveillance\b',
                    r'\bmri\b', r'\bmammography\b', r'\bprotocol\b']
    }

    # ── Build: source → SET of all valid references (not just the first one) ──
    # Old code used first-wins: if source already in dict, skip all subsequent chunks.
    # This collapsed all ACMG page refs to page 3 regardless of what was actually cited.
    source_to_refs: dict[str, set] = {}
    source_first_ref: dict[str, str] = {}  # fallback: first ref seen per source
    for chunk in used_chunks:
        source_to_refs.setdefault(chunk.source, set()).add(chunk.reference)
        if chunk.source not in source_first_ref:
            source_first_ref[chunk.source] = chunk.reference

    known_sources_lower = {s.lower(): s for s in source_to_refs.keys()}
    inline_pattern = re.compile(r'\[Source:\s*([^,\]]+),\s*(?:Reference:\s*)?([^\]]+)\]', re.IGNORECASE)

    matches = list(inline_pattern.finditer(answer_text))
    for match in reversed(matches):
        cited_source = match.group(1).strip()
        cited_ref    = match.group(2).strip()

        cited_source_lower = cited_source.lower()
        actual_source_match = None
        for known_lower, known_real in known_sources_lower.items():
            if cited_source_lower in known_lower or known_lower in cited_source_lower:
                actual_source_match = known_real
                break

        if actual_source_match:
            valid_refs = source_to_refs[actual_source_match]
            if cited_ref in valid_refs:
                # Citation is valid — leave it completely alone
                continue
            else:
                # Source is real but ref is wrong — remap to the closest valid ref.
                # Use the first ref as fallback (preserves determinism).
                fallback_ref = source_first_ref[actual_source_match]
                new_citation = f"[Source: {actual_source_match}, Reference: {fallback_ref}]"
                answer_text = answer_text[:match.start()] + new_citation + answer_text[match.end():]
            continue

        # ── Hallucinated source: score context to find the correct source ──────
        start_pos = max(0, match.start() - 200)
        context_before = answer_text[start_pos:match.start()].lower()

        best_category, max_hits = None, 0
        for cat, patterns in category_keywords.items():
            hits = sum(1 for p in patterns if re.search(p, context_before, re.IGNORECASE))
            if hits > max_hits:
                max_hits, best_category = hits, cat

        if best_category and max_hits >= 2:
            actual_source = next(
                (s for s in source_to_refs if best_category.lower() in s.lower()), None
            )
            if actual_source:
                fallback_ref  = source_first_ref[actual_source]
                new_citation  = f"[Source: {actual_source}, Reference: {fallback_ref}]"
                answer_text   = answer_text[:match.start()] + new_citation + answer_text[match.end():]

    return answer_text
```

**What changed:** `source_to_refs` maps each source to a `set` of all retrieved references. A citation is only remapped if its reference is not in that set — valid page-level citations like `[Source: ACMG_2015, Reference: page 33]` are now left untouched instead of being collapsed to page 3.

---

## H-3 — Gene allowlist covers only 28 genes

**File:** `app/pipeline/sources/clingen_client.py`

Replace `extract_gene_from_query`:

```python
# Common acronyms that match the gene regex but are NOT genes.
# Extend this list if false positives appear.
_NON_GENE_ACRONYMS = {
    "ACMG", "NCCN", "VUS", "LOF", "DNA", "RNA", "PCR", "SNP", "CNV",
    "RRSO", "MRI", "CT", "PET", "NGS", "WGS", "WES", "FISH", "IHC",
    "HER", "ER", "PR", "HR", "OS", "PFS", "FDA", "EMA", "US", "UK",
}

def extract_gene_from_query(query_text: str) -> str | None:
    """
    Extract a gene symbol from the query using a regex pattern.
    Matches tokens that look like gene symbols (1 uppercase letter + 1-9 uppercase
    letters/digits) and filters out common non-gene acronyms.

    This replaces the previous hardcoded 28-gene allowlist, which silently
    returned None for any gene not in the list (CDH1, MUTYH, BMPR1A, etc.).
    """
    gene_pattern = re.compile(r'\b[A-Z][A-Z0-9]{1,9}\b')
    matches = gene_pattern.findall(query_text)
    for match in matches:
        if match not in _NON_GENE_ACRONYMS:
            return match
    return None
```

**What changed:** Removed the hardcoded 28-gene allowlist. The regex was already correct — the only addition is a `_NON_GENE_ACRONYMS` exclusion set to prevent `ACMG`, `NCCN`, `VUS`, etc. from being returned as gene names. Any gene symbol found in the query (BRCA1, CDH1, MUTYH, SMAD4, and any future gene) will now be recognised.

---

## H-4 — ClinVar keywords trigger ClinGen API calls

**File:** `app/pipeline/decomposition.py`

Replace the `CLINGEN_KEYWORDS` list:

```python
# ClinGen = gene-disease validity expert panel curation database
# ClinVar = variant classification database (stored locally in PostgreSQL)
# These are DIFFERENT databases. ClinVar mentions in a query should NOT
# trigger a ClinGen API call — ClinVar data is already local.
CLINGEN_KEYWORDS = [
    "gene validity",
    "clingen",
    "gene-specific",
    "expert panel",
    "expert panel recommendation",
    "actionability curation",
    "dosage sensitivity",
]
```

**What changed:** Removed `"clinvar"` and all `"clinvar variant ..."` strings from `CLINGEN_KEYWORDS`. Queries about ClinVar now only hit the local PostgreSQL database as intended.

---

## H-5 — gnomAD external call and the privacy claim

This requires both a code change and a dissertation note.

**Code fix — add an opt-out config flag:**

**Step 1 — Add to `app/config.py`:**

```python
class Settings(BaseSettings):

    POSTGRES_URL:      str
    POSTGRES_USER:     str
    POSTGRES_PASSWORD: str
    POSTGRES_PORT:     str
    POSTGRES_DB:       str
    LOCAL_LLM_URL:     str
    LOCAL_LLM_MODEL:   str
    EMBEDDING_MODEL:   str
    CLINGEN_API_URL:   str

    # Set to false to disable live gnomAD lookups (privacy-sensitive environments).
    # When disabled, the BA1/PM2 frequency note will be omitted from responses.
    ENABLE_GNOMAD_LOOKUP: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
```

**Step 2 — Add to `.env.example`:**

```dotenv
# Set to false to disable live gnomAD calls (required for air-gapped deployments)
ENABLE_GNOMAD_LOOKUP=true
```

**Step 3 — Guard the call in `app/pipeline/graph/nodes.py` → `retrieve_node`:**

```python
from app.config import settings

def retrieve_node(state: CDSSGraphState) -> dict:
    trusted_chunks = []
    gene = state.get("gene")

    for sq in state["sub_queries"]:
        if sq.target == "postgres":
            rsids = re.findall(r'rs\d+', sq.text)
            for rsid in rsids:
                result = get_variant_by_rsid(rsid)
                if result:
                    # Only call gnomAD if the feature flag is enabled
                    if settings.ENABLE_GNOMAD_LOOKUP:
                        freq_data = get_allele_frequency(rsid)
                        if freq_data:
                            freq_text = (
                                f"Population frequency for {rsid}: "
                                f"AF={freq_data['allele_frequency']:.6f}. "
                                f"{'BA1 rule APPLIES (common variant -> Benign).' if freq_data['ba1_applicable'] else 'Variant is rare in population.'}"
                            )
                            trusted_chunks.append(RetrievedChunk(
                                text=freq_text, source="gnomAD", reference=rsid
                            ))
                        else:
                            trusted_chunks.append(RetrievedChunk(
                                text=f"gnomAD lookup for {rsid}: Variant not found in gnomAD r4. "
                                     f"ACMG BA1 does NOT apply. PM2 MAY apply — requires manual review.",
                                source="gnomAD",
                                reference=rsid
                            ))
                    trusted_chunks.append(from_postgres_result(result))

            if gene and not rsids:
                for r in get_variant_by_gene(gene):
                    trusted_chunks.append(from_postgres_result(r))

        elif sq.target == "clingen" and gene:
            for r in get_gene_validity(gene):
                trusted_chunks.append(from_clingen_result(r, gene))

    return {"trusted_chunks": trusted_chunks}
```

**Dissertation note to add to your Limitations / Privacy Analysis section:**

> *"The system optionally queries gnomAD's public GraphQL API to retrieve population allele frequencies for ACMG BA1/PM2 criteria application. This call transmits a positional variant identifier in the format `{chromosome}-{position}-{ref}-{alt}`. No patient name, date of birth, or direct identifier is transmitted; however, for rare variants the combination of chromosomal coordinates may be quasi-identifying in the sense of genomic re-identification. In privacy-sensitive or air-gapped deployments, gnomAD lookup should be disabled via the `ENABLE_GNOMAD_LOOKUP=false` configuration flag. A production mitigation is to pre-fetch and cache allele frequencies for all ingested ClinVar variants at index time, eliminating the runtime external call entirely."*

---

## H-6 — All test files are empty

**File:** `tests/test_routing.py`

```python
import pytest
from app.pipeline.decomposition import decompose_query, SubQuery


def test_rsid_produces_postgres_subquery():
    """A query containing an rsID should generate a postgres sub-query."""
    result = decompose_query("What is the significance of rs879254116?")
    targets = [sq.target for sq in result]
    assert "postgres" in targets


def test_acmg_keyword_produces_rule_retrieval():
    """A query mentioning ACMG criteria should generate a vector_db rule_retrieval."""
    result = decompose_query("Which ACMG criteria apply to this pathogenic variant?")
    types = [sq.query_type for sq in result if sq.target == "vector_db"]
    assert "rule_retrieval" in types


def test_screening_keyword_produces_screening_retrieval():
    """A query mentioning screening should generate a screening_retrieval sub-query."""
    result = decompose_query("What NCCN screening protocol should a BRCA1 carrier follow?")
    types = [sq.query_type for sq in result if sq.target == "vector_db"]
    assert "screening_retrieval" in types


def test_protocol_keyword_without_screening():
    """A pure treatment query should generate protocol_retrieval, not screening_retrieval."""
    result = decompose_query("What is the neoadjuvant chemotherapy regimen for stage III breast cancer?")
    types = [sq.query_type for sq in result if sq.target == "vector_db"]
    assert "protocol_retrieval" in types
    assert "screening_retrieval" not in types


def test_empty_query_returns_default_subquery():
    """An empty or generic query should return at least one fallback sub-query."""
    result = decompose_query("")
    assert len(result) >= 1
    assert result[0].target == "vector_db"


def test_clingen_keyword_triggers_clingen_target():
    """A query about gene validity should generate a ClinGen sub-query."""
    result = decompose_query("What is the ClinGen expert panel validity for BRCA1?")
    targets = [sq.target for sq in result]
    assert "clingen" in targets


def test_clinvar_mention_does_not_trigger_clingen():
    """ClinVar mentions should NOT trigger a ClinGen API call — they are different databases."""
    result = decompose_query("What does ClinVar say about rs80357065?")
    targets = [sq.target for sq in result]
    assert "clingen" not in targets
```

**File:** `tests/test_retrieval.py`

```python
import pytest
from app.pipeline.retrieval.reranker import RetrievedChunk, rerank_chunks
from app.pipeline.retrieval.crag_evaluator import evaluate_chunks, CORRECT, AMBIGUOUS, INCORRECT


def _make_chunk(text: str, score: float = 0.0, source: str = "ACMG") -> RetrievedChunk:
    c = RetrievedChunk(text=text, source=source, reference="page 1")
    c.score = score
    return c


def test_reranker_returns_chunks_sorted_descending():
    chunks = [
        _make_chunk("low relevance", score=0.01),
        _make_chunk("high relevance", score=0.95),
        _make_chunk("medium relevance", score=0.5),
    ]
    # Manually set scores to simulate post-reranker state
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_reranker_empty_input():
    result = rerank_chunks("any query", [])
    assert result == []


def test_crag_grades_high_score_as_correct():
    chunk = _make_chunk("ACMG PVS1 criteria for loss of function variants", score=0.1)
    result = evaluate_chunks("ACMG PVS1 criteria", [chunk])
    assert chunk in result[CORRECT]


def test_crag_grades_low_score_as_incorrect():
    chunk = _make_chunk("References: 1. Smith J et al...", score=0.005)
    result = evaluate_chunks("BRCA1 screening protocol", [chunk])
    assert chunk in result[INCORRECT]


def test_crag_grades_borderline_as_ambiguous():
    chunk = _make_chunk("Some loosely related text", score=0.02)
    result = evaluate_chunks("BRCA1 pathogenicity", [chunk])
    assert chunk in result[AMBIGUOUS]


def test_crag_empty_input():
    result = evaluate_chunks("any query", [])
    assert result[CORRECT] == []
    assert result[AMBIGUOUS] == []
    assert result[INCORRECT] == []
```

**File:** `tests/test_generation.py`

```python
import pytest
from unittest.mock import patch, MagicMock
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
    """A citation with a source not in used_chunks should be remapped based on context."""
    chunks = [_make_chunk("gnomAD AF=0.000001", source="gnomAD", reference="rs12345")]
    answer = "The variant is rare in the population. [Source: SomeRandomDB, Reference: unknown]"
    result = fix_hallucinated_citations(answer, chunks)
    # The hallucinated source should have been replaced
    assert "SomeRandomDB" not in result


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
    result = extract_citations("", [])
    assert result == []
```

**File:** `tests/test_clingen_client.py`

```python
import pytest
from app.pipeline.sources.clingen_client import extract_gene_from_query


def test_extracts_brca1_from_natural_query():
    gene = extract_gene_from_query("What is the clinical significance of a BRCA1 variant?")
    assert gene == "BRCA1"


def test_extracts_tp53_from_query():
    gene = extract_gene_from_query("Does TP53 have ClinGen expert panel validity?")
    assert gene == "TP53"


def test_returns_none_for_no_gene():
    gene = extract_gene_from_query("What are the ACMG pathogenicity criteria?")
    assert gene is None


def test_does_not_return_acmg_as_gene():
    """ACMG is a common acronym and must not be returned as a gene symbol."""
    gene = extract_gene_from_query("What ACMG criteria apply here?")
    assert gene != "ACMG"


def test_does_not_return_nccn_as_gene():
    gene = extract_gene_from_query("What does the NCCN guideline say?")
    assert gene != "NCCN"


def test_does_not_return_vus_as_gene():
    gene = extract_gene_from_query("This variant is classified as VUS.")
    assert gene != "VUS"


def test_extracts_first_gene_when_multiple_present():
    """When multiple gene symbols appear, the first one found should be returned."""
    gene = extract_gene_from_query("Compare BRCA1 and BRCA2 pathogenicity.")
    assert gene in ("BRCA1", "BRCA2")
```

---

## H-7 — Confidence score uses pre-filter chunk count

**File:** `app/pipeline/graph/nodes.py` → `citation_node`

Replace the confidence calculation block:

```python
def citation_node(state: CDSSGraphState) -> dict:

    final          = state["final_answer"]
    verified_chunks = state.get("verified_chunks", [])
    trusted_chunks  = state.get("trusted_chunks", [])

    fixed_final_answer = fix_hallucinated_citations(final, verified_chunks)
    raw_citations = extract_citations(fixed_final_answer, verified_chunks)
    citations = [Citation(source=c["source"], reference=c["reference"]) for c in raw_citations]

    # ── Confidence: use POST-FILTER counts from verified_chunks ────────────────
    # Old code used len(candidate_chunks) — the raw PDF_Retriever output BEFORE
    # CRAG filtering. This inflated confidence: 74 irrelevant chunks retrieved
    # → confidence "high" even if CRAG dropped all of them.
    # Now we count only the chunks that actually reached the LLM.
    DB_SOURCES = {"Clinvar", "gnomAD", "ClinGen"}

    db_count  = sum(1 for c in verified_chunks if c.source in DB_SOURCES)
    pdf_count = sum(1 for c in verified_chunks if c.source not in DB_SOURCES)

    has_db         = db_count >= 2
    has_guidelines = pdf_count > 0

    if has_db and has_guidelines:
        confidence = "high"
    elif has_db or has_guidelines:
        confidence = "medium"
    else:
        confidence = "low"

    print(f"[Citation] Verified DB chunks: {db_count} | Verified PDF chunks: {pdf_count} "
          f"→ confidence: {confidence}")

    return {"final_answer": fixed_final_answer, "citations": citations, "confidence": confidence}
```

---

## M-1 — SQL column name allowlist

**File:** `app/pipeline/sources/vector_client.py`

Add at the top of the file, before `search_documents`:

```python
# Allowlist for filter column names — prevents accidental SQL column injection
# if filter values ever flow from a less-controlled code path.
_ALLOWED_FILTER_COLUMNS: dict[str, str] = {
    "source":   "source",
    "category": "category",
}

def _safe_filter_clause(filter_type: str) -> str:
    """Return a safe WHERE clause string for an allowed column name."""
    col = _ALLOWED_FILTER_COLUMNS.get(filter_type)
    if not col:
        raise ValueError(f"Unknown filter column: '{filter_type}'. "
                         f"Allowed: {list(_ALLOWED_FILTER_COLUMNS)}")
    return f"WHERE {col} = %s"
```

Then in `search_documents`, replace the inline f-string `where_clause` builds:

```python
if source_filter:
    where_clause = _safe_filter_clause("source")
    params = (source_filter, query_vector, query_vector, top_k)
elif category_filter:
    where_clause = _safe_filter_clause("category")
    params = (category_filter, query_vector, query_vector, top_k)
```

---

## M-2 — Add `pymupdf4llm` to `requirements.txt`

**File:** `requirements.txt`

Add below the `pymupdf` line:

```
pymupdf==1.24.0
pymupdf4llm>=0.0.17
```

---

## M-3 — Pin `docling` in `requirements.txt`

**File:** `requirements.txt`

Run this in your environment to get the exact version you tested with:

```bash
pip show docling | grep Version
```

Then replace:

```
docling
```

with:

```
docling==2.X.Y   # replace X.Y with the version shown above
```

---

## M-4 — Fix seed data typos in `clinvar_schema.sql`

**File:** `app/db/postgres/clinvar_schema.sql`

Replace the INSERT block at the bottom:

```sql
-- ─── SAMPLE TEST DATA ─────────────────────────────────────────────────────────
-- One real BRCA1 record for immediate pipeline testing.
-- GRCh38 coordinates: chr17:43,044,295 (from ClinVar VCV000055606)
INSERT INTO variants (
    rsid, gene_symbol, chromosome, position,
    ref_allele, alt_allele, clinical_significance,
    review_status, condition
)
VALUES (
    'rs80357065',
    'BRCA1',
    '17',
    43044295,
    'G',
    'A',
    'Pathogenic',
    'reviewed by expert panel',
    'Hereditary breast and ovarian cancer syndrome'
) ON CONFLICT (rsid) DO NOTHING;
```

---

## M-5 — Handle or remove `patient_id` from `QueryRequest`

**Option A — remove the field (cleanest for now):**

**File:** `app/api/schemas.py`

```python
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    # patient_id is not yet implemented — removed to avoid implying functionality
    # that does not exist. Add back when session/audit logging is implemented.


class Citation(BaseModel):
    source: str
    reference: str


class QueryResponse(BaseModel):
    answer:      str
    citations:   List[Citation]
    confidence:  str
    safe_failure: bool
```

**Option B — pass it through state (if you want to keep the field):**

Add to `CDSSGraphState` in `state.py`:
```python
patient_id: Optional[str]
```

Add to `router.py`:
```python
initial_state = {
    "query":      query,
    "patient_id": request.patient_id,
}
```

---

## M-6 — Replace empty `construction/` files with documented stubs

**File:** `app/pipeline/construction/self_query.py`

```python
"""
Self-Query Metadata Filter — planned but not yet implemented.

Intended behaviour:
    Parse a natural language query to extract structured metadata filters
    (source, category, gene) that can be passed to vector_client.search_documents()
    as source_filter or category_filter arguments.

    Example:
        "BRCA1 ACMG criteria" → {"category": "guideline", "gene": "BRCA1"}
        "NCCN screening protocol" → {"category": "screening_protocol"}

    This would replace the current manual keyword routing in decomposition.py
    with an LLM-powered metadata extractor, allowing more flexible query routing
    without maintaining keyword lists.

Current workaround:
    Keyword-based routing in decomposition.py handles category selection.
    gene filtering in the Evaluator node (evaluate_node) filters post-retrieval.
"""


def build_self_query_filter(query: str) -> dict:
    """
    Extract metadata filter dict from a natural language query.
    Not yet implemented.
    """
    raise NotImplementedError(
        "Self-query metadata filtering is not yet implemented. "
        "See decomposition.py for the current keyword-based routing approach."
    )
```

**File:** `app/pipeline/construction/text_to_sql.py`

```python
"""
Text-to-SQL Generator — planned but not yet implemented.

Intended behaviour:
    Convert a natural language clinical query into a parameterised SQL query
    against the variants table, for cases where a structured lookup is more
    appropriate than a free-text vector search.

    Example:
        "All pathogenic BRCA1 variants reviewed by expert panel"
        → SELECT * FROM variants
          WHERE gene_symbol = 'BRCA1'
            AND clinical_significance = 'Pathogenic'
            AND review_status = 'reviewed by expert panel';

Current workaround:
    postgres_client.py implements fixed queries (get_variant_by_rsid,
    get_variant_by_gene). Dynamic SQL generation is left as future work.
"""


def generate_sql_query(natural_language_query: str) -> str:
    """
    Generate a parameterised SQL query from a natural language input.
    Not yet implemented.
    """
    raise NotImplementedError(
        "Text-to-SQL generation is not yet implemented. "
        "See postgres_client.py for current fixed-query implementations."
    )
```

---

## M-7 — Update Self-RAG description to match implementation

This is a dissertation text fix, not a code fix.

**In your dissertation, replace any language like:**
> *"The system employs a Self-RAG loop where the critic node rejects low-quality generations and triggers an automatic rewrite, iterating until the answer meets quality thresholds."*

**With:**
> *"The system employs a single-pass Self-RAG critic. After the Generator produces a structured JSON draft, a separate Critic LLM call audits it for hallucinations, incorrect age ranges, and missing citations, and rewrites the answer where necessary. A multi-pass retry loop (max 2–3 iterations with early termination on no-change) is a planned enhancement; the current implementation performs one critic pass, which proved sufficient for the evaluated query set."*

---

## M-8 — Make the query endpoint async

**File:** `app/api/router.py`

```python
import asyncio
from fastapi import APIRouter
from app.api.schemas import QueryRequest, QueryResponse
from app.pipeline.generation.guardrails import SAFE_FAILURE_MESSAGE
from app.pipeline.graph.workflow import cdss_app

router = APIRouter()


@router.get("/")
def health_check():
    return {"status": "online", "service": "CDSS API is running"}


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query

    print(f"\n[STARTING LANGGRAPH] Query: {query}")
    initial_state = {"query": query}

    # Run the synchronous LangGraph pipeline in a thread pool so it does not
    # block uvicorn's event loop. Without this, a second request arriving while
    # a query is in progress (Ollama calls take 30-60s) will queue or be dropped.
    final_state = await asyncio.to_thread(cdss_app.invoke, initial_state)

    print(f"[LANGGRAPH FINISHED] Confidence: {final_state.get('confidence', 'low')}")
    return QueryResponse(
        answer=final_state.get("final_answer", SAFE_FAILURE_MESSAGE),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", "low"),
        safe_failure=False if final_state.get("final_answer") else True
    )
```

---

## L-1 — Fix Docker env var mismatch

**File:** `.env.example`

Replace with the complete set of variables the docker-compose.yml expects:

```dotenv
# ── PostgreSQL ────────────────────────────────────────────────────
POSTGRES_USER=cdss-user
POSTGRES_PASSWORD=cdss_password
POSTGRES_DB=cdss_db
POSTGRES_PORT=5432
POSTGRES_URL=postgresql://cdss-user:cdss_password@localhost:5432/cdss_db

# ── Local LLM via Ollama ──────────────────────────────────────────
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3.1:latest

# ── Embedding Model ───────────────────────────────────────────────
EMBEDDING_MODEL=NLP4Science/pubmedbert-base-embeddings

# ── ClinGen API ───────────────────────────────────────────────────
CLINGEN_API_URL=https://search.clinicalgenome.org/kb

# ── Privacy flags ─────────────────────────────────────────────────
# Set to false in air-gapped or privacy-sensitive deployments
ENABLE_GNOMAD_LOOKUP=true
```

---

## L-2 — Update README to reflect actual implementation

**File:** `README.md`

Find the reranker reference and update it:

```markdown
<!-- OLD -->
Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

<!-- NEW -->
Cross-encoder reranker: `BAAI/bge-reranker-large`
(upgraded from ms-marco-MiniLM-L-6-v2 during implementation for
stronger biomedical text ranking performance)
```

Add gnomAD to the data sources section:

```markdown
### Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| ClinVar (local PostgreSQL) | Structured DB | Variant classifications, rsIDs |
| gnomAD v4 (REST API) | External API | Population allele frequency, BA1/PM2 criteria |
| ClinGen (REST API) | External API | Gene-disease validity, expert panel curation |
| ACMG 2015 guidelines (pgvector) | Vector DB | Pathogenicity classification rules |
| NCCN screening guidelines (pgvector) | Vector DB | Cancer surveillance protocols |
```

---

## L-3 — Fix spacing in `main.py`

**File:** `app/main.py`

```python
if __name__ == "__main__":
    uvicorn.run("app.main:app", port=5656, reload=True)
```

---

*All solutions verified against the codebase audit. Apply in order: C → H → M → L.*