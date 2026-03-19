# CDSS — Known Problems & Fix Tracker

> Generated from deep code audit. Work through these in order — Critical first, then High, then Medium/Low.
> Check off each item as you fix it.

---

## Legend

| Priority | Meaning |
|----------|---------|
| 🔴 Critical | Breaks functionality or produces wrong output right now |
| 🟠 High | Serious correctness / safety / academic risk |
| 🟡 Medium | Production gaps, fragile patterns, missing pieces |
| 🔵 Low | Polish, robustness, minor correctness |

---

## 🔴 Critical

---

### C-1 — SQL parameter order mismatch in `vector_client.py`

- **File:** `app/pipeline/sources/vector_client.py`
- **Status:** [ ] Open

**Problem:**
When `filter_param` is set, the SQL has three placeholders (`%s`, `%s`, `%s`) but `params` is built as a 4-tuple — and in the wrong order. The `WHERE category = %s` placeholder expects the filter value first, but `params` puts `query_vector` at index 0.

```python
# Current (WRONG) — params tuple doesn't match SQL placeholder order
sql = f"""
    SELECT ... FROM medical_documents
    {where_clause}           -- WHERE category = %s  ← 1st placeholder
    ORDER BY embedding <=> %s::vector  ← 2nd placeholder
    LIMIT %s;                          ← 3rd placeholder
"""
params = (query_vector, filter_param, query_vector, top_k)  # 4 values, wrong order
```

**Fix:**
```python
# Correct — match placeholder order exactly
params = (filter_param, query_vector, top_k)
```

Every filtered vector search (ACMG, NCCN, screening) hits this path. This raises a `psycopg2.ProgrammingError` on any real query with a `category_filter`.

---

### C-2 — Critic node strips citations (`self_rag.py`)

- **File:** `app/pipeline/generation/self_rag.py` → `self_rag_critic()`
- **Status:** [ ] Open

**Problem:**
The Generator uses `format=ClinicalResponse.model_json_schema()` (grammar-constrained JSON), guaranteeing structured citations. The Critic uses free-text generation with no `format=` constraint. RESULT.txt shows the Critic reduced the answer from 1588 → 1048 chars, dropping inline citations. The citation enforcer then cannot fix what was never there.

**Fix option A — apply the same JSON schema to the Critic:**
```python
response = ollama.chat(
    model=settings.LOCAL_LLM_MODEL,
    messages=[...],
    format=ClinicalResponse.model_json_schema()  # add this
)
# then re-parse and re-render just like generate_answer()
```

**Fix option B (simpler) — make the Critic citation-aware in the prompt, or skip it for now:**
Add an explicit instruction and a regex post-check: if the critic output has fewer `[Source:` tags than the draft, return the draft unchanged.

```python
draft_citation_count = draft_answer.count("[Source:")
final_citation_count = final_answer.count("[Source:")
if final_citation_count < draft_citation_count:
    log.warning("Critic dropped citations — reverting to draft")
    return draft_answer
```

---

### C-3 — LangGraph parallel fan-out not actually parallel (`workflow.py`)

- **File:** `app/pipeline/graph/workflow.py`
- **Status:** [ ] Open

**Problem:**
```python
workflow.add_edge("Decomposer", "DB_Retriever")
workflow.add_edge("Decomposer", "PDF_Retriever")
```
Two `add_edge` calls from the same source node do **not** create true parallel execution in LangGraph. The nodes run sequentially, and the second node's state update may overwrite the first's depending on the TypedDict merge. You are not getting concurrent execution and risk silent state loss.

**Fix — use LangGraph's `Send` API for fan-out:**
```python
from langgraph.constants import Send

def fan_out(state: CDSSGraphState):
    return [
        Send("DB_Retriever", state),
        Send("PDF_Retriever", state),
    ]

workflow.add_conditional_edges("Decomposer", fan_out, ["DB_Retriever", "PDF_Retriever"])
```

Also ensure `CDSSGraphState` uses `Annotated` list fields with a merge reducer so both nodes' results accumulate rather than overwrite:
```python
from typing import Annotated
import operator

class CDSSGraphState(TypedDict):
    trusted_chunks:   Annotated[list, operator.add]
    candidate_chunks: Annotated[list, operator.add]
    # ... other fields unchanged
```

---

### C-4 — No database connection pooling anywhere

- **Files:** `app/pipeline/sources/postgres_client.py`, `vector_client.py`, `gnomad_client.py`
- **Status:** [ ] Open

**Problem:**
Every function calls `psycopg2.connect()` and closes immediately in a `finally`. The indexing pipeline opens a new connection per chunk insert. Under any concurrent load this exhausts PostgreSQL's connection limit. Indexing large documents is also significantly slower than it needs to be.

**Fix — add a module-level pool in each client:**
```python
# At the top of postgres_client.py (and vector_client.py)
from psycopg2 import pool as pg_pool
from app.config import settings

_pool = pg_pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=settings.POSTGRES_URL)

def get_connection():
    return _pool.getconn()

def release_connection(conn):
    _pool.putconn(conn)
```
Then replace `psycopg2.connect()` with `get_connection()` and `conn.close()` with `release_connection(conn)` throughout. Use a try/finally to always release.

---

## 🟠 High

---

### H-1 — CRAG thresholds are a no-op (all chunks pass as CORRECT)

- **File:** `app/pipeline/retrieval/crag_evaluator.py`
- **Status:** [ ] Open

**Problem:**
```python
CORRECT_THRESHOLD  = 0.0
AMBIGUOUS_THRESHOLD = -3.0
```
In the live RESULT.txt, all 57 chunks scored between 0.001 and 0.195 — every single one was graded `CORRECT`. The AMBIGUOUS and INCORRECT branches never fire. Bibliography entries, Li-Fraumeni syndrome pages, and References sections all pass through to the LLM context.

**Fix:**
Empirically tune the thresholds by examining score distributions across a sample of queries. As a starting point based on the BGE score distribution in RESULT.txt:

```python
CORRECT_THRESHOLD   = 0.05   # meaningful relevance floor
AMBIGUOUS_THRESHOLD = 0.01   # borderline — keep but deprioritise
# anything < 0.01 → INCORRECT → dropped
```

For the dissertation, document why you chose these values (e.g. "set to retain the top N% of scored chunks from pilot queries"). Also add a log line showing how many chunks were actually dropped.

---

### H-2 — Citation enforcer collapses all references to first-chunk ref

- **File:** `app/pipeline/generation/citation_enforcer.py`
- **Status:** [ ] Open

**Problem:**
```python
source_to_ref = {}
for chunk in used_chunks:
    if chunk.source not in source_to_ref:  # first-wins — ignores all others
        source_to_ref[chunk.source] = chunk.reference
```
If 12 ACMG chunks were retrieved (pages 3, 9, 14, 33, 34, etc.), ALL inline citations `[Source: ACMG_2015, Reference: ...]` get remapped to whatever page was first in the list. Page-level citation granularity, which was the whole point of the metadata headers, is destroyed.

**Fix — build a set of valid references per source and only remap if the cited reference isn't in the set:**
```python
# Build: source → set of valid references
source_to_refs: dict[str, set] = {}
for chunk in used_chunks:
    source_to_refs.setdefault(chunk.source, set()).add(chunk.reference)

# When validating a citation: only remap if the cited ref is not in the valid set
if cited_ref not in source_to_refs.get(actual_source_match, set()):
    # pick the closest valid ref, or the first one as fallback
    fallback_ref = next(iter(source_to_refs[actual_source_match]))
    new_citation = f"[Source: {actual_source_match}, Reference: {fallback_ref}]"
```

---

### H-3 — Gene allowlist covers only 28 genes

- **File:** `app/pipeline/sources/clingen_client.py` → `extract_gene_from_query()`
- **Status:** [ ] Open

**Problem:**
`known_genes` contains 28 hardcoded symbols. Any query about `CDH1`, `MUTYH`, `BMPR1A`, `SMAD4`, `NBN`, `EPCAM`, or hundreds of other actionable genes returns `None`. The postgres sub-query then gets `gene=None`, finds nothing, and the user gets no variant data with no warning.

**Fix option A — expand the allowlist** from a curated source (ClinGen's gene list, OMIM, etc.)

**Fix option B — remove the allowlist entirely** and rely on the regex pattern alone:
```python
# The regex r'\b[A-Z][A-Z0-9]{1,9}\b' already finds all-caps gene-like tokens
# Return the first match that is not a common English acronym
COMMON_ACRONYMS = {"ACMG", "NCCN", "MRI", "VUS", "LOF", "DNA", "RNA", "PCR"}
for match in matches:
    if match not in COMMON_ACRONYMS:
        return match
return None
```

Fix option B is simpler and more complete. Add the acronym exclusion set to avoid false positives.

---

### H-4 — `ClinVar` keywords incorrectly trigger ClinGen API calls

- **File:** `app/pipeline/decomposition.py` → `CLINGEN_KEYWORDS`
- **Status:** [ ] Open

**Problem:**
```python
CLINGEN_KEYWORDS = ["clinvar", "clinvar variant", ...]
```
ClinVar and ClinGen are different databases. A query like "what does ClinVar say about BRCA1?" triggers a ClinGen REST API call. This is an unnecessary external network request, and it conflates two distinct data sources in the dissertation's architecture description.

**Fix:**
```python
# Remove ClinVar from ClinGen trigger list entirely
CLINGEN_KEYWORDS = [
    "gene validity", "clingen", "gene-specific",
    "expert panel", "expert panel recommendation"
]
# ClinVar data is already in local PostgreSQL — no keyword needed to trigger it
```

---

### H-5 — gnomAD external call breaks the privacy-preserving claim

- **File:** `app/pipeline/sources/gnomad_client.py`
- **Status:** [ ] Open

**Problem:**
The system sends `{chrom}-{pos}-{ref}-{alt}` to `gnomad.broadinstitute.org` — a public external API. For rare or private variants, genomic coordinates can re-identify a patient. Your dissertation title is "Privacy-Preserving CDSS" and you have no threat model entry for this.

**Fixes (choose one):**

1. **Cache gnomAD data locally at index time** — for any rsID you ingest into ClinVar, pre-fetch and store the allele frequency in the `variants` table as an `allele_frequency FLOAT` column. No runtime external calls needed.
2. **Document the limitation explicitly** — add a "Limitations" section to the dissertation: "gnomAD population frequency lookup requires a live external call; local caching is left as future work. No patient identifiers are transmitted — only positional variant IDs."
3. **Make it opt-in** — add a config flag `ENABLE_GNOMAD_LOOKUP=false` in `.env.example` and skip the call when false.

Option 3 is the quickest fix for the dissertation. Option 1 is the correct production fix.

---

### H-6 — All test files are empty

- **Files:** `tests/test_routing.py`, `tests/test_retrieval.py`, `tests/test_generation.py`, `tests/test_clingen_client.py`
- **Status:** [ ] Open

**Problem:**
Every test file has zero content. For a dissertation claiming a "verified" clinical decision support system, an examiner will open these files. This is the single biggest academic presentation risk.

**Minimum viable tests to write per module:**

`test_routing.py` — `decompose_query()`:
- Happy path: query with rsID produces a `postgres` sub-query
- Happy path: query with "acmg" keyword produces a `rule_retrieval` sub-query
- Happy path: query with "screening" produces a `screening_retrieval` sub-query
- Edge case: empty string returns a default vector sub-query

`test_retrieval.py` — `rerank_chunks()`, `evaluate_chunks()`:
- Reranker returns chunks sorted by score descending
- CRAG grades a high-score chunk as CORRECT
- CRAG grades a negative-score chunk as INCORRECT
- Empty input returns empty output

`test_generation.py` — `generate_answer()`, `safe_failure`:
- Empty `verified_chunks` returns `SAFE_FAILURE_MESSAGE`
- Output contains `[Source:` citation markers
- `fix_hallucinated_citations()` leaves valid citations unchanged

`test_clingen_client.py` — `extract_gene_from_query()`:
- Recognises BRCA1 in a natural query
- Returns None for a query with no gene symbol
- Does not return acronyms like "ACMG" as a gene

---

### H-7 — Confidence score uses pre-filter chunk count

- **File:** `app/pipeline/graph/nodes.py` → `citation_node()`
- **Status:** [ ] Open

**Problem:**
```python
pdf_count = len(candidate_chunks)   # raw PDF_Retriever output — BEFORE CRAG filtering
```
A query that retrieves 74 irrelevant PDF chunks will register `has_guidelines = True` and push confidence to `"high"`, even if CRAG dropped all of them. The confidence metric measures retrieval volume, not answer quality.

**Fix:**
```python
# Use post-filter count — chunks that actually reached the LLM
db_count  = sum(1 for c in trusted_chunks if c.source in DB_SOURCES)
pdf_count = sum(1 for c in verified_chunks if c.source not in DB_SOURCES)
```
Also add a note to the dissertation: "Confidence reflects retrieval coverage, not answer faithfulness. A faithfulness-based metric (e.g. NLI entailment score against retrieved context) is left as future work."

---

## 🟡 Medium

---

### M-1 — SQL column name interpolation (injection-adjacent pattern)

- **File:** `app/pipeline/sources/vector_client.py`
- **Status:** [ ] Open

**Problem:**
```python
where_clause = "WHERE source = %s"      # column name is hardcoded string
where_clause = "WHERE category = %s"    # safe today — but fragile
```
The column name itself is an f-string interpolation (not parameterized). If `category_filter` ever flows from user input or a new code path, this becomes a SQL injection vector.

**Fix — use an explicit allowlist:**
```python
ALLOWED_FILTERS = {
    "source":   "source",
    "category": "category",
}

def _build_where(filter_type: str) -> str:
    col = ALLOWED_FILTERS.get(filter_type)
    if not col:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return f"WHERE {col} = %s"
```

---

### M-2 — `pymupdf4llm` missing from `requirements.txt`

- **File:** `requirements.txt`
- **Status:** [ ] Open

**Problem:**
`indexing.py` imports `pymupdf4llm` at the top level, but it is not listed in `requirements.txt`. A fresh `pip install -r requirements.txt` will succeed, but running the indexing pipeline will immediately fail with `ModuleNotFoundError`.

**Fix:**
```
pymupdf4llm>=0.0.17
```
Add to `requirements.txt` alongside `pymupdf==1.24.0`.

---

### M-3 — `docling` unpinned in `requirements.txt`

- **File:** `requirements.txt`
- **Status:** [ ] Open

**Problem:**
```
docling          # no version pin
```
Every other dependency is exactly pinned. `docling` has heavy transitive dependencies (torch, transformers) and breaking changes between versions.

**Fix:**
```
docling==2.15.1  # or whatever version you tested with — pin it
```
Run `pip show docling` in your working environment to find the current version.

---

### M-4 — Typo in seed data (`clinvar_schema.sql`)

- **File:** `app/db/postgres/clinvar_schema.sql`
- **Status:** [ ] Open

**Problem:**
```sql
condition = 'Heredetitotry breast and ovarian cancer syndorme'
-- Also: chromosome = '14' for BRCA1 (should be '17')
-- Also: position = 2432344 (BRCA1 is at ~43,044,295 on GRCh38)
```

**Fix:**
```sql
INSERT INTO variants (rsid, gene_symbol, chromosome, position, ref_allele, alt_allele,
                      clinical_significance, review_status, condition)
VALUES (
    'rs123424454', 'BRCA1', '17', 43044295, 'A', 'T',
    'Pathogenic', 'reviewed by expert panel',
    'Hereditary breast and ovarian cancer syndrome'
) ON CONFLICT (rsid) DO NOTHING;
```

---

### M-5 — `patient_id` is accepted by the API but silently discarded

- **File:** `app/api/schemas.py`, `app/api/router.py`
- **Status:** [ ] Open

**Problem:**
`QueryRequest` has `patient_id: Optional[str]` but `router.py` does `query = request.query` and never touches `patient_id`. For a privacy-preserving system this is both an unimplemented feature and a latent privacy risk if the field is ever stored.

**Fix option A — remove the field until it's implemented:**
```python
class QueryRequest(BaseModel):
    query: str
    # patient_id removed — not yet implemented
```

**Fix option B — pass it through state and log it (without storing):**
```python
initial_state = {
    "query": query,
    "patient_id": request.patient_id  # add to CDSSGraphState too
}
```
Then add `patient_id: Optional[str]` to `CDSSGraphState`. Even if unused downstream, it documents intent and makes future implementation straightforward.

---

### M-6 — `construction/` package is empty dead code

- **Files:** `app/pipeline/construction/self_query.py`, `app/pipeline/construction/text_to_sql.py`
- **Status:** [ ] Open

**Problem:**
Both files are completely empty (0 bytes). The package exists with an `__init__.py`, implying a planned self-query metadata filtering and text-to-SQL pathway that was never implemented.

**Fix:**
Either implement the stubs with at minimum a `NotImplementedError` and a docstring explaining the planned behaviour, or delete the files and remove the package. Empty files in a dissertation project look like abandoned work.

```python
# self_query.py — minimal stub
def build_self_query_filter(query: str) -> dict:
    """
    Planned: extract metadata filters (source, category, gene) from natural language
    to narrow vector search before embedding lookup.
    Not yet implemented — vector_client.py category_filter handles this manually.
    """
    raise NotImplementedError
```

---

### M-7 — Self-RAG described as a retry loop but implemented as a single pass

- **File:** `app/pipeline/generation/self_rag.py`, README
- **Status:** [ ] Open

**Problem:**
The README and architecture description imply a "Reject → Rewrite" loop with automatic retries. The actual `self_rag_critic()` function makes one LLM call and returns. There is no loop, no `max_retries`, no termination condition.

**Fix option A — implement the loop (2–3 max iterations):**
```python
MAX_CRITIC_PASSES = 2

def self_rag_critic(query, draft_answer, verified_chunks) -> str:
    current = draft_answer
    for attempt in range(MAX_CRITIC_PASSES):
        revised = _run_critic_pass(query, current, verified_chunks)
        if revised == current:
            break   # no change — stop
        current = revised
    return current
```

**Fix option B — update the dissertation description** to accurately say "single-pass critic" rather than "retry loop." This is quicker and honest.

---

### M-8 — Sync FastAPI endpoint blocks uvicorn thread pool

- **File:** `app/api/router.py`
- **Status:** [ ] Open

**Problem:**
```python
@router.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):  # sync def
    final_state = cdss_app.invoke(initial_state)  # blocks for 30–60s
```
The entire LangGraph pipeline (Ollama calls + DB queries) runs synchronously in the request thread. Any second request while a query is running will be queued or rejected.

**Fix:**
```python
import asyncio

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    final_state = await asyncio.to_thread(cdss_app.invoke, initial_state)
```

---

## 🔵 Low

---

### L-1 — Docker healthcheck env var mismatch

- **File:** `docker-compose.yml`, `.env.example`
- **Status:** [ ] Open

**Problem:**
The healthcheck uses `pg_isready -U ${POSTGRES_USER}` but `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` are not in `.env.example` (only `POSTGRES_URL` is shown). A fresh deployment will fall back to the `-cdss_user` default and the healthcheck may fail if the container was initialized with different credentials.

**Fix — add the missing vars to `.env.example`:**
```dotenv
POSTGRES_USER=cdss-user
POSTGRES_PASSWORD=cdss_password
POSTGRES_DB=cdss_db
POSTGRES_PORT=5432
POSTGRES_URL=postgresql://cdss-user:cdss_password@localhost:5432/cdss_db
```

---

### L-2 — BGE reranker upgrade and gnomAD integration are undocumented

- **File:** `README.md`, dissertation
- **Status:** [ ] Open

**Problem:**
The README states `ms-marco-MiniLM-L-6-v2` but the code uses `BAAI/bge-reranker-large`. gnomAD integration is implemented and working but mentioned nowhere in the README or architecture diagram.

These are genuine improvements over what was planned — they should be foregrounded, not hidden.

**Fix — update README to reflect actual implementation:**
- Change reranker reference to `BAAI/bge-reranker-large`
- Add gnomAD as a fourth data source in the architecture description
- Note in the dissertation that the reranker was upgraded during implementation for better biomedical text ranking

---

### L-3 — `uvicorn.run()` call has inconsistent spacing (`main.py`)

- **File:** `app/main.py`
- **Status:** [ ] Open

**Problem:**
```python
uvicorn.run ("app.main:app", port=5656, reload= True)
#            ↑ space before ( )                ↑ space before =
```
Minor style issue but inconsistent with the rest of the codebase.

**Fix:**
```python
uvicorn.run("app.main:app", port=5656, reload=True)
```

---

## Fix order recommendation

```
C-1 → C-2 → C-3 → C-4   (Critical — fix before any further testing)
H-1 → H-2 → H-3 → H-4   (High — correctness and dissertation risk)
H-5 → H-6 → H-7          (High — privacy claim and test coverage)
M-1 → M-2 → M-3 → M-4   (Medium — stability and accuracy)
M-5 → M-6 → M-7 → M-8   (Medium — design cleanup)
L-1 → L-2 → L-3          (Low — documentation and polish)
```

---

*Last updated: audit pass on full codebase including RESULT.txt live run output.*