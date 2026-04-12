# Privacy-Preserving CDSS — Deep Project Analysis

> A **Hybrid Retrieval-Augmented Generation (RAG)** Clinical Decision Support System for genomic variant interpretation, using a **LangGraph** state-machine orchestrator, **PostgreSQL + pgvector** dual-store, and a local **Ollama** LLM.

---

## 1. Repository Map

```
Privacy-Preserving-CDSS/
├── app/
│   ├── main.py                          # FastAPI entrypoint (uvicorn, port 5656)
│   ├── config.py                        # Pydantic BaseSettings (.env loader)
│   ├── api/
│   │   ├── router.py                    # POST /query → invokes LangGraph
│   │   └── schemas.py                   # QueryRequest, QueryResponse, Citation
│   ├── db/
│   │   ├── postgres/
│   │   │   ├── clinvar_schema.sql       # DDL: variants + medical_documents tables
│   │   │   └── clinvar_ingestion.py     # Downloads & ingests ClinVar variants
│   │   └── vector/
│   │       └── indexing.py              # PDF/TXT → chunks → embeddings → pgvector
│   ├── models/
│   │   └── embeddings.py                # SentenceTransformer wrapper (768-dim)
│   └── pipeline/
│       ├── decomposition.py             # Keyword-based query → SubQuery routing
│       ├── construction/                # (empty — reserved for self_query / text2sql)
│       ├── graph/
│       │   ├── state.py                 # CDSSGraphState TypedDict
│       │   ├── nodes.py                 # 7 node functions (decompose→cite)
│       │   └── workflow.py              # LangGraph DAG wiring
│       ├── retrieval/
│       │   ├── multi_query.py           # LLM query expansion + vector search
│       │   ├── reranker.py              # BGE cross-encoder re-ranking
│       │   └── crag_evaluator.py        # Corrective RAG: score-based grading
│       ├── generation/
│       │   ├── guardrails.py            # System prompt + context builder
│       │   ├── self_rag.py              # JSON-schema-constrained generation
│       │   └── citation_enforcer.py     # Hallucination fix + citation extraction
│       └── sources/
│           ├── postgres_client.py       # ClinVar SQL lookups (rsid / gene)
│           ├── vector_client.py         # pgvector cosine similarity search
│           ├── gnomad_client.py         # gnomAD GraphQL API (allele frequency)
│           └── clingen_client.py        # ClinGen REST API (gene validity)
├── frontend/
│   ├── templates/index.html             # (empty — placeholder)
│   └── static/{css/style.css, js/main.js}
├── docs/
│   ├── manifest.json                    # Parser routing per document
│   ├── protocols/                       # NCCN Breast.pdf
│   └── screening/                       # NCCN Genetic-Familial.pdf
├── tests/
│   ├── test_routing.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_clingen_client.py
├── docker-compose.yml                   # pgvector + pgAdmin
└── requirements.txt
```

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph External["External Data Sources"]
        ClinVarFTP["ClinVar FTP (NCBI)"]
        gnomAD["gnomAD GraphQL API"]
        ClinGen["ClinGen REST API"]
    end

    subgraph Infrastructure["Infrastructure (Docker)"]
        PG["PostgreSQL + pgvector"]
        Ollama["Ollama LLM (local)"]
    end

    subgraph Backend["FastAPI Backend"]
        API["POST /query"]
        LG["LangGraph Orchestrator"]
    end

    subgraph Offline["Offline Ingestion"]
        CI["ClinVar Ingestion"]
        DI["Document Indexing"]
        DOCS["PDF/TXT Documents"]
    end

    User -->|"JSON: {query}"| API
    API -->|invoke| LG
    LG -->|SQL| PG
    LG -->|HTTP| gnomAD
    LG -->|HTTP| ClinGen
    LG -->|embed + search| PG
    LG -->|generate / critique| Ollama
    LG -->|"JSON response"| API
    API -->|"answer + citations + confidence"| User

    ClinVarFTP -->|"variant_summary.txt.gz"| CI
    CI -->|"INSERT INTO variants"| PG
    DOCS --> DI
    DI -->|"embed → INSERT INTO medical_documents"| PG
```

---

## 3. LangGraph Workflow — The Core Pipeline

The entire query lifecycle is a 7-node **Directed Acyclic Graph** defined in [workflow.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/workflow.py). Two parallel retrieval branches merge at the Evaluator.

```mermaid
graph LR
    START(("START")) --> D["Decomposer"]
    D --> DBR["DB_Retriever"]
    D --> PDFR["PDF_Retriever"]
    DBR --> E["Evaluator"]
    PDFR --> E
    E --> G["Generator"]
    G --> CE["Citation_Enforcer"]
    CE --> END_(("END"))

    style D fill:#4a90d9,color:#fff
    style DBR fill:#e67e22,color:#fff
    style PDFR fill:#e67e22,color:#fff
    style E fill:#27ae60,color:#fff
    style G fill:#8e44ad,color:#fff
    style CE fill:#2c3e50,color:#fff
```

### Node-by-Node Data Flow

| # | Node | File | Reads from State | Writes to State | Key Operations |
|---|------|------|-----------------|----------------|----------------|
| 1 | **Decomposer** | [nodes.py:16](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L16-L22) | [query](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/api/router.py#13-32) | [gene](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#128-135), `sub_queries` | Extract gene symbol; keyword-match → [SubQuery](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/decomposition.py#4-9) list |
| 2 | **DB_Retriever** | [nodes.py:24](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L24-L58) | `sub_queries`, [gene](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#128-135) | `trusted_chunks` | ClinVar SQL, gnomAD API, ClinGen API |
| 3 | **PDF_Retriever** | [nodes.py:61](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L61-L79) | `sub_queries` | `candidate_chunks` | Multi-query expansion → pgvector search per category |
| 4 | **Evaluator** | [nodes.py:82](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L82-L126) | `candidate_chunks`, `trusted_chunks`, [query](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/api/router.py#13-32), [gene](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#128-135) | `verified_chunks` | Deduplicate → rerank → CRAG grade → gene-filter NCCN → merge |
| 5 | **Generator** | [nodes.py:128](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L128-L134) | [query](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/api/router.py#13-32), `verified_chunks` | `draft_answer` | JSON-schema-constrained Ollama call → structured clinical response |
| 6 | **Citation_Enforcer** | [nodes.py:140](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/nodes.py#L140-L176) | `draft_answer`, `verified_chunks`, `trusted_chunks`, `candidate_chunks` | `final_answer`, [citations](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/generation/citation_enforcer.py#83-100), `confidence` | Fix hallucinated citations, extract citation list, compute confidence |

---

## 4. State Schema

Defined in [state.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/graph/state.py) as a `TypedDict`:

```python
class CDSSGraphState(TypedDict):
    query: str                              # original user question
    gene: Optional[str]                     # extracted gene symbol (e.g. "BRCA1")
    sub_queries: list                       # list[SubQuery] from decomposition

    trusted_chunks: list[RetrievedChunk]    # DB_Retriever output (ClinVar/gnomAD/ClinGen)
    candidate_chunks: list[RetrievedChunk]  # PDF_Retriever output (vector search)

    verified_chunks: list[RetrievedChunk]   # Evaluator output (merged + filtered)
    draft_answer: str                       # Generator output

    final_answer: str                       # Citation_Enforcer output
    citations: list[Citation]               # extracted inline citations
    confidence: str                         # "high" | "medium" | "low"
```

---

## 5. Query Decomposition Flow

[decomposition.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/decomposition.py) routes the query by keyword matching into sub-queries:

```mermaid
flowchart TD
    Q["User Query"] --> V{"Contains rsID?<br/>(rs\\d+)"}
    V -->|Yes| SQ1["SubQuery(target=postgres, type=data_extraction)"]
    Q --> S{"Screening keywords?"}
    Q --> P{"Protocol keywords?"}
    P -->|"Yes AND NOT screening"| SQ3["SubQuery(target=vector_db, type=protocol_retrieval)"]
    P -->|"Yes AND screening"| SKIP["Skipped (screening takes priority)"]
    S -->|Yes| SQ4["SubQuery(target=vector_db, type=screening_retrieval)"]
    Q --> CL{"ClinGen keywords?"}
    CL -->|Yes| SQ5["SubQuery(target=clingen, type=clingen_lookup)"]
    Q --> NONE{"No matches?"}
    NONE -->|Fallback| SQ6["SubQuery(target=vector_db, type=general)"]

    style SQ1 fill:#e67e22,color:#fff
    style SQ3 fill:#9b59b6,color:#fff
    style SQ4 fill:#27ae60,color:#fff
    style SQ5 fill:#e74c3c,color:#fff
    style SQ6 fill:#95a5a6,color:#fff
```

### Keyword Groups

| Category | Target | Keywords (sample) |
|----------|--------|-------------------|
| Variant IDs | [postgres](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/retrieval/reranker.py#45-59) | `rs\d+`, `NM_\d+`, `NP_\d+` |
| Protocol | `vector_db` (protocol) | `chemotherapy`, `radiotherapy`, `neoadjuvant`, `treatment regimen` |
| Screening | `vector_db` (screening_protocol) | `nccn`, `screening`, `surveillance`, `rrso`, `hereditary` |
| ClinGen | [clingen](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/retrieval/reranker.py#85-123) API | [clingen](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/retrieval/reranker.py#85-123), `gene validity`, `expert panel` |

---

## 6. DB_Retriever — Structured Data Path

```mermaid
flowchart TD
    SQ["Each SubQuery"] --> TGT{"target?"}

    TGT -->|postgres| RS{"Has rsID?"}
    RS -->|Yes| CV["postgres_client.get_variant_by_rsid()"]
    CV --> GN["gnomad_client.get_allele_frequency()"]
    GN -->|Found| AF["RetrievedChunk(source=gnomAD, AF + BA1 rule)"]
    GN -->|Not found| NF["RetrievedChunk(source=gnomAD, 'not found, PM2 may apply')"]
    CV --> CVR["RetrievedChunk(source=Clinvar, classification)"]

    RS -->|"No rsID but gene exists"| GE["postgres_client.get_variant_by_gene()"]
    GE --> CVR2["RetrievedChunk(source=Clinvar) per variant"]

    TGT -->|clingen| CG["clingen_client.get_gene_validity()"]
    CG --> CGR["RetrievedChunk(source=ClinGen, validity summary)"]

    AF --> TC["trusted_chunks[]"]
    NF --> TC
    CVR --> TC
    CVR2 --> TC
    CGR --> TC

    style TC fill:#e67e22,color:#fff
```

### Data Sources Deep Dive

| Source | Module | Type | Details |
|--------|--------|------|---------|
| **ClinVar** | [postgres_client.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/sources/postgres_client.py) | Local SQL | `SELECT FROM variants WHERE rsid = %s` or `WHERE gene_symbol = %s` |
| **gnomAD** | [gnomad_client.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/sources/gnomad_client.py) | External GraphQL | Resolves rsID → chr-pos-ref-alt via local DB, then queries `gnomad_r4`. Returns AF + BA1 applicability |
| **ClinGen** | [clingen_client.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/sources/clingen_client.py) | External REST | `GET /api/genes?search=BRCA1` → gene-disease validity, actionability, dosage, variant curation status |

---

## 7. PDF_Retriever — Unstructured Data Path

```mermaid
flowchart TD
    SQ["SubQuery(target=vector_db)"] --> QT{"query_type?"}

    QT -->|protocol_retrieval| MQPR["multi_query_search(category='protocol')"]
    QT -->|screening_retrieval| MQSC["multi_query_search(category='screening_protocol')"]
    QT -->|general| MQALL["multi_query_search(no filter)"]

    MQPR --> EXP["expand_queries() → Ollama LLM<br/>(JSON schema constrained)"]
    MQSC --> EXP
    MQALL --> EXP

    EXP --> VS["For each expanded query:<br/>vector_client.search_documents()"]
    VS --> DD["Deduplicate by chunk_text[:160]"]
    DD --> CC["candidate_chunks[]"]

    style CC fill:#3498db,color:#fff
```

### Multi-Query Expansion Detail

1. **Prompt selection** — different system prompts for `screening_retrieval`, `protocol_retrieval`, and `general` ([multi_query.py:58-103](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/retrieval/multi_query.py#L58-L103))
2. **JSON-schema constrained generation** — `ollama.chat(format=ExpandedQueries.model_json_schema())` forces the LLM to output `{"queries": [...]}` with no preamble
3. **Preamble stripping** — a safety net [_clean_variants()](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/retrieval/multi_query.py#38-56) strips numbered lines, bullet prefixes
4. **Search** — `original_query` + `n` variants each search pgvector with cosine similarity, filtered by `category`
5. **Deduplication** — by first 160 chars of `chunk_text`

---

## 8. Evaluator — Merging & Filtering

```mermaid
flowchart TD
    CC["candidate_chunks"] --> DEDUP["Deduplicate by text[:200]"]
    DEDUP --> RERANK["reranker.rerank_chunks()<br/>BGE cross-encoder scores"]
    RERANK --> CRAG["crag_evaluator.evaluate_chunks()"]

    CRAG --> CORRECT["CORRECT (score ≥ 0.0)"]
    CRAG --> AMB["AMBIGUOUS (-3.0 ≤ score < 0.0)"]
    CRAG --> INCORR["INCORRECT (score < -3.0) → dropped"]

    CORRECT --> POOL["all_passing = correct + ambiguous"]
    AMB --> POOL

    POOL --> GFILT{"Gene filter:<br/>NCCN chunks must<br/>mention target gene"}
    GFILT --> CAP["Cap 6 chunks per source"]
    CAP --> FILTERED["filtered_candidates"]

    TC["trusted_chunks"] --> MERGE["verified_chunks = trusted + filtered"]
    FILTERED --> MERGE

    style MERGE fill:#27ae60,color:#fff
```

---

## 9. Generation — Self-RAG with JSON Schema

```mermaid
flowchart TD
    VC["verified_chunks"] --> CTX["build_context_block()<br/>=== DB FACTS ===<br/>=== GUIDELINES ==="]
    CTX --> PROMPT["User prompt + CRITICAL RULES"]
    SP["build_system_prompt()<br/>(guardrails.py)"] --> OLLAMA

    PROMPT --> OLLAMA["ollama.chat()<br/>format=ClinicalResponse.model_json_schema()"]
    OLLAMA --> JSON["JSON: {summary, clingen_validity,<br/>screening_protocol}"]
    JSON --> RENDER["Render to markdown sections:<br/>**Clinical Summary**, **Screening**, etc."]
    RENDER --> FIX["fix_hallucinated_citations()"]
    FIX --> DRAFT["draft_answer"]

    style DRAFT fill:#8e44ad,color:#fff
```

### ClinicalResponse Schema (Pydantic)

```python
class ClinicalResponse(BaseModel):
    summary: ClinicalClaim             # variant classification + population freq
    clingen_validity: list[ClinicalClaim]  # gene-disease validity from ClinGen
    screening_protocol: list[ClinicalClaim]  # NCCN screening/surveillance
```

Each [ClinicalClaim](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/generation/self_rag.py#11-14) = `{text: str, citations: list[str]}`, where citations are `[Source: X, Reference: Y]` format.

---

---

## 10. Citation Enforcer & Confidence Scoring

```mermaid
flowchart TD
    FA["final_answer"] --> FIX["fix_hallucinated_citations()"]
    FIX --> EXTRACT["extract_citations()"]
    EXTRACT --> CITES["citations: list[Citation]"]

    TC["trusted_chunks"] --> CONF{"Confidence Scoring"}
    CCC["candidate_chunks"] --> CONF

    CONF --> HI["HIGH: ≥2 DB hits AND ≥1 PDF hit"]
    CONF --> MED["MEDIUM: DB OR PDF hits"]
    CONF --> LOW["LOW: neither"]

    style CITES fill:#2c3e50,color:#fff
```

### Citation Fix Logic ([citation_enforcer.py](file:///c:/Users/Master/Documents/GitHub/Privacy-Preserving-CDSS/app/pipeline/generation/citation_enforcer.py))

1. Parse all `[Source: X, Reference: Y]` in the answer text
2. If cited source **exists** in retrieved chunks → trust source, only fix reference
3. If cited source is **unknown** → score surrounding text against keyword dictionaries (`Clinvar`, `ClinGen`, `gnomAD`, `NCCN`) → remap to correct source (requires ≥2 keyword hits)

---

## 12. Offline Ingestion Pipelines

### 12a. ClinVar Ingestion

```mermaid
flowchart LR
    FTP["NCBI FTP"] -->|"variant_summary.txt.gz<br/>(~250 MB)"| DL["download_clinvar()"]
    DL --> MD5["verify_checksum()"]
    MD5 --> PARSE["Parse TSV with csv.DictReader"]
    PARSE --> FILT["is_relevant():<br/>GRCh38 + pathogenic/benign<br/>+ has assertion"]
    FILT --> BATCH["Batch INSERT (1000 rows)<br/>ON CONFLICT DO UPDATE"]
    BATCH --> PG["PostgreSQL variants table"]
```

### 12b. Document Indexing

```mermaid
flowchart TD
    DISC["discover_documents()<br/>reads manifest.json"] --> ROUTE{"Parser?"}

    ROUTE -->|docling| DOC["Docling: batch PDF conversion<br/>(10 pages at a time)<br/>+ NCCN boilerplate scrubbing"]
    ROUTE -->|txt| TXT["Plain text read"]

    DOC --> SPLIT1["table_aware_split()<br/>MarkdownHeaderSplitter + table isolation"]
    TXT --> SPLIT3["semantic_markdown_split()"]

    SPLIT1 --> CHILD["RecursiveCharacterTextSplitter<br/>(800 chars, 100 overlap)"]
    SPLIT3 --> CHILD

    CHILD --> EMBED["embed_text() → 768-dim vector"]
    EMBED --> INSERT["INSERT INTO medical_documents<br/>(source, category, gene, chunk, parent, embedding)"]

    style INSERT fill:#27ae60,color:#fff
```

### Document Categories

| Subfolder | Category DB Value | Parser | Documents |
|-----------|------------------|--------|-----------|
| `docs/_archive/` | *(removed)* | *(archived)* | ACMG 2015 v3 (no longer indexed) |
| `docs/protocols/` | `protocol` | docling | NCCN Breast v2 2026 |
| `docs/screening/` | `screening_protocol` | docling | NCCN Genetic/Familial High-Risk |

---

## 13. Database Schema

```mermaid
erDiagram
    variants {
        SERIAL id PK
        VARCHAR rsid UK "e.g. rs80357713"
        TEXT gene_symbol
        VARCHAR chromosome
        BIGINT position
        TEXT ref_allele
        TEXT alt_allele
        TEXT clinical_significance
        TEXT review_status
        TEXT condition
        DATE last_evaluated
        TIMESTAMP created_at
    }

    medical_documents {
        SERIAL id PK
        VARCHAR source "e.g. NCCN_Breast_v2_2026"
        VARCHAR category "protocol|screening_protocol"
        VARCHAR gene
        TEXT chunk_text
        VECTOR embedding "768-dim (pgvector)"
        JSONB metadata "Header_1..Header_6, page"
        TEXT parent_text "full parent context"
        TIMESTAMP created_at
    }
```

---

## 14. End-to-End Data Flow Trace

Here is a complete trace for an example query: **"What is the clinical significance of rs80357713 in BRCA1 and what are the NCCN screening recommendations?"**

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI /query
    participant D as Decomposer
    participant DBR as DB_Retriever
    participant PDFR as PDF_Retriever
    participant E as Evaluator
    participant G as Generator
    participant CE as Citation_Enforcer

    U->>API: POST {query: "rs80357713 BRCA1 screening..."}
    API->>D: invoke({query})

    Note over D: extract_gene → "BRCA1"
    Note over D: keyword match → 2 SubQueries:<br/>1. postgres/data_extraction (rs80357713)<br/>2. vector_db/screening_retrieval (NCCN)

    par DB_Retriever (parallel)
        D->>DBR: sub_queries + gene
        DBR->>DBR: ClinVar: get_variant_by_rsid("rs80357713")
        DBR->>DBR: gnomAD: get_allele_frequency("rs80357713")
        Note over DBR: trusted_chunks: [ClinVar chunk, gnomAD chunk]
    and PDF_Retriever (parallel)
        D->>PDFR: sub_queries
        PDFR->>PDFR: expand_queries() → 4 search queries
        PDFR->>PDFR: vector search × 4 queries × 15 top_k
        Note over PDFR: candidate_chunks: [~20 unique chunks]
    end

    DBR->>E: trusted_chunks
    PDFR->>E: candidate_chunks
    E->>E: deduplicate → rerank → CRAG grade
    E->>E: gene-filter NCCN for BRCA1
    Note over E: verified_chunks = 2 trusted + 5 filtered

    E->>G: verified_chunks + query
    G->>G: build_context_block() → system prompt
    G-->>G: Ollama JSON-schema gen → ClinicalResponse
    Note over G: draft_answer (markdown)

    G-->>CE: draft_answer + all chunks
    CE-->>CE: fix_hallucinated_citations()
    CE-->>CE: extract_citations()
    CE-->>CE: confidence = "high" (2 DB + PDF)

    CE-->>API: {final_answer, citations, confidence}
    API->>U: QueryResponse JSON
```

---

## 15. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | HTTP interface, Pydantic validation |
| **Orchestration** | LangGraph (StateGraph) | DAG-based pipeline with parallel branches |
| **LLM** | Ollama (local, configurable model) | Query expansion, generation, critic |
| **Embeddings** | SentenceTransformer (768-dim) | Document & query embedding |
| **Reranking** | `BAAI/bge-reranker-large` (CrossEncoder) | Precision re-ranking of candidate chunks |
| **Database** | PostgreSQL 17 + pgvector | Dual-store: structured SQL + vector similarity |
| **PDF Parsing** | Docling (table-aware) | Layout-aware PDF → Markdown conversion |
| **Chunking** | LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | Hierarchical → child chunk splitting |
| **External APIs** | gnomAD GraphQL, ClinGen REST | Population frequency, gene validity |
| **Config** | pydantic-settings + `.env` | Environment-based configuration |
| **Container** | Docker Compose | pgvector + pgAdmin orchestration |

---

## 16. Key Design Patterns

1. **Hybrid RAG** — structured DB facts ("trusted") + unstructured PDF chunks ("candidates") merged at evaluation
2. **Corrective RAG (CRAG)** — cross-encoder scoring with three grades (correct/ambiguous/incorrect) to filter low-quality chunks
3. **Self-RAG** — JSON-schema-constrained generation ensures structured output
4. **Multi-Query Expansion** — dedicated prompt templates per document category (screening/protocol) to improve retrieval recall
5. **Citation Enforcement** — post-hoc hallucination detection using keyword dictionaries + source existence checks
6. **Parallel Retrieval** — LangGraph fan-out: DB_Retriever and PDF_Retriever execute concurrently, merge at Evaluator
7. **Privacy-Preserving** — all LLM inference runs locally via Ollama; no patient data leaves the system

---

## 17. Unused / Placeholder Modules

| File | Status |
|------|--------|
| `pipeline/construction/self_query.py` | Empty — reserved for self-querying retrieval |
| `pipeline/construction/text_to_sql.py` | Empty — reserved for natural language → SQL |
| `frontend/templates/index.html` | Empty — no frontend implementation yet |
