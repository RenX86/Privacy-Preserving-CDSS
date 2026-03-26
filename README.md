# 🧬 Privacy-Preserving Clinical Decision Support System

<div align="center">

**A locally-hosted Hybrid RAG pipeline for clinical genomic queries**

*Combining air-gapped local databases · Live expert APIs · Local LLM inference · Zero hallucination tolerance*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.131-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17_+_pgvector-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF6B35?style=flat)]()
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-8E44AD?style=flat)]()
[![ClinGen](https://img.shields.io/badge/ClinGen-Live_API-6C3483?style=flat)]()
[![gnomAD](https://img.shields.io/badge/gnomAD_v4-Live_API-E74C3C?style=flat)]()
[![Status](https://img.shields.io/badge/Status-In_Development-F39C12?style=flat)]()

</div>

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [System at a Glance](#-system-at-a-glance)
- [Data Sources](#-data-sources--why-hybrid-rag)
- [The 7-Node LangGraph Pipeline](#-the-7-node-langgraph-pipeline)
  - [Node 1 — Decomposer](#node-1--decomposer)
  - [Node 2 — DB\_Retriever](#node-2--db_retriever)
  - [Node 3 — PDF\_Retriever](#node-3--pdf_retriever)
  - [Node 4 — Evaluator](#node-4--evaluator)
  - [Node 5 — Generator](#node-5--generator)
  - [Node 6 — Critic](#node-6--critic)
  - [Node 7 — Citation\_Enforcer](#node-7--citation_enforcer)
- [Offline Ingestion Pipelines](#-offline-ingestion-pipelines)
- [Database Schema](#-database-schema)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Example Query & Response](#-example-query--response)
- [Roadmap](#-roadmap)

---

## 🔬 What It Does

This system answers complex clinical and genomic queries — variant classification, ACMG criteria application, NCCN protocol retrieval, and ClinGen expert panel validation — with **zero hallucinations and full data privacy**.

**Example query:**
> *"What is the clinical significance of rs879254116 in BRCA1, which ACMG pathogenicity criteria apply, and what cancer screening protocol should the patient follow? Also confirm the ClinGen expert panel validity for BRCA1."*

**Example output structure:**
```
**Clinical Summary**
Variant rs879254116 in BRCA1 is classified as Pathogenic.
[Source: Clinvar, Reference: rs879254116]

**ClinGen Expert Panel Validity**
* BRCA1 has gene-disease validity curated by ClinGen expert panel.
  Actionability curations available. Last curated: 08/29/2024.
  [Source: ClinGen, Reference: BRCA1 (HGNC:1100)]

**ACMG Pathogenicity Criteria**
* PVS1: Null variant (nonsense, frameshift, canonical ±1 or 2 splice sites,
  initiation codon, or multi-exon deletion) in a gene where loss of function
  is a known mechanism of disease.
  [Source: ACMG_2015, Reference: page 33]
* PM2: Absent from controls (or at extremely low frequency if recessive)
  in population databases. The gnomAD record shows this variant was
  not found, which is relevant to this criterion.
  [Source: ACMG_2015, Reference: page 33]

**Cancer Screening Protocol**
* Annual breast MRI and mammography starting at age 25-29 years.
  [Source: Genetic-Familial High-Risk Assessment, Reference: NCCN Guidelines
  Version 3.2026 Genetic/Familial High-Risk Assessment...]
* Risk-reducing salpingo-oophorectomy between ages 35-40.
  [Source: Genetic-Familial High-Risk Assessment,
  Reference: Bilateral Salpingo-Oophorectomy]
```

> **Design principle — ACMG criteria:** The system reports ACMG criteria **definitions** from the guideline text and links them to relevant database evidence. It does NOT conclude which criteria "apply" to the specific variant — that requires clinical review. This prevents the LLM from inventing variant biology (e.g., guessing frameshift vs. missense when no variant type is in the data).

**Key guarantees:**
| Guarantee | How it's enforced |
|-----------|------------------|
| 🔒 No patient data leaves the system | All LLM inference via local Ollama; gnomAD receives only positional coords |
| 📎 Every claim has a citation | Citation manifest injected into prompt; enforcer post-validates all refs |
| 🚫 No hallucinated variant biology | Rules 5+7 forbid inventing variant types or applying ACMG criteria from memory |
| 🧬 No cross-gene table slippage | Rule 6 requires gene-name verification in every NCCN table row |
| ✅ Structured, auditable output | 4-section Pydantic schema: Summary · ClinGen · ACMG · Screening |
| ⚡ Safe failure | Insufficient data → predefined refusal, never a guess |

---

## 🗺️ System at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRIVACY-PRESERVING CDSS — FULL SYSTEM                    │
└─────────────────────────────────────────────────────────────────────────────┘

  OFFLINE (run once)                      ONLINE (per query)
  ─────────────────                       ────────────────────────────────────

  ClinVar FTP (~250MB)                    👨‍⚕️ Clinician
       │                                        │  POST /query
       ▼                                        ▼
  clinvar_ingestion.py              ┌─── FastAPI (port 5656) ───────────────┐
  • Download + MD5 verify           │                                       │
  • Filter: GRCh38, P/LP/B/LB       │   LangGraph 7-Node Pipeline           │
  • Batch INSERT (1000/tx)          │                                       │
       │                            │  ① Decomposer                         │
       ▼                            │       ↓ fan-out (parallel)            │
  ┌──────────────┐  indexing.py     │  ② DB_Retriever ── ③ PDF_Retriever    │
  │  PostgreSQL  │◄── Docling       │       ↓ merge          ↓              │
  │  + pgvector  │    PyMuPDF4LLM   │  ④ Evaluator ◄──────────────────────  │
  │              │                  │       ↓                               │
  │  variants    │    800-char      │  ⑤ Generator                          │
  │  table       │    chunks        │       ↓                               │
  │              │                  │  ⑥ Critic                             │
  │  medical_    │                  │       ↓                               │
  │  documents   │                  │  ⑦ Citation_Enforcer                  │
  │  (768-dim)   │                  │       ↓                               │
  └──────────────┘                  │  QueryResponse (JSON)                 │
         ▲                          └───────────────────────────────────────┘
         │                                   │
         │ Live calls (per query)             ▼
  ┌──────┴───────────────────────┐    answer + citations + confidence
  │  gnomAD v4  │  ClinGen REST  │
  │  (allele    │  (gene expert  │
  │  frequency) │   panels)      │
  └──────────────────────────────┘
```

---

## 🗄️ Data Sources — Why Hybrid RAG?

```
                    ┌────────────────────────────────────────────────┐
                    │         FOUR DATA SOURCES EXPLAINED            │
                    └────────────────────────────────────────────────┘

  ┌──────────────────────┐   ┌──────────────────────┐
  │  🗄️  PostgreSQL       │   │  🔮 pgvector       │
  │  ClinVar (local)     │   │  ACMG + NCCN (local) │
  │                      │   │                      │
  │  • Millions of       │   │  • ACMG 2015         │
  │    variant records   │   │    guidelines        │
  │  • rsID lookups      │   │  • NCCN Breast v2026 │
  │  • Gene-level scan   │   │  • NCCN Genetic/     │
  │  • SQL precision     │   │    Familial High-Risk│
  │                      │   │  • 768-dim BGE embed │
  │  ✅ Air-gapped       │   │    embeddings       │
  │  ✅ Sub-millisecond  │   │                     │
  │  ✅ Exact match      │   │  ✅ Air-gapped      │
  └──────────────────────┘   │  ✅ Semantic search  │
                              └─────────────────────┘

  ┌──────────────────────┐   ┌──────────────────────┐
  │  🌐 gnomAD v4 API     │   │  🌐 ClinGen REST API  │
  │  (live, opt-out)     │   │  (live, gene-only)   │
  │                      │   │                      │
  │  • Population allele │   │  • Gene-disease      │
  │    frequency         │   │    validity          │
  │  • ACMG BA1 rule:    │   │  • Actionability     │
  │    AF ≥ 5% → Benign  │   │  • Dosage sensitivity│
  │  • ACMG PM2 rule:    │   │  • Variant expert    │
  │    absent → may apply│   │    panel status      │
  │  • Sends: chrom-pos- │   │  • Last curated date │
  │    ref-alt only      │   │  • Sends: gene symbol│
  │                      │   │    only (e.g. BRCA1) │
  │  ⚠️  Disable with    │   │                      │
  │  ENABLE_GNOMAD=false │   │  ✅ Real-time data    │
  └──────────────────────┘   └──────────────────────┘
```

---

## ⚙️ The 7-Node LangGraph Pipeline

```
                        ┌─────────────────────────┐
                        │   CDSSGraphState (shared │
                        │   across all 7 nodes)    │
                        │                          │
                        │  query: str              │
                        │  gene: Optional[str]     │
                        │  sub_queries: list       │
                        │  trusted_chunks: list    │
                        │  candidate_chunks: list  │
                        │  verified_chunks: list   │
                        │  draft_answer: str       │
                        │  final_answer: str       │
                        │  citations: list         │
                        │  confidence: str         │
                        └─────────────────────────┘

   START
     │
     ▼
 ╔═══════════╗
 ║  Node 1   ║  decompose_node()
 ║ Decomposer║  • Extract gene symbol via regex
 ╚═════╤═════╝  • Keyword match → typed SubQuery list
       │         • Writes: gene, sub_queries
       │
       ├──────────────────────────┐
       │  (parallel via Send API) │
       ▼                          ▼
 ╔═══════════╗            ╔═════════════╗
 ║  Node 2   ║            ║   Node 3    ║
 ║DB_Retriever            ║PDF_Retriever║
 ╚═════╤═════╝            ╚══════╤══════╝
       │                         │
  ClinVar SQL               Multi-query
  gnomAD API                expansion (LLM)
  ClinGen API               pgvector search
       │                    Deduplication
       │                         │
       └───────────┬─────────────┘
                   │  (merge via Annotated[list, operator.add])
                   ▼
           ╔══════════════╗
           ║   Node 4     ║  evaluate_node()
           ║  Evaluator   ║  • Deduplicate candidate_chunks
           ╚══════╤═══════╝  • BGE cross-encoder rerank
                  │          • CRAG grade (CORRECT/AMBIGUOUS/INCORRECT)
                  │          • Gene-filter NCCN chunks
                  │          • Cap 6 chunks per source
                  │          • Merge: trusted + filtered_candidates
                  ▼
           ╔══════════════╗
           ║   Node 5     ║  generate_node()
           ║  Generator   ║  • Build ⚑ VARIANT FACTS context block
           ╚══════╤═══════╝  • Build numbered CITATION MANIFEST
                  │          • JSON-schema constrained Ollama call
                  │          • Parse ClinicalResponse → markdown
                  ▼
           ╔══════════════╗
           ║   Node 6     ║  critic_node()
           ║    Critic    ║  • LLM audit: ages, genes, citations
           ╚══════╤═══════╝  • Safety net: revert if citations dropped
                  │
                  ▼
           ╔══════════════════╗
           ║     Node 7       ║  citation_node()
           ║Citation_Enforcer ║  • fix_hallucinated_citations()
           ╚══════╤═══════════╝  • extract_citations()
                  │              • Confidence scoring
                  ▼
                 END → QueryResponse{answer, citations, confidence}
```

---

### Node 1 — Decomposer

Breaks the query into typed `SubQuery` objects via keyword and regex matching. **Critical design decision:** each sub-query gets a **focused text** containing only the keywords relevant to that topic — NOT the full user query. This prevents embedding dilution and ensures the reranker scores chunks against the correct topic.

```
Query: "What is rs879254116 in BRCA1 and what NCCN screening applies? ClinGen validity?"
                              │
     ┌────────────────────────┼───────────────────────────┐
     │                        │                           │
     ▼                        ▼                           ▼
┌─────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│SubQuery 1       │   │SubQuery 2            │   │SubQuery 3          │
│target: postgres │   │target: vector_db     │   │target: clingen     │
│type: data_ext.  │   │type: screening_ret.  │   │type: rule_retrieval│
│                 │   │                      │   │                    │
│text: "Get       │   │text: "NCCN cancer    │   │text: original query│
│clinical signif. │   │screening surveillance│   │(API call, not      │
│for rs879254116" │   │mammography MRI BRCA1 │   │ vector search)     │
│                 │   │carrier management"   │   │                    │
│✅ Already focused│   │✅ Focused — no ACMG  │   │✅ API-based         │
└─────────────────┘   │   or ClinGen terms   │   └────────────────────┘
                      └──────────────────────┘
```

> **Why focused sub-query text matters:** When the reranker sees `"ACMG criteria PVS1 PM2"` vs an ACMG table about PVS1, it scores **0.601** (CORRECT). When it sees the full 40-word mixed query, the same table scores **0.001** (INCORRECT → dropped). Focused text = 200-400x improvement in relevance scoring.

**Keyword groups that trigger each sub-query type:**

| Detected pattern | Sub-query target | Sub-query type | Focused text template |
|---|---|---|---|
| `rs\d+`, `NM_\d+.\d+`, `NP_\d+.\d+` | `postgres` | `data_extraction` | `"Get clinical significance for {rsID}"` |
| `acmg`, `pathogenic`, `pvs1`, `criteria`, `vus`, `benign` | `vector_db` | `rule_retrieval` | `"ACMG pathogenicity criteria PVS1 PS1-4 PM1-5 PP1-5 {gene} {variant}"` |
| `nccn`, `screening`, `surveillance`, `rrso`, `hereditary`, `mammography`, `mri` | `vector_db` | `screening_retrieval` | `"NCCN cancer screening surveillance mammography MRI risk-reducing {gene} carrier"` |
| `chemotherapy`, `neoadjuvant`, `radiotherapy`, `treatment regimen` | `vector_db` | `protocol_retrieval` | `"Cancer treatment protocol chemotherapy surgery radiation {gene}"` |
| `clingen`, `gene validity`, `expert panel`, `actionability` | `clingen` | `rule_retrieval` | Original query (ClinGen is API-based, not vector search) |
| *(no match)* | `vector_db` | `rule_retrieval` | Original query *(fallback)* |

> **Screening takes priority over protocol**: if both keywords appear (e.g. "BRCA1 screening protocol"), only `screening_retrieval` fires — `protocol_retrieval` is suppressed to avoid noise.

---

### Node 2 — DB\_Retriever

Fetches verified structured facts from three sources. Results become `trusted_chunks` — they bypass CRAG filtering and go directly to the LLM as ground truth.

```
For each SubQuery where target == "postgres":
  ┌─────────────────────────────────────────────────────────────────┐
  │ rsID found in query?                                            │
  │      YES                            NO (gene only)             │
  │       │                                  │                     │
  │       ▼                                  ▼                     │
  │  get_variant_by_rsid(rsid)      get_variant_by_gene(gene)      │
  │       │                                  │                     │
  │       ▼                                  ▼                     │
  │  gnomAD GraphQL API             Multiple RetrievedChunks       │
  │  chrom-pos-ref-alt query         (source=Clinvar)              │
  │       │                                                        │
  │  Found?    Not found?                                          │
  │    │            │                                              │
  │    ▼            ▼                                              │
  │  AF + BA1   PM2 may apply note                                 │
  │  (source=gnomAD)                                               │
  └─────────────────────────────────────────────────────────────────┘

For each SubQuery where target == "clingen":
  ┌─────────────────────────────────────────────────────────────────┐
  │  GET /api/genes?search={gene_symbol}                           │
  │       │                                                        │
  │       ▼                                                        │
  │  Filter to exact symbol match (BRCA1 ≠ BRCA1P1)               │
  │       │                                                        │
  │       ▼                                                        │
  │  RetrievedChunk(source=ClinGen):                               │
  │    • Gene-disease validity: YES/NOT CURATED                    │
  │    • Actionability curations: YES/NO                           │
  │    • Dosage sensitivity: Curated/Not curated                   │
  │    • Variant expert panel: active/not active                   │
  │    • date_last_curated                                         │
  └─────────────────────────────────────────────────────────────────┘
```

---

### Node 3 — PDF\_Retriever

Searches the local pgvector database using multi-query expansion. Each sub-query's results are **reranked per-subquery** against their own focused text before being pooled — ensuring ACMG criteria are scored against ACMG queries, not against the full multi-topic user query.

```
For each SubQuery where target == "vector_db":

  STEP 1 — Query Expansion (Ollama LLM, JSON-schema constrained)
  ┌───────────────────────────────────────────────────────────────┐
  │  seed = focused sub-query text (from Decomposer)              │
  │                                                               │
  │  e.g. "ACMG pathogenicity criteria PVS1 PS1 PM2 BRCA1"       │
  │       → LLM generates 3 variants:                             │
  │       • ACMG PVS1 criteria loss of function null alleles      │
  │       • ACMG PP3 PP4 moderate computational prediction        │
  │       • ACMG BS1 BS2 benign evidence population frequency     │
  │                                                               │
  │  Output: ExpandedQueries {queries: [q1, q2, q3]}             │
  └───────────────────────────────────────────────────────────────┘

  STEP 2 — Vector Search (4 queries × top_k=15)
  ┌───────────────────────────────────────────────────────────────┐
  │  [focused_sub_query, q1, q2, q3]                              │
  │         │                                                     │
  │         ▼  for each query:                                    │
  │  embed_text() → 768-dim BGE-base-en-v1.5 vector              │
  │         │                                                     │
  │         ▼                                                     │
  │  SELECT ... FROM medical_documents                            │
  │  WHERE category = {guideline|protocol|screening_protocol}     │
  │  ORDER BY embedding <=> query_vector                          │
  │  LIMIT 15                                                     │
  └───────────────────────────────────────────────────────────────┘

  STEP 3 — Deduplication + Per-Subquery Reranking
  ┌───────────────────────────────────────────────────────────────┐
  │  4 queries × 15 results → deduplicate → ~30-40 unique chunks │
  │                                                               │
  │  ★ BGE cross-encoder reranks THIS batch against the focused  │
  │    sub-query text (not the full user query)                   │
  │                                                               │
  │  Why: ACMG Table 3 scores 0.601 vs "ACMG criteria PVS1 PM2" │
  │       but only 0.001 vs the full 40-word user query           │
  │                                                               │
  │  Chunks arrive at Evaluator already scored per-topic          │
  └───────────────────────────────────────────────────────────────┘
```

**Indexed documents:**

| File | Source name | Category | Parser |
|---|---|---|---|
| `ACMG_2015_v3.pdf` | `ACMG_2015` | `guideline` | PyMuPDF4LLM (page chunks) |
| `Breast.pdf` | `NCCN_Breast_v2_2026` | `protocol` | Docling (table-aware) |
| `Genetic-Familial High-Risk Assessment...pdf` | `Genetic-Familial High-Risk Assessment` | `screening_protocol` | Docling (table-aware) |

---

### Node 4 — Evaluator

Merges both retrieval branches, applies quality filtering, and produces the final `verified_chunks` list that gets sent to the LLM. Chunks arrive **pre-scored** from per-subquery reranking in PDF_Retriever — no re-reranking against the full query happens here.

```
  trusted_chunks (DB_Retriever)       candidate_chunks (PDF_Retriever)
         │                            (already scored per-subquery)
         │                                        │
         │                                        ▼
         │                          STEP 1: Deduplicate by text[:200]
         │                          (same chunk from multiple sub-queries
         │                           → keep first occurrence's score)
         │                                        │
         │                                        ▼
         │                          STEP 2: Sort by score (descending)
         │                          (chunks pre-scored in PDF_Retriever;
         │                           NO re-reranking against full query)
         │                                        │
         │                                        ▼
         │                          STEP 3: CRAG Grading
         │                          ┌─────────────────────────────────┐
         │                          │  score ≥ 0.05  → CORRECT  ✅   │
         │                          │  score 0.01–0.05 → AMBIGUOUS ⚠️ │
         │                          │  score < 0.01  → INCORRECT ❌   │
         │                          │                   (dropped)      │
         │                          └─────────────────────────────────┘
         │                                        │
         │                          STEP 4: Gene Filter (NCCN only)
         │                          ┌─────────────────────────────────┐
         │                          │  NCCN chunk doesn't mention     │
         │                          │  target gene → DROPPED          │
         │                          │  (prevents STK11/CDH1 protocols │
         │                          │   appearing in BRCA1 queries)   │
         │                          └─────────────────────────────────┘
         │                                        │
         │                          STEP 5: Source cap (max 6 per source)
         │                                        │
         └──────────────────┬─────────────────────┘
                            ▼
                     verified_chunks
              (trusted DB chunks + filtered PDF chunks)
                   typically 3 DB + 8–12 PDF chunks
```

> **Previous design flaw (fixed):** The Evaluator previously re-reranked ALL chunks against the full multi-topic user query. This caused ACMG criteria tables to score 0.001 (mixing NCCN/screening terms diluted the relevance signal) and get dropped as INCORRECT. Now chunks are scored per-subquery in PDF_Retriever before reaching the Evaluator.

---

### Node 5 — Generator

Builds a structured clinical response using grammar-constrained JSON generation and a 7-rule system prompt.

```
  verified_chunks
        │
        ▼
  build_system_prompt() — 7 anti-hallucination rules:
  ┌──────────────────────────────────────────────────────┐
  │  RULE 1: Variant facts block is ground truth          │
  │  RULE 2: Citations must be copied from manifest       │
  │  RULE 3: Exact ages and terminology from guidelines   │
  │  RULE 4: If not in context, say "Data unavailable"    │
  │  RULE 5: Never invent variant biology (frameshift/    │
  │          missense/de novo/computational predictions)   │
  │  RULE 6: Table row isolation — verify gene name in    │
  │          every NCCN table row before extracting data   │
  │  RULE 7: ACMG — report definitions only, do not      │
  │          apply criteria to the specific variant        │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  build_context_block()
  ┌──────────────────────────────────────────────────────┐
  │  ⚑ ═══════════════════════════════════════════════  │
  │  ⚑  VARIANT DATABASE FACTS — READ THIS FIRST       │
  │  ⚑ ═══════════════════════════════════════════════  │
  │                                                      │
  │  [Source: Clinvar, Reference: rs879254116]           │
  │  *** CONFIRMED CLASSIFICATION: Pathogenic ***        │
  │                                                      │
  │  [Source: gnomAD, Reference: rs879254116]            │
  │  Variant not found. PM2 (absent) MAY apply.          │
  │                                                      │
  │  [Source: ClinGen, Reference: BRCA1 (HGNC:1100)]    │
  │  Gene-disease validity: True. Actionability: True.   │
  │                                                      │
  │  ── CLINICAL GUIDELINES ──────────────────────────  │
  │  [Source: ACMG_2015, Reference: page 33]             │
  │  Table 3: Criteria for Classifying Pathogenic...     │
  │  ...                                                 │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  _build_reference_manifest()
  ┌──────────────────────────────────────────────────────┐
  │  CITATION MANIFEST — ONLY these are permitted        │
  │  [1] [Source: Clinvar, Reference: rs879254116]       │
  │  [2] [Source: gnomAD, Reference: rs879254116]        │
  │  [3] [Source: ClinGen, Reference: BRCA1 (HGNC:1100)]│
  │  [4] [Source: ACMG_2015, Reference: page 33]         │
  │  [5] [Source: Genetic-Familial..., Reference: BRCA   │
  │       PATHOGENIC VARIANT-POSITIVE MANAGEMENT]        │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  ollama.chat(format=ClinicalResponse.model_json_schema())
  ┌──────────────────────────────────────────────────────┐
  │  Grammar-constrained — model CANNOT output free text │
  │                                                      │
  │  ClinicalResponse {                                  │
  │    summary: ClinicalClaim,         ← variant facts   │
  │    clingen_validity: [ClinicalClaim], ← ClinGen data │
  │    acmg_rules: [ClinicalClaim],    ← definitions     │
  │    screening_protocol: [ClinicalClaim] ← NCCN        │
  │  }                                                   │
  │                                                      │
  │  ClinicalClaim {                                     │
  │    text: str,          ← full sentence explanation   │
  │    citations: [str]    ← copied from manifest ONLY   │
  │  }                                                   │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  _strip_thinking() → remove <think>...</think> blocks
        │
        ▼
  Parse JSON → render markdown sections → draft_answer
```

---

### Node 6 — Critic

A second LLM pass that audits the draft for clinical accuracy errors.

```
  draft_answer + verified_chunks
        │
        ▼
  ollama.chat() [free text — no JSON schema]
  ┌────────────────────────────────────────────────────────────┐
  │  AUDIT CHECKLIST:                                          │
  │  1. Wrong gene data? (wrong age range / wrong surgery       │
  │     assigned to gene — e.g. BRCA2 RRSO applied to BRCA1)  │
  │  2. Hallucinations? (claims not in verified_chunks)        │
  │  3. Missing citations? (every sentence needs [Source: X])  │
  └────────────────────────────────────────────────────────────┘
        │
        ▼
  Citation safety net:
  ┌────────────────────────────────────────────────────────────┐
  │  draft_cite_count = draft_answer.count("[Source:")         │
  │  final_cite_count = final_answer.count("[Source:")         │
  │                                                            │
  │  if final_cite_count < draft_cite_count:                   │
  │      WARNING: Critic dropped citations → REVERT TO DRAFT   │
  └────────────────────────────────────────────────────────────┘
```

---

### Node 7 — Citation\_Enforcer

Post-processes the final answer and computes the confidence score.

```
  final_answer + verified_chunks
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
  fix_hallucinated_citations()           extract_citations()
  ┌──────────────────────────┐          ┌──────────────────────┐
  │  For each [Source: X,    │          │  Regex scan for all  │
  │  Reference: Y] in text:  │          │  [Source:...] tags   │
  │                          │          │  Deduplicate          │
  │  Source in chunks?       │          │  → citations list    │
  │  ├─YES: ref valid? ──────┤          └──────────────────────┘
  │  │  ├─YES: leave alone   │
  │  │  └─NO: remap to       │
  │  │        first valid ref │          Confidence scoring:
  │  └─NO: keyword-score     │          ┌──────────────────────┐
  │        200-char context  │          │  DB hits ≥ 2         │
  │        ≥2 hits → remap   │          │  AND PDF hits ≥ 1    │
  │        to correct source  │          │  → HIGH              │
  └──────────────────────────┘          │                      │
                                        │  DB hits OR PDF hits │
                                        │  → MEDIUM            │
                                        │                      │
                                        │  Neither → LOW       │
                                        └──────────────────────┘
```

---

## 📥 Offline Ingestion Pipelines

### ClinVar Variant Ingestion

```
NCBI FTP Server
      │
      ▼ urllib.request.urlretrieve()
variant_summary.txt.gz (~250MB)
      │
      ▼ MD5 checksum verify (re-download if corrupted)
      │
      ▼ gzip.open() + csv.DictReader (tab-separated)
      │
      ▼ is_relevant() filter:
        • Assembly == "GRCh38" (discard GRCh37)
        • ClinicalSignificance contains: Pathogenic / Likely Pathogenic
          / Benign / Likely Benign
        • ReviewStatus ≠ "no assertion" / "no interpretation"
        • rsID present (not "-1")
      │
      ▼ Batch INSERT (1000 rows per transaction)
        ON CONFLICT (rsid) DO UPDATE  ← keeps database current
      │
      ▼ PostgreSQL variants table
```

### Document Indexing Pipeline

```
docs/manifest.json
(source, category, gene, parser per file)
      │
      ▼ discover_documents()
      │
      ├─── ACMG_2015_v3.pdf ──► parser: pymupdf
      │         │
      │         ▼ pymupdf4llm.to_markdown(page_chunks=True)
      │         │  One dict per page → no header-splitting needed
      │         │  Scrub: journal headers, HHS notices, author lines
      │         │
      │         ▼ sections = list of page dicts
      │
      ├─── Breast.pdf (NCCN) ──► parser: docling
      │         │
      │         ▼ Docling (do_table_structure=True, do_ocr=False)
      │         │  Batch 10 pages at a time → Markdown
      │         │  Scrub: NCCN copyright, version headers, ToC
      │         │
      │         ▼ table_aware_split():
      │            MarkdownHeaderSplitter (H1/H2/H3)
      │            + isolate pipe-table blocks separately
      │
      └─── Genetic-Familial...pdf ──► parser: docling (same path)
                │
                ▼ (same as above)

      All paths converge:
      │
      ▼ RecursiveCharacterTextSplitter
        chunk_size=800, chunk_overlap=100
        enriched_child = "[Header_2: SECTION NAME]\nchunk text"
      │
      ▼ embed_text() → BGE-base-en-v1.5 768-dim vector
      │
      ▼ INSERT INTO medical_documents
        (source, category, gene, chunk_text, metadata,
         parent_text, embedding::vector)
```

---

## 🗃️ Database Schema

```
┌─────────────────────────────────────────────────────────────────┐
│  variants                                                       │
├──────────────────────┬──────────────────────────────────────────┤
│  id SERIAL PK        │  rsid VARCHAR(20) UNIQUE                 │
│  gene_symbol TEXT    │  chromosome VARCHAR(10)                  │
│  position BIGINT     │  ref_allele TEXT                         │
│  alt_allele TEXT     │  clinical_significance TEXT              │
│  review_status TEXT  │  condition TEXT                          │
│  last_evaluated DATE │  created_at TIMESTAMP                    │
├──────────────────────┴──────────────────────────────────────────┤
│  Indexes: idx_variants_rsid (primary lookup)                    │
│           idx_variants_gene (gene-level scans)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  medical_documents                                              │
├──────────────────────┬──────────────────────────────────────────┤
│  id SERIAL PK        │  source VARCHAR(50)                      │
│  category VARCHAR(50)│  gene VARCHAR(50)                        │
│  chunk_text TEXT     │  embedding vector(768)  ← pgvector       │
│  metadata JSONB      │  parent_text TEXT                        │
│  created_at TIMESTAMP│                                          │
├──────────────────────┴──────────────────────────────────────────┤
│  metadata JSONB shape:                                          │
│  { "Header_1": "...", "Header_2": "SECTION NAME",               │
│    "Header_3": "...", "page": 33 }                              │
│                                                                 │
│  Reference extraction priority:                                 │
│  Header_6 → Header_5 → ... → Header_1 → page number           │
│                                                                 │
│  Indexes: idx_documents_source                                  │
│           idx_documents_embedding (ivfflat cosine_ops)          │
│           idx_documents_parent_text                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| **API** | FastAPI + Uvicorn | 0.131 / 0.41 | HTTP interface, Pydantic I/O validation |
| **Orchestration** | LangGraph (StateGraph) | via langchain | 7-node DAG, parallel Send-based fan-out |
| **LLM** | Ollama | 0.6.1 | Local inference — query expansion, generation, critic |
| **Embeddings** | SentenceTransformer | 5.2.3 | BGE-base-en-v1.5 768-dim vectors |
| **Reranking** | BAAI/bge-reranker-large | — | Cross-encoder relevance scoring |
| **Database** | PostgreSQL 17 + pgvector | psycopg2 2.9.11 | Structured variants + vector similarity search |
| **Connection pool** | psycopg2 ThreadedConnectionPool | — | min=2 / max=10 connections shared |
| **PDF — NCCN** | Docling | unpinned | Table-structure-aware PDF → Markdown |
| **PDF — ACMG** | PyMuPDF4LLM | — | Page-chunk PDF → Markdown |
| **Chunking** | LangChain text splitters | 1.2.10 | MarkdownHeader + Recursive (800 chars, 100 overlap) |
| **External API 1** | gnomAD v4 GraphQL | httpx 0.28.1 | Allele frequency, BA1/PM2 criteria |
| **External API 2** | ClinGen REST | httpx 0.28.1 | Gene validity, expert panel curation |
| **Config** | pydantic-settings | 2.13.1 | `.env`-based configuration |
| **Container** | Docker Compose | — | pgvector/pgAdmin |

---

## 📁 Project Structure

```
Privacy-Preserving-CDSS/
│
├── app/
│   ├── main.py                     FastAPI entrypoint — port 5656
│   ├── config.py                   Pydantic settings, reads .env
│   │
│   ├── api/
│   │   ├── router.py               POST /query → LangGraph invoke
│   │   └── schemas.py              QueryRequest, QueryResponse, Citation
│   │
│   ├── db/
│   │   ├── pool.py                 ★ Shared ThreadedConnectionPool (min=2, max=10)
│   │   ├── postgres/
│   │   │   ├── clinvar_schema.sql  DDL: variants + medical_documents + indexes
│   │   │   └── clinvar_ingestion.py Download, verify MD5, batch ingest ClinVar
│   │   └── vector/
│   │       └── indexing.py         PDF→chunk→embed→pgvector pipeline
│   │
│   ├── models/
│   │   └── embeddings.py           SentenceTransformer wrapper (PubMedBERT)
│   │
│   └── pipeline/
│       ├── decomposition.py        Keyword routing → SubQuery list
│       │
│       ├── construction/           Stubs (planned features)
│       │   ├── self_query.py       [STUB] LLM-powered metadata filter
│       │   └── text_to_sql.py      [STUB] Natural language → SQL
│       │
│       ├── graph/
│       │   ├── state.py            CDSSGraphState TypedDict
│       │   ├── nodes.py            7 node functions
│       │   └── workflow.py         LangGraph DAG compilation
│       │
│       ├── retrieval/
│       │   ├── multi_query.py      LLM expansion + pgvector search + dedup
│       │   ├── reranker.py         BGE cross-encoder + RetrievedChunk factories
│       │   └── crag_evaluator.py   Score-based chunk grading (0.05 / 0.01 thresholds)
│       │
│       ├── generation/
│       │   ├── guardrails.py       System prompt + ⚑ context block builder
│       │   ├── self_rag.py         JSON-schema gen + manifest + critic + thinking strip
│       │   └── citation_enforcer.py Hallucination fix + citation extraction
│       │
│       └── sources/
│           ├── postgres_client.py  ClinVar: get_variant_by_rsid / get_variant_by_gene
│           ├── vector_client.py    pgvector cosine search (named params, pooled)
│           ├── gnomad_client.py    gnomAD GraphQL (allele freq, BA1/PM2)
│           └── clingen_client.py   ClinGen REST + gene symbol extraction
│
├── docs/
│   ├── manifest.json               Parser + source + category metadata per PDF
│   ├── guidelines/                 ACMG_2015_v3.pdf
│   ├── protocols/                  NCCN Breast v2 2026
│   └── screening/                  NCCN Genetic/Familial High-Risk Assessment
│
├── tests/
│   ├── test_routing.py             decompose_query() unit tests
│   ├── test_retrieval.py           reranker + CRAG unit tests
│   ├── test_generation.py          citation enforcer unit tests
│   └── test_clingen_client.py      gene extraction unit tests
│
├── RESULTS/                        Logged pipeline run outputs
├── Problems.md                     Audit findings tracker
├── Solutions.md                    Fix implementations
├── docker-compose.yml              pgvector (pg17) + pgAdmin
├── .env.example
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Docker Desktop | latest | PostgreSQL + pgvector |
| [Ollama](https://ollama.com) | latest | Local LLM inference |

### Step-by-step

**1. Clone**
```bash
git clone https://github.com/RenX86/Privacy-Preserving-CDSS.git
cd Privacy-Preserving-CDSS
```

**2. Install dependencies**
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Pull a local LLM**
```bash
# Best quality for this task (recommended)
ollama pull ministral-3:14b

# Alternatives — smaller but faster
ollama pull llama3.1:8b
ollama pull qwen3.5:9b
```

**4. Configure environment**
```bash
cp .env.example .env
# Open .env and set LOCAL_LLM_MODEL to your chosen model
```

**5. Start PostgreSQL + pgvector**
```bash
docker compose up -d postgres
# Wait ~10s for healthcheck to pass
```

**6. Ingest ClinVar variants**
```bash
# Downloads ~250MB from NCBI FTP, verifies MD5 checksum,
# filters to GRCh38 P/LP/B/LB variants, batch inserts into PostgreSQL
python -m app.db.postgres.clinvar_ingestion
```

**7. Index medical documents**
```bash
# Reads docs/ folder, routes each PDF to correct parser,
# chunks → embeds → inserts into pgvector
python app/db/vector/indexing.py
```

**8. Start the API**
```bash
python -m uvicorn app.main:app --port 5656 --reload
# Open http://localhost:5656/docs for Swagger UI
```

**9. Run a query**
```bash
curl -X POST http://localhost:5656/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the clinical significance of rs879254116 in BRCA1, which ACMG pathogenicity criteria apply, and what cancer screening protocol should the patient follow according to NCCN guidelines? Also confirm the ClinGen expert panel validity for BRCA1."
  }'
```

---

## 🔧 Configuration

```dotenv
# ── PostgreSQL ────────────────────────────────────────────────────
POSTGRES_USER=cdss-user
POSTGRES_PASSWORD=cdss_password
POSTGRES_DB=cdss_db
POSTGRES_PORT=5432
POSTGRES_URL=postgresql://cdss-user:cdss_password@localhost:5432/cdss_db

# ── Local LLM via Ollama ──────────────────────────────────────────
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=ministral-3:14b   # any ollama model — swap freely

# ── Embedding Model ───────────────────────────────────────────────
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# ── ClinGen API ───────────────────────────────────────────────────
CLINGEN_API_URL=https://search.clinicalgenome.org/kb

# ── Privacy flags ─────────────────────────────────────────────────
# false = fully air-gapped, disables live gnomAD allele frequency calls
# true  = enables gnomAD BA1/PM2 criteria (sends chrom-pos-ref-alt only)
ENABLE_GNOMAD_LOOKUP=true
```

**LLM model selection guide:**

| Model | Size | Speed | Quality | Notes |
|---|---|---|---|---|
| `ministral-3:14b` | 9GB | ★★ | ★★★★★ | Best citation compliance + structured output (recommended) |
| `qwen3.5:9b` | 6.6GB | ★★★ | ★★★★ | 256K context, good reasoning, requires Ollama ≥ v0.17 |
| `llama3.1:8b` | 4.7GB | ★★★★ | ★★ | Faster but poor citation compliance and short output |

---

## 📊 Example Query & Response

**Live run log (ministral-3:14b) — with focused sub-queries and per-subquery reranking:**
```
[STARTING LANGGRAPH] Query: What is the clinical significance of rs879254116
  in BRCA1, which ACMG pathogenicity criteria apply, and what cancer
  screening protocol should the patient follow according to NCCN guidelines.
  Also confirm the ClinGen expert panel validity for BRCA1.

[Decomposer] ACMG sub-query: ACMG pathogenicity criteria variant
  classification rules PVS1 PS1 PS2 PS3 PS4 PM1 PM2...
[Decomposer] Screening sub-query: NCCN cancer screening surveillance
  mammography MRI risk-reducing prophylactic salpingo-oophorectomy BRCA1...

PostgreSQL connection pool initialised (min=2, max=10)
[gnomAD] Querying variant 17-43049159-GA-G → Variant not found
[ClinGen] 1 record(s) for BRCA1
  symbol=BRCA1 | validity=True | actionability=True | curated=08/29/2024

[MultiQuery] ACMG expansion — 3 targeted queries (JSON schema):
  • ACMG PVS1 criteria loss of function variants null alleles
  • ACMG PP3 PP4 moderate pathogenic evidence computational prediction
  • ACMG BS1 BS2 BS3 benign evidence population frequency gnomAD
4 queries × 15 results = 35 unique chunks
[PDF] Reranked 35 chunks against: ACMG pathogenicity criteria PVS1...

[MultiQuery] Screening expansion — 3 targeted queries (JSON schema):
  • BRCA1/2 carriers NCCN guidelines mammography MRI surveillance ages
  • risk-reducing salpingo-oophorectomy (RRSO) timing mastectomy
  • hereditary breast and ovarian cancer surveillance protocols
4 queries × 15 results = 30 unique chunks
[PDF] Reranked 30 chunks against: NCCN cancer screening surveillance...

[CRAG] Grading 48 PDF chunks:    ← total from both sub-queries
  0.824 [CORRECT]  BRCA PATHOGENIC/LIKELY PATHOGENIC VARIANT-POSITIVE MANAGEMENT
  0.745 [CORRECT]  NCCN Guidelines Version 3.2026...
  0.699 [CORRECT]  Bilateral Salpingo-Oophorectomy
  0.601 [CORRECT]  ACMG_2015 page 33 — Table 3 Criteria for Classifying...
  0.465 [CORRECT]  ACMG_2015 page 10 — PM1–6, PP1–5 weight definitions
  0.440 [CORRECT]  ACMG_2015 page 36 — Table 5 Rules for Combining Criteria
  ...                                   ← 9 additional CORRECT chunks
  0.009 [INCORRECT] → dropped

[CRAG] Kept 37 chunks | Dropped 11 INCORRECT chunks

Calling Ollama [ministral-3:14b] (3 DB + 12 PDF chunks)
Draft JSON generated (4016 chars)

  ACMG output (definitions, not applied):
  • PVS1: Null variant in a gene where LoF is known mechanism of disease.
  • PM2: Absent from controls in population databases.
  • PP5: Reputable source classifies the variant as pathogenic.

  Screening output (BRCA1-specific only):
  • Annual breast MRI + mammography starting at age 25-29
  • Risk-reducing salpingo-oophorectomy between ages 35-40
  • ❌ NO colonoscopy (table row isolation prevented cross-gene slippage)

LLM Critic finished. Length: 3217 → 3246
[Citation] Verified DB chunks: 3 | Verified PDF chunks: 12 → confidence: high
[LANGGRAPH FINISHED] Confidence: high
```

---

## 🗺️ Roadmap

**Completed:**
- [x] PostgreSQL schema + ClinVar bulk ingestion with MD5 verification
- [x] pgvector setup + ACMG/NCCN document indexing (dual-parser: PyMuPDF + Docling)
- [x] Query decomposition — keyword routing to typed SubQuery list
- [x] LangGraph 7-node DAG with parallel Send-based fan-out
- [x] Multi-query expansion with per-category prompt templates
- [x] BGE cross-encoder reranking (`BAAI/bge-reranker-large`)
- [x] CRAG evaluator with empirically tuned thresholds (0.05 / 0.01)
- [x] gnomAD v4 GraphQL client + BA1/PM2 rule application
- [x] ClinGen REST API client + gene symbol extraction
- [x] JSON-schema-constrained generation (ClinicalResponse Pydantic schema)
- [x] Citation manifest injection (prevents hallucinated references)
- [x] Self-RAG single-pass critic with citation safety net
- [x] Thinking-mode stripping (`<think>` blocks)
- [x] PostgreSQL connection pooling (ThreadedConnectionPool)
- [x] gnomAD opt-out flag for air-gapped deployments
- [x] **Focused sub-query text** — each sub-query gets topic-specific search text (not full query)
- [x] **Per-subquery reranking** — chunks reranked against their own topic before CRAG
- [x] **Anti-hallucination guardrails** (Rules 5–7):
  - Rule 5: Never invent variant biology (frameshift/missense/de novo/computational)
  - Rule 6: Table row isolation — verify gene name in every NCCN table row
  - Rule 7: ACMG definition-only reporting — report definitions, don't apply criteria
- [x] **ClinGen validity field** in ClinicalResponse schema
- [x] **Screening/protocol mutual exclusion** — prevents cross-category retrieval noise

**In progress / planned:**
- [ ] Full unit test suite (test files stubbed)
- [ ] Systematic evaluation framework (Ragas/custom faithfulness metrics)
- [ ] Self-query metadata filter (`construction/self_query.py`)
- [ ] Text-to-SQL dynamic query builder (`construction/text_to_sql.py`)
- [ ] Frontend UI (`frontend/`)
- [ ] Multi-pass Self-RAG critic loop (currently single-pass)
- [ ] gnomAD local cache (pre-fetch at index time, eliminate runtime external call)
- [ ] Clinical validation with domain experts

---

## ⚠️ Disclaimer

This system is a **clinical decision support tool** only. It does not replace professional medical judgment. All outputs must be reviewed by a qualified clinician before use in patient care.

---

<div align="center">
  <sub>Built with ❤️ for safer, more transparent clinical AI</sub>
</div>