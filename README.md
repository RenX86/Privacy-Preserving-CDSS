# 🧬 Privacy-Preserving Clinical Decision Support System (CDSS)

> A highly rigorous, locally-hosted **Hybrid RAG pipeline** for clinical genomic queries — combining air-gapped local databases with targeted live API calls, built to prevent hallucinations, enforce auditability, and never expose patient data externally.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-ClinVar_Local-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![VectorDB](https://img.shields.io/badge/Vector_DB-ACMG_%2F_NCCN-1ABC9C?style=flat)]()
[![ClinGen](https://img.shields.io/badge/ClinGen_API-Live_Expert_Panels-6C3483?style=flat)]()
[![LLM](https://img.shields.io/badge/LLM-Local_Only-8E44AD?style=flat)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)]()
[![Status](https://img.shields.io/badge/Status-In_Development-orange?style=flat)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Why Hybrid RAG?](#-why-hybrid-rag)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [Phase 1 — Query Decomposition](#phase-1--query-decomposition)
  - [Phase 2 — Intelligent Routing](#phase-2--intelligent-routing)
  - [Phase 3 — Query Construction](#phase-3--query-construction)
  - [Phase 4 — Data Sources](#phase-4--data-sources)
  - [Phase 5 — Retrieval & Verification](#phase-5--retrieval--verification)
  - [Phase 6 — Protected Generation](#phase-6--protected-generation)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🔬 Overview

This system answers complex clinical and genomic queries — such as BRCA1 variant classification, ACMG criteria application, and NCCN protocol retrieval — **with zero hallucinations and full data privacy**.

**Key design principles:**

- 🔒 **Local-first** — all LLM inference runs on-premise; patient data never leaves the secure environment
- 🌐 **Hybrid RAG** — air-gapped local databases for static data + targeted live ClinGen API calls for real-time expert panel rules
- ⚡ **Deterministic routing** — regex fast-path for structured IDs before any LLM is invoked
- ✅ **Verifiable retrieval** — every retrieved chunk is graded (Correct / Ambiguous / Incorrect) before generation
- 📎 **Mandatory citations** — every clinical claim in the output is traceable to a primary source
- 🚫 **Hard safe-failure** — insufficient data produces a predefined refusal, never a hallucinated guess

**Example query this system handles:**

> *"What is the clinical significance of rs28897696, what are the ACMG criteria for pathogenic BRCA1 mutations, and what NCCN oncology protocols apply for breast cancer with BRCA1?"*

---

## 🏗️ Architecture

The system is a 6-phase sequential pipeline. Each phase has strict entry/exit conditions before passing context downstream.

```
Clinician Query
      │
      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Phase 1        │    │  Phase 2         │    │  Phase 3            │
│  Query          │───▶│  Intelligent     │───▶│  Query              │
│  Decomposition  │    │  Routing         │    │  Construction       │
└─────────────────┘    └──────────────────┘    └──────────┬──────────┘
                              │                            │
                              │ (ClinGen bypass)           │
                              │                            │
                              ▼                            ▼
                   ┌─────────────────────────────────────────────────┐
                   │              Phase 4 — Data Sources              │
                   │                                                   │
                   │  🗄️ PostgreSQL (ClinVar)   ✅ Local, air-gapped  │
                   │  🔮 Vector DB (ACMG/NCCN)  ✅ Local, air-gapped  │
                   │  🌐 ClinGen REST API        ⚠️ Live external call │
                   └─────────────────────────────────────────────────┘
                                        │
                                        ▼
                                        │
                                        ▼
                   ┌──────────────────────────────────────────────────┐
                   │           Phase 5 — Retrieval & Verification      │
                   │                                                    │
                   │  LangGraph State Machine Orchestration            │
                   │  Cross-Encoder Re-Ranking                         │
                   │  CRAG Evaluator (Correct / Ambiguous / Incorrect) │
                   │  Fallback loop → re-query if ambiguous/incorrect  │
                   └──────────────────────────────────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────────────┐
                   │          Phase 6 — Protected Generation           │
                   │                                                    │
                   │  Strict System Prompting                          │
                   │  Self-RAG Internal Critic                         │
                   │  Forced Citations (Audit Trail)                   │
                   └──────────────────────────────────────────────────┘
                                        │
                                        ▼
                          ✅ Verified Clinical Response
                             with Mandatory Citations
```

---

## 🔀 Why Hybrid RAG?

This architecture uses **three distinct data sources**, each chosen for a specific reason:

| Data Source | Type | Why |
|---|---|---|
| 🗄️ **PostgreSQL (ClinVar)** | Local, air-gapped | ClinVar contains millions of static variant records that can be downloaded and queried locally via SQL. Fast, private, no external calls. |
| 🔮 **Vector DB (ACMG / NCCN)** | Local, air-gapped | Clinical guidelines are static documents that are chunked, embedded with domain-specific models (PubMedBERT), and stored locally for semantic retrieval. |
| 🌐 **ClinGen REST API** | Live, external | ClinGen Expert Panels publish **constantly-updating, gene-specific** ACMG rule adaptations. These are too specialized and dynamic to replicate locally — a live API call is the only way to guarantee up-to-date accuracy. **No patient data is sent in this call.** |

> This is what makes the system a **Hybrid RAG**: it combines the privacy and speed of air-gapped local retrieval with the accuracy of targeted real-time external API calls — only when medically necessary.

---

## 🔄 Pipeline Walkthrough

### Phase 1 — Query Decomposition

An incoming clinical question is decomposed into typed sub-queries, each targeting a specific data source.

| Sub-Query | Type | Destination |
|---|---|---|
| **Sub-Query 1** | Data Extraction | Detect `rs28897696` → route to PostgreSQL for ClinVar facts |
| **Sub-Query 2** | Rule Retrieval | Fetch ACMG criteria for pathogenic BRCA1 mutation → Vector DB (Guidelines) |
| **Sub-Query 3** | Protocol Retrieval | Fetch NCCN protocols for breast cancer + BRCA1 → Vector DB (Protocols) |
| **Sub-Query 4** | Screening Retrieval | Fetch high-risk surveillance for BRCA1 → Vector DB (Screening) |

---

### Phase 2 — Intelligent Routing

A two-layer routing system directs each sub-query to the correct data source without unnecessary LLM calls.

#### Layer 1 — Deterministic Routing (Regex)

```
FastAPI backend → Regex scan → Detects structured IDs (e.g. rs28897696, NM_007294.4)
                                          │
                            Matched ──────┴────── Not matched
                               │                       │
                               ▼                       ▼
                         PostgreSQL DB          Layer 2 (LLM Router)
```

If a strict variant ID or structured keyword is detected, the sub-query is routed immediately to PostgreSQL — **no LLM invoked, no latency**.

#### Layer 2 — Logical Routing (Local LLM)

If no strict ID is detected, a local LLM performs semantic routing between **three targets**:

| Target | When Used |
|---|---|
| **Vector DB** | Query is about classification rules, guidelines, or protocols (ACMG, NCCN) |
| **ClinGen API** | Query requires real-time, gene-specific expert panel adaptations |
| **PostgreSQL** | Query contains a structured ID missed by Layer 1 |

---

### Phase 3 — Query Construction

Routed sub-queries are translated into the native query language of their target. **ClinGen API calls bypass this phase entirely** — they go directly from the router to the API.

#### Relational DB Path — Text-to-SQL

```sql
-- Input: "Get the clinical significance for rs28897696"
SELECT clinical_significance
FROM clinvar
WHERE rsid = 'rs28897696';
```

#### Vector DB Path — Self-Query Retriever

An LLM simultaneously generates a vector embedding **and** hard metadata filters:

```json
{
  "query": "PVS1 rule for BRCA1 pathogenic mutations",
  "filter": {
    "gene": "BRCA1",
    "source": "ACMG_2015"
  }
}
```

> The metadata filter guarantees the Vector DB only searches within the exact right document namespace — preventing cross-contamination between guidelines.

#### ClinGen API Path — Direct Bypass

```
Layer 2 Router ──(dashed bypass)──▶ ClinGen REST API
                  No SQL or embedding
                  construction needed
```

---

### Phase 4 — Data Sources

All three data sources return context to the verification layer.

#### 🗄️ PostgreSQL — ClinVar (Local)

Stores structured variant records. Queried via the SQL command constructed in Phase 3.

#### 🔮 Vector Database — Medical Protocols (Local)

Stores chunked and embedded ACMG guidelines, NCCN protocols, and NCCN screening surveillance guidelines. Indexed using:

- **Source Separation:** Documents are grouped into three distinct namespaces: `guideline`, `protocol`, and `screening_protocol`.
- **Chunking:** Semantic / Markdown Header splitters preserving full rule definitions within each chunk
- **Embeddings:** Domain-specific models — PubMedBERT, ClinicalBERT, MedEIR

#### 🌐 ClinGen REST API — Expert Panels (Live External)

Returns real-time, gene-specific ACMG rule adaptations from ClinGen's curated expert panels.

```http
GET https://search.clinicalgenome.org/kb/gene-validity?gene=BRCA1
```

> ⚠️ **Privacy note:** Only the gene symbol (e.g. `BRCA1`) is sent to ClinGen. No patient identifiers, variant IDs, or clinical data are transmitted in this call.

---

### Phase 5 — Retrieval & Verification

All context — from PostgreSQL, Vector DB, **and ClinGen** — passes through the same two verification stages.

#### Cross-Encoder Re-Ranking

```
Top-K retrieved chunks (from all sources)
              │
              ▼
Cross-Encoder (ms-marco-MiniLM-L-6-v2)
  → Computes deep query ↔ chunk relevance score
  → Re-ranks all chunks by clinical accuracy
              │
              ▼
Most relevant chunk promoted to position [0]
```

#### CRAG Evaluator Agent

A lightweight evaluator grades each chunk independently, regardless of source:

| Grade | Action |
|---|---|
| ✅ **Correct** | Pass to generation phase |
| ⚠️ **Ambiguous** | Trigger fallback — re-query with refined parameters |
| ❌ **Incorrect** | Reject chunk — do not pass downstream |

---

### Phase 6 — Protected Generation

Verified context enters generation with three enforced guardrails.

#### Guardrail 1 — Strict System Prompting

The LLM is instructed to only synthesize claims **explicitly present** in the provided verified context. If context is insufficient:

```
"Insufficient clinical data to safely provide an assessment."
```

#### Guardrail 2 — Self-RAG Internal Critic

```
Generate hidden draft response
          │
          ▼
Is every claim supported by a verified chunk?
          │
    ┌─────┴─────┐
   Yes          No
    │           │
    ▼           ▼
  Pass       Reject → Rewrite automatically
```

#### Guardrail 3 — Forced Citations (Audit Trail)

Every factual claim in the final response includes an inline citation. ClinGen sources are cited alongside local sources:

```
rs28897696 is classified as Pathogenic. [Source: ClinVar, rs28897696]
ACMG criteria applied: PVS1 (very strong), PS1 (strong). [Source: ACMG_2015, Richards et al.]
ClinGen BRCA1 Expert Panel adaptation applied: domain-specific PM2 threshold. [Source: ClinGen, BRCA1-EP]
NCCN recommendation: Annual breast MRI + mammography from age 25. [Source: NCCN, Breast-v3.2024]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI (Python) |
| **Relational Database** | PostgreSQL |
| **Vector Database** | Qdrant / Weaviate / ChromaDB |
| **Local LLM** | Ollama / vLLM (self-hosted) |
| **Embeddings** | PubMedBERT · ClinicalBERT · MedEIR |
| **Re-Ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **RAG Framework** | LangGraph / LangChain |
| **Live External API** | ClinGen REST API (Expert Panels) |
| **Local Data Sources** | ClinVar (PostgreSQL) · ACMG Guidelines · NCCN Protocols · NCCN Screening (Vector DB) |

---

## 📁 Project Structure

```
cdss/
├── api/
│   ├── main.py                    # FastAPI entrypoint
│   ├── router.py                  # Layer 1 (Regex) & Layer 2 (LLM) routing
│   └── schemas.py                 # Request/response models
├── pipeline/
│   ├── decomposition.py           # Query decomposition
│   ├── graph/                     # LangGraph State Machine
│   │   ├── nodes.py               # Graph nodes
│   │   ├── state.py               # Graph state definition
│   │   └── workflow.py            # Graph compilation
│   ├── construction/
│   │   ├── text_to_sql.py         # SQL query builder (PostgreSQL path)
│   │   └── self_query.py          # Self-Query Retriever (Vector DB path)
│   ├── sources/
│   │   ├── postgres_client.py     # ClinVar structured queries
│   │   ├── vector_client.py       # ACMG / NCCN semantic retrieval
│   │   └── clingen_client.py      # ClinGen REST API client (live)
│   ├── retrieval/
│   │   ├── reranker.py            # Cross-encoder re-ranking
│   │   └── crag_evaluator.py      # CRAG chunk grader
│   └── generation/
│       ├── guardrails.py          # Strict system prompt enforcement
│       ├── self_rag.py            # Internal critic loop
│       └── citation_enforcer.py   # Mandatory citation injection
├── db/
│   ├── postgres/
│   │   └── clinvar_schema.sql     # ClinVar structured schema
│   └── vector/
│       └── indexing.py            # Chunking + embedding pipeline
├── models/
│   └── embeddings.py              # Domain-specific embedding loaders
├── tests/
│   ├── test_routing.py
│   ├── test_clingen_client.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── docs/
│   └── CDSS_Architecture.drawio   # System architecture diagram
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Docker (recommended for Vector DB)
- Ollama or vLLM for local LLM hosting

### 1. Clone the repository

```bash
git clone https://github.com/RenX86/cdss.git
cd cdss
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

```env
# Local databases
POSTGRES_URL=postgresql://user:password@localhost:5432/clinvar
VECTOR_DB_URL=http://localhost:6333

# Local LLM
LOCAL_LLM_URL=http://localhost:11434
EMBEDDING_MODEL=pubmedbert-base-embeddings

# ClinGen API (no auth required for public endpoints)
CLINGEN_API_URL=https://search.clinicalgenome.org/kb
```

### 4. Initialize databases

```bash
# PostgreSQL — ClinVar schema
psql -U postgres -f db/postgres/clinvar_schema.sql

# Vector DB — start with Docker
docker run -p 5757:5757 qdrant/qdrant

# Index medical documents into Vector DB
python db/vector/indexing.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 5656
```

### 6. Test a query

```bash
curl -X POST http://localhost:5656/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the clinical significance of rs28897696 for BRCA1?"}'
```

---

## 🗺️ Roadmap

- [x] System architecture design (Hybrid RAG)
- [x] Project work plan & phase definitions
- [x] Phase 1 — PostgreSQL schema + ClinVar data ingestion
- [x] Phase 1 — Vector DB setup + ACMG/NCCN document indexing (PyMuPDF4LLM & Docling)
- [x] Phase 2 — Layer 1 Regex routing (FastAPI)
- [x] Phase 2 — Layer 2 LLM logical router (LangGraph orchestration)
- [x] Phase 3 — Text-to-SQL constructor
- [x] Phase 3 — Multi-Query Retriever with contextual metadata filters
- [ ] Phase 4 — ClinGen REST API client with privacy-safe call pattern
- [x] Phase 5 — Cross-encoder re-ranker (all sources unified)
- [x] Phase 5 — CRAG evaluator agent + Context limit caps
- [x] Phase 6 — Strict system prompt + Pydantic structured output
- [x] Phase 6 — Self-RAG internal LLM-based critic
- [x] Phase 6 — Citation enforcer (ClinVar, ACMG, NCCN)
- [ ] End-to-end integration testing
- [ ] Clinical validation with domain experts
- [ ] User Acceptance Testing (UAT)

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss any significant changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/clingen-api-client`)
3. Commit your changes (`git commit -m 'Add ClinGen REST API client with privacy-safe pattern'`)
4. Push to the branch (`git push origin feature/clingen-api-client`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This system is intended as a **clinical decision support tool** only. It does not replace professional medical judgment. All outputs must be reviewed by a qualified clinician before use in patient care.

---

<div align="center">
  <sub>Built with ❤️ for safer, more transparent clinical AI</sub>
</div>
