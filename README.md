# 🧬 Privacy-Preserving Clinical Decision Support System (CDSS)

> A highly rigorous, locally-hosted RAG pipeline for clinical genomic queries — built to prevent hallucinations, enforce auditability, and never expose patient data externally.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Structured_DB-336791?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![LLM](https://img.shields.io/badge/LLM-Local_Only-8E44AD?style=flat&logo=openai&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)]()
[![Status](https://img.shields.io/badge/Status-In_Development-orange?style=flat)]()

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [Phase 1 — Query Decomposition](#phase-1--query-decomposition)
  - [Phase 2 — Intelligent Routing](#phase-2--intelligent-routing)
  - [Phase 3 — Query Construction](#phase-3--query-construction)
  - [Phase 4 — Retrieval & Verification](#phase-4--retrieval--verification)
  - [Phase 5 — Protected Generation](#phase-5--protected-generation)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)

---

## 🔬 Overview

This system is designed to answer complex clinical and genomic queries — such as BRCA1 variant classification, ACMG criteria application, and NCCN protocol retrieval — **with zero hallucinations and full data privacy**.

**Key design principles:**

- 🔒 **Local-first** — all LLM inference runs on-premise; patient data never leaves the secure environment
- 🎯 **Deterministic routing** — regex-based fast-path for structured IDs before any LLM is invoked
- ✅ **Verifiable retrieval** — every retrieved chunk is graded before it reaches the generator
- 📎 **Mandatory citations** — every clinical claim in the output is traceable to a primary source
- 🚫 **Hard safe-failure** — insufficient data produces a predefined refusal, never a guess

**Example query this system handles:**

> *"What is the clinical significance of rs28897696, what are the ACMG criteria for pathogenic BRCA1 mutations, and what NCCN oncology protocols apply for breast cancer with BRCA1?"*

---

## 🏗️ Architecture

The system is a 5-phase sequential pipeline. Each phase has strict entry/exit conditions before passing context downstream.

```
Clinician Query
      │
      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Phase 1        │    │  Phase 2         │    │  Phase 3            │
│  Query          │───▶│  Intelligent     │───▶│  Query              │
│  Decomposition  │    │  Routing         │    │  Construction       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                         │
                              ┌──────────────────────────┤
                              │                          │
                              ▼                          ▼
                       PostgreSQL DB              Vector Database
                       (ClinVar facts)          (ACMG / NCCN rules)
                              │                          │
                              └──────────┬───────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐    ┌──────────────────────┐
                              │  Phase 4             │    │  Phase 5             │
                              │  Retrieval &         │───▶│  Protected           │
                              │  Verification        │    │  Generation          │
                              └──────────────────────┘    └──────────────────────┘
                                                                    │
                                                                    ▼
                                                       ✅ Verified Clinical Response
                                                          with Mandatory Citations
```

---

## 🔄 Pipeline Walkthrough

### Phase 1 — Query Decomposition

An incoming clinical question is broken down into typed sub-queries, each targeting a specific data source.

| Sub-Query | Type | Target |
|---|---|---|
| **Sub-Query 1** | Data Extraction | Extract variant ID `rs28897696` → PostgreSQL (ClinVar facts) |
| **Sub-Query 2** | Rule Retrieval | Fetch ACMG criteria for pathogenic BRCA1 mutation → Vector DB |
| **Sub-Query 3** | Protocol Retrieval | Fetch NCCN oncology protocols for breast cancer + BRCA1 → Vector DB |

---

### Phase 2 — Intelligent Routing

A two-layer routing system directs each sub-query to the correct data source without wasting LLM calls.

#### Layer 1 — Deterministic Routing (Regex)

```
FastAPI backend → Regex scan → Detects structured IDs (e.g. rs28897696, NM_007294.4)
                                        │
                          Matched ──────┴────── Not matched
                             │                       │
                             ▼                       ▼
                       PostgreSQL DB          Layer 2 (LLM Router)
```

A fast Python regex scan runs first. If a strict variant ID or structured keyword is found, the sub-query is routed immediately to PostgreSQL — **no LLM invoked**.

#### Layer 2 — Logical Routing (Local LLM)

If no strict ID is detected, a local LLM performs semantic routing between:

- **Vector DB** — for rule-based queries (ACMG criteria, NCCN guidelines)
- **ClinGen API** — for gene-specific curations and expert panel adaptations

---

### Phase 3 — Query Construction

Each routed sub-query is translated into the native query language of its target database.

#### Relational DB Path — Text-to-SQL

An LLM or strict Python script converts the English sub-query into a SQL command:

```sql
-- Input: "Get the clinical significance for rs28897696"
SELECT clinical_significance
FROM clinvar
WHERE rsid = 'rs28897696';
```

#### Vector DB Path — Self-Query Retriever

An LLM simultaneously generates:

1. A **vector embedding** for semantic similarity search
2. **Hard metadata filters** extracted from the query to restrict the search space

```json
{
  "query": "PVS1 rule for BRCA1 pathogenic mutations",
  "filter": {
    "gene": "BRCA1",
    "source": "ACMG_2015"
  }
}
```

> This dual approach guarantees the Vector DB only searches within the exact right document namespace — preventing cross-contamination and hallucinations from unrelated guidelines.

---

### Phase 4 — Retrieval & Verification

Retrieved context goes through two verification stages before it can reach the generator.

#### Cross-Encoder Re-Ranking

```
Top-K retrieved chunks
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

A lightweight evaluator grades each chunk independently:

| Grade | Action |
|---|---|
| ✅ **Correct** | Pass to generation phase |
| ⚠️ **Ambiguous** | Trigger fallback — re-query with refined parameters |
| ❌ **Incorrect** | Reject chunk — do not pass downstream |

---

### Phase 5 — Protected Generation

Verified context enters the generation phase, which enforces three layers of guardrails.

#### Guardrail 1 — Strict System Prompting

The LLM is instructed via a rigid system prompt to:

- Only synthesize claims **explicitly present** in the provided context
- Never use external or parametric knowledge
- Return a safe-failure message if context is insufficient:

```
"Insufficient clinical data to safely provide an assessment."
```

#### Guardrail 2 — Self-RAG Internal Critic

```
Generate hidden draft response
          │
          ▼
Verify: Is every claim supported by a retrieved chunk?
          │
    ┌─────┴─────┐
    │           │
   Yes          No
    │           │
    ▼           ▼
  Pass       Reject draft → Rewrite
```

#### Guardrail 3 — Forced Citations (Audit Trail)

Every factual claim in the final response must include an inline citation to its exact source:

```
rs28897696 is classified as Pathogenic. [Source: ClinVar, rs28897696]
ACMG criteria applied: PVS1 (very strong), PS1 (strong). [Source: ACMG_2015, Richards et al.]
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
| **RAG Framework** | LangChain / LlamaIndex |
| **Data Sources** | ClinVar · ClinGen API · ACMG Guidelines · NCCN Protocols |

---

## 📁 Project Structure

```
cdss/
├── api/
│   ├── main.py                  # FastAPI entrypoint
│   ├── router.py                # Layer 1 & 2 routing logic
│   └── schemas.py               # Request/response models
├── pipeline/
│   ├── decomposition.py         # Query decomposition
│   ├── construction/
│   │   ├── text_to_sql.py       # SQL query builder
│   │   └── self_query.py        # Vector self-query retriever
│   ├── retrieval/
│   │   ├── reranker.py          # Cross-encoder re-ranking
│   │   └── crag_evaluator.py    # CRAG chunk grader
│   └── generation/
│       ├── guardrails.py        # System prompt enforcement
│       ├── self_rag.py          # Internal critic loop
│       └── citation_enforcer.py # Mandatory citation injection
├── db/
│   ├── postgres/
│   │   └── clinvar_schema.sql   # ClinVar structured schema
│   └── vector/
│       └── indexing.py          # Chunking + embedding pipeline
├── models/
│   └── embeddings.py            # Domain-specific embedding loaders
├── tests/
│   ├── test_routing.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── docs/
│   └── architecture.drawio      # System architecture diagram
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
git clone https://github.com/your-username/cdss.git
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
# Edit .env with your DB credentials and LLM endpoint
```

```env
POSTGRES_URL=postgresql://user:password@localhost:5432/clinvar
VECTOR_DB_URL=http://localhost:6333
LOCAL_LLM_URL=http://localhost:11434
EMBEDDING_MODEL=pubmedbert-base-embeddings
```

### 4. Initialize databases

```bash
# PostgreSQL
psql -U postgres -f db/postgres/clinvar_schema.sql

# Vector DB (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Index medical documents
python db/vector/indexing.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Test a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the clinical significance of rs28897696 for BRCA1?"}'
```

---

## 🗺️ Roadmap

- [x] System architecture design
- [x] Project work plan & phase definitions
- [ ] Phase 1 — PostgreSQL schema + ClinVar data ingestion
- [ ] Phase 1 — Vector DB setup + medical document indexing
- [ ] Phase 2 — Regex routing layer (FastAPI)
- [ ] Phase 2 — LLM logical router
- [ ] Phase 3 — Text-to-SQL constructor
- [ ] Phase 3 — Self-Query Retriever
- [ ] Phase 4 — Cross-encoder re-ranker integration
- [ ] Phase 4 — CRAG evaluator agent
- [ ] Phase 5 — Self-RAG internal critic
- [ ] Phase 5 — Citation enforcer
- [ ] End-to-end integration testing
- [ ] Clinical validation with domain experts
- [ ] User Acceptance Testing (UAT)

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first to discuss any significant changes.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/phase-2-routing`)
3. Commit your changes (`git commit -m 'Add LLM logical router'`)
4. Push to the branch (`git push origin feature/phase-2-routing`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This system is intended as a **clinical decision support tool** only. It does not replace professional medical judgment. All outputs must be reviewed by a qualified clinician before use in patient care.

---

<div align="center">
  <sub>Built with ❤️ for safer, more transparent clinical AI</sub>
</div>
