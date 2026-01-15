# Development of a Privacy-Preserving Clinical Decision Support System (CDSS)

## Using Retrieval-Augmented Generation (RAG) and Vector Databases

------------------------------------------------------------------------

## Abstract

Clinical Decision Support Systems (CDSS) play a critical role in
supporting clinicians with diagnosis, prognosis, and treatment planning.
However, most contemporary AI-driven CDSS solutions rely on cloud-hosted
large language models (LLMs), which introduce substantial risks
including patient data leakage, lack of transparency, hallucinated
clinical facts, and limited control over model updates. These
limitations prevent safe adoption in healthcare environments governed by
strict regulatory frameworks such as HIPAA and GDPR.

This dissertation proposes the development of a **fully local,
privacy-preserving Clinical Decision Support System** using a
**Retrieval-Augmented Generation (RAG)** architecture. Verified clinical
guidelines and curated biomedical datasets are embedded and stored in a
**locally hosted PostgreSQL database with pgvector**. A locally deployed
LLM performs inference on an offline GPU (NVIDIA RTX 3060), ensuring
zero external data transmission. By grounding all responses in retrieved
authoritative sources, the system minimizes hallucinations while
improving explainability, traceability, and regulatory compliance.

------------------------------------------------------------------------

## Keywords

Clinical Decision Support System, Privacy-Preserving AI,
Retrieval-Augmented Generation, Vector Database, Local LLM,
Bioinformatics

------------------------------------------------------------------------

## 1. Introduction

### 1.1 Background

Artificial Intelligence (AI) and Natural Language Processing (NLP) have
demonstrated strong potential in automating knowledge extraction and
reasoning over biomedical literature. Large language models are capable
of synthesizing complex medical information, yet their deployment in
healthcare remains limited due to concerns around privacy, reliability,
and explainability.

Healthcare environments require deterministic behavior, auditable
outputs, and strict data isolation. Cloud-hosted AI systems typically
fail to meet these requirements.

### 1.2 Problem Statement

Current cloud-based AI systems suffer from:

1.**Hallucinations:** Fabrication of unsupported medical facts.
2.**Privacy Risks:** Transmission of sensitive patient data tothird-party servers.
3.**Static Knowledge:** Limited awareness of updated clinicalguidelines.
4.**Lack of Explainability:** Inability to trace responses back tosource documents.

These risks directly impact patient safety and regulatory compliance.

### 1.3 Proposed Solution

This project proposes a **local, offline Clinical Decision Support
System** using a Retrieval-Augmented Generation pipeline. The system
retrieves relevant content from verified medical documents stored
locally and constrains the LLM to generate answers strictly grounded in
retrieved evidence.

------------------------------------------------------------------------

## 2. Objectives

1.Design a privacy-preserving CDSS architecture using local inference.
2.Implement a Retrieval-Augmented Generation pipeline.
3.Store embeddings using PostgreSQL with pgvector.
4.Reduce hallucinations through evidence grounding.
5.Evaluate system performance using quantitative metrics.

------------------------------------------------------------------------

## 3. Literature Review

### 3.1 Clinical Decision Support Systems

Traditional CDSS rely on rule-based engines and structured databases,
offering high reliability but limited scalability with unstructured
text.

### 3.2 Large Language Models in Healthcare

LLMs have shown promise in summarization and medical question answering
but raise concerns about hallucination and privacy leakage.

### 3.3 Retrieval-Augmented Generation

RAG integrates document retrieval with text generation, improving
factual grounding and traceability.

### 3.4 Vector Databases in Biomedical NLP

Vector databases enable semantic similarity search across large
biomedical corpora. pgvector allows tight integration with relational
databases.

------------------------------------------------------------------------

## 4. System Architecture

### 4.1 High-Level Components

1.**Local Language Model:** Offline inference on GPU.
2.**Vector Database:** PostgreSQL + pgvector running locally viaDocker.
3.**Workflow Orchestrator:** Self-hosted n8n.
4.**User Interface:** Streamlit frontend.
5.**Containerization:** Docker Compose.

### 4.2 Deployment Strategy

- Entire system runs locally using Docker.
- Optional tunneling via ngrok or Cloudflare Tunnel fordemonstrations.
- No cloud-based GPU or database services are used.

------------------------------------------------------------------------

## 5. Technology Stack

  Component                Technology
  ------------------------ --------------------------------------
  Programming Language     Python
  Language Model           LLaMA-3 / BioMistral (Ollama)
  Embeddings               Sentence-Transformers
  Vector Database          PostgreSQL + pgvector (Local Docker)
  Workflow Orchestration   n8n (Self-hosted)
  Frontend                 Streamlit
  Containerization         Docker Compose
  Hardware                 NVIDIA RTX 3060

------------------------------------------------------------------------

## 6. Methodology

### 6.1 Local Model Deployment

LLM deployed locally using Ollama with GPU acceleration.

### 6.2 Knowledge Base Construction

- Document ingestion
- Cleaning and normalization
- Chunking (\~500 words)
- Embedding generation
- Storage in pgvector

### 6.3 Retrieval-Augmented Generation

- Query embedding
- Top-k semantic retrieval
- Context injection
- Evidence-grounded generation

### 6.4 Workflow Automation

n8n orchestrates ingestion, retrieval, and response generation workflows
locally.

### 6.5 User Interface

Streamlit provides clinician-facing interaction and source inspection.

------------------------------------------------------------------------

## 7. Datasets

  Source            Purpose
  ----------------- ------------------------
  ACMG Guidelines   Variant interpretation
  NCCN Guidelines   Oncology treatment
  ClinVar           Mutation reference
  CIViC             Clinical evidence

------------------------------------------------------------------------

## 8. Evaluation Strategy

### 8.1 Metrics

- Precision@k
- Hallucination Rate
- Faithfulness
- Latency
- Privacy Compliance

### 8.2 Golden Dataset

A manually curated dataset of 20--30 question-answer pairs derived from
official guidelines will serve as ground truth.

------------------------------------------------------------------------

## 9. Ethical and Privacy Considerations

- No patient-identifiable data stored.
- All computation remains local.
- AI assists decision-making, not diagnosis.

------------------------------------------------------------------------

## 10. Limitations

- Knowledge limited to ingested documents.
- Local hardware constraints.
- Not a replacement for clinicians.

------------------------------------------------------------------------

## 11. Future Work

- EHR integration
- Multimodal data
- Federated learning

------------------------------------------------------------------------

## 12. Conclusion

This project demonstrates the feasibility of building a
privacy-preserving, explainable CDSS using modern AI techniques while
maintaining regulatory compliance.

------------------------------------------------------------------------

cdss-project/
│
├── data/                          # All data files
│   ├── raw/                       # Original PDF documents
│   │   ├── acmg_guidelines.pdf
│   │   ├── nccn_guidelines.pdf
│   │   └── README.md             # Document sources and descriptions
│   │
│   └── processed/                 # Processed/chunked text files
│       ├── acmg_chunks.json
│       └── metadata.json
│
├── src/                           # Source code
│   ├── **init**.py
│   │
│   ├── config/                    # Configuration files
│   │   ├── **init**.py
│   │   └── settings.py           # Centralized settings
│   │
│   ├── database/                  # Database operations
│   │   ├── **init**.py
│   │   ├── db_connection.py      # Database connection handler
│   │   └── db_operations.py      # CRUD operations
│   │
│   ├── processing/                # Document processing
│   │   ├── **init**.py
│   │   ├── pdf_extractor.py      # Extract text from PDFs
│   │   ├── text_cleaner.py       # Clean and normalize text
│   │   └── chunker.py            # Split text into chunks
│   │
│   ├── embedding/                 # Embedding generation
│   │   ├── **init**.py
│   │   ├── embed_model.py        # Load embedding model
│   │   └── generate_embeddings.py # Create embeddings
│   │
│   ├── rag/                       # RAG pipeline
│   │   ├── **init**.py
│   │   ├── retrieval.py          # Vector similarity search
│   │   ├── generation.py         # LLM response generation
│   │   └── pipeline.py           # Complete RAG workflow
│   │
│   ├── api/                       # API and LLM interactions
│   │   ├── **init**.py
│   │   ├── ollama_client.py      # Ollama API wrapper
│   │   └── prompts.py            # Prompt templates
│   │
│   └── utils/                     # Utility functions
│       ├── **init**.py
│       ├── logger.py             # Logging configuration
│       └── helpers.py            # Common helper functions
│
├── scripts/                       # Executable scripts
│   ├── setup_database.py         # Initialize database schema
│   ├── ingest_documents.py       # Process and store documents
│   ├── test_retrieval.py         # Test vector search
│   └── test_rag.py               # Test complete RAG pipeline
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_tests.ipynb
│   ├── 03_retrieval_evaluation.ipynb
│   └── 04_rag_experiments.ipynb
│
├── app/                           # Frontend application
│   ├── streamlit_app.py          # Main Streamlit interface
│   └── components/               # Reusable UI components
│       ├── **init**.py
│       ├── chat_interface.py
│       └── source_display.py
│
├── tests/                         # Unit tests
│   ├── **init**.py
│   ├── test_database.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_rag.py
│
├── models/                        # Saved models (optional)
│   └── README.md
│
├── docs/                          # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── setup_guide.md
│   └── dissertation/
│       ├── chapters/
│       └── figures/
│
├── docker/                        # Docker configurations
│   ├── docker-compose.yml
│   ├── Dockerfile.app
│   └── postgres/
│       └── init.sql
│
├── .env.example                   # Example environment variables
├── .env                          # Actual environment variables (gitignored)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
├── setup.py                      # Package installation
└── LICENSE                       # License file
