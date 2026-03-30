CREATE EXTENSION IF NOT EXISTS vector;

-- ─── VARIANTS TABLE ───────────────────────────────────────────────────────────
-- Stores structured variant records from ClinVar
CREATE TABLE IF NOT EXISTS variants (
    id                     SERIAL PRIMARY KEY,
    rsid                   VARCHAR(20) UNIQUE,
    gene_symbol            TEXT,
    chromosome             VARCHAR(10),
    position               BIGINT,
    ref_allele             TEXT,
    alt_allele             TEXT,
    clinical_significance  TEXT,
    review_status          TEXT,
    condition              TEXT,
    last_evaluated         DATE,
    created_at             TIMESTAMP DEFAULT NOW()
);

-- ─── MEDICAL DOCUMENTS TABLE ──────────────────────────────────────────────────
-- Stores chunked NCCN guideline text with their vector embeddings (pgvector)

CREATE TABLE IF NOT EXISTS medical_documents (
    id                     SERIAL PRIMARY KEY,
    source                 VARCHAR(50),
    category               VARCHAR(50),
    gene                   VARCHAR(50),
    chunk_text             TEXT NOT NULL,
    embedding              vector(768),
    metadata               JSONB,
    parent_text            TEXT,
    created_at             TIMESTAMP DEFAULT NOW()
);

-- ─── INDEXES ──────────────────────────────────────────────────────────────────
-- Indexes make lookups fast (like a book index)

CREATE INDEX IF NOT EXISTS idx_variants_rsid ON variants (rsid);
CREATE INDEX IF NOT EXISTS idx_variants_gene ON variants(gene_symbol);
CREATE INDEX IF NOT EXISTS idx_documents_source ON medical_documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_gene ON medical_documents(gene);
CREATE INDEX IF NOT EXISTS idx_documents_parent_text ON medical_documents(parent_text);

-- Vector similarity search index (for fast pgvector searches)
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON medical_documents USING ivfflat (embedding vector_cosine_ops);

