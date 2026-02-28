CREATE EXTENSION IF NOT EXISTS vector;

-- ─── VARIANTS TABLE ───────────────────────────────────────────────────────────
-- Stores structured variant records from ClinVar
CREATE TABLE IF NOT EXISTS variants (
    id                     SERIAL PRIMARY KEY,
    rsid                   VARCHAR(20) UNIQUE,
    gene_symbol            VARCHAR(50),
    chromosome             VARCHAR(10),
    position               BIGINT,
    ref_allele             VARCHAR(255),
    alt_allele             VARCHAR(255),
    clinical_significance  VARCHAR(100),
    review_status          VARCHAR(100),
    condition              TEXT,
    last_evaluated         DATE,
    created_at             TIMESTAMP DEFAULT NOW()
);

-- ─── MEDICAL DOCUMENTS TABLE ──────────────────────────────────────────────────
-- Stores chunked ACMG/NCCN text with their vector embeddings (pgvector)

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

-- ─── SAMPLE TEST DATA ─────────────────────────────────────────────────────────
-- One real record so we can immediately test our Python code

INSERT INTO variants (rsid, gene_symbol, chromosome, position, ref_allele, alt_allele, clinical_significance, review_status, condition)
VALUES (
    'rs123424454',
    'BRCA1',
    '14',
    2432344,
    'A',
    'T',
    'pathogenic',
    'reviewed by expert panel',
    'Heredetitotry breast and ovarian cancer syndorme'
)ON CONFLICT (rsid)DO NOTHING;