-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for storing document embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),  -- Dimension for sentence-transformers/all-MiniLM-L6-v2
    source_file VARCHAR(255),
    chunk_index INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index on source_file for filtering
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source_file);
