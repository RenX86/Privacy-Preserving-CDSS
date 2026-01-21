"""
Embedding Generator Module

PURPOSE:
Orchestrates the embedding generation process for document chunks.
Combines processing and embedding into a single workflow.

USAGE:
    from src.embedding.generate_embeddings import embed_document_chunks
    
    # Embed chunks and prepare for database
    embedded_chunks = embed_document_chunks(chunks, source_file="guideline.pdf")
"""

from typing import List, Dict, Any

from src.embedding.embed_model import embed_texts
from src.utils.logger import logger


def embed_document_chunks(
    chunks: List[str],
    source_file: str,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for document chunks and prepare for database insertion.
    
    Args:
        chunks: List of text chunks
        source_file: Name of the source document
        batch_size: Batch size for embedding generation
        
    Returns:
        List of dicts ready for database insertion with:
        - content: the text chunk
        - embedding: the vector
        - source_file: source document name
        - chunk_index: position in document
        - metadata: word/char counts
    """
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return []
    
    logger.info(f"Generating embeddings for {len(chunks)} chunks from {source_file}")
    
    # Generate embeddings for all chunks
    embeddings = embed_texts(chunks, batch_size=batch_size)
    
    # Combine chunks with embeddings and metadata
    embedded_chunks = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        embedded_chunks.append({
            "content": chunk,
            "embedding": embedding,
            "source_file": source_file,
            "chunk_index": i,
            "metadata": {
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            }
        })
    
    logger.info(f"✓ Created {len(embedded_chunks)} embedded chunks")
    return embedded_chunks


def embed_and_store(
    chunks: List[str],
    source_file: str,
    batch_size: int = 32
) -> int:
    """
    Generate embeddings and store directly in the database.
    Convenience function that combines embedding and insertion.
    
    Args:
        chunks: List of text chunks
        source_file: Name of the source document
        batch_size: Batch size for processing
        
    Returns:
        Number of chunks stored
    """
    from src.database.db_operations import insert_documents_batch
    
    # Generate embeddings
    embedded_chunks = embed_document_chunks(chunks, source_file, batch_size)
    
    if not embedded_chunks:
        return 0
    
    # Store in database
    count = insert_documents_batch(embedded_chunks)
    
    logger.info(f"✓ Stored {count} chunks from {source_file} in database")
    return count


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Embedding Generator Test ===\n")
    
    # Test chunks
    test_chunks = [
        "Pathogenic variants are those with strong evidence of causing disease.",
        "Likely pathogenic variants have supporting but not definitive evidence.",
        "Variants of uncertain significance require careful communication with patients."
    ]
    
    print(f"Test chunks: {len(test_chunks)}")
    for i, chunk in enumerate(test_chunks):
        print(f"  {i}: {chunk[:50]}...")
    
    print("\nGenerating embeddings...")
    result = embed_document_chunks(test_chunks, source_file="test.pdf")
    
    print(f"\nResult: {len(result)} embedded chunks")
    for item in result:
        print(f"  Chunk {item['chunk_index']}: {len(item['embedding'])}d vector, {item['metadata']['word_count']} words")
    
    print("\n✓ Embedding generator ready!")
