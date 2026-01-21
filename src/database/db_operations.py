"""
Database Operations Module

PURPOSE:
CRUD operations for the documents table with vector embeddings.
Provides functions to insert, search, and manage document chunks.

USAGE:
    from src.database.db_operations import insert_document, search_similar
    
    # Insert a document chunk with its embedding
    insert_document(
        content="Clinical guideline text...",
        embedding=[0.1, 0.2, ...],  # 384-dimensional vector
        source_file="acmg_guidelines.pdf",
        chunk_index=0
    )
    
    # Search for similar documents
    results = search_similar(query_embedding, top_k=5)
"""

from typing import List, Dict, Any, Optional
import json

from src.database.db_connection import get_db_cursor
from src.config.settings import settings
from src.utils.logger import logger


def insert_document(
    content: str,
    embedding: List[float],
    source_file: str,
    chunk_index: int,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Insert a document chunk with its embedding into the database.
    
    Args:
        content: The text content of the chunk
        embedding: Vector embedding (list of floats)
        source_file: Name of the source PDF/document
        chunk_index: Position of this chunk in the document
        metadata: Optional additional metadata (stored as JSON)
        
    Returns:
        The ID of the inserted document
    """
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (content, embedding, source_file, chunk_index, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                content,
                embedding,  # pgvector accepts list directly
                source_file,
                chunk_index,
                json.dumps(metadata) if metadata else None
            )
        )
        doc_id = cursor.fetchone()[0]
        logger.debug(f"Inserted document chunk {chunk_index} from {source_file} (ID: {doc_id})")
        return doc_id


def insert_documents_batch(documents: List[Dict[str, Any]]) -> int:
    """
    Insert multiple document chunks at once (faster for bulk inserts).
    
    Args:
        documents: List of dicts with keys: content, embedding, source_file, chunk_index, metadata
        
    Returns:
        Number of documents inserted
    """
    if not documents:
        return 0
        
    with get_db_cursor() as cursor:
        from psycopg2.extras import execute_values
        
        values = [
            (
                doc["content"],
                doc["embedding"],
                doc["source_file"],
                doc["chunk_index"],
                json.dumps(doc.get("metadata")) if doc.get("metadata") else None
            )
            for doc in documents
        ]
        
        execute_values(
            cursor,
            """
            INSERT INTO documents (content, embedding, source_file, chunk_index, metadata)
            VALUES %s
            """,
            values
        )
        
        logger.info(f"Batch inserted {len(documents)} document chunks")
        return len(documents)


def search_similar(
    query_embedding: List[float],
    top_k: int = 5,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents similar to the query embedding using cosine distance.
    
    This is the CORE of RAG - finding relevant chunks to use as context!
    
    Args:
        query_embedding: Vector embedding of the search query
        top_k: Number of results to return
        source_filter: Optional filter by source file name
        
    Returns:
        List of matching documents with content, source, and similarity score
    """
    # Convert list to string format for pgvector
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    with get_db_cursor(dict_cursor=True) as cursor:
        if source_filter:
            cursor.execute(
                """
                SELECT 
                    id,
                    content,
                    source_file,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                WHERE source_file = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, source_filter, embedding_str, top_k)
            )
        else:
            cursor.execute(
                """
                SELECT 
                    id,
                    content,
                    source_file,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, top_k)
            )
        
        results = cursor.fetchall()
        logger.debug(f"Found {len(results)} similar documents")
        return [dict(row) for row in results]


def get_document_count() -> int:
    """Get the total number of documents in the database."""
    with get_db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        return count


def get_unique_sources() -> List[str]:
    """Get list of all unique source files in the database."""
    with get_db_cursor() as cursor:
        cursor.execute("SELECT DISTINCT source_file FROM documents ORDER BY source_file")
        return [row[0] for row in cursor.fetchall()]


def delete_by_source(source_file: str) -> int:
    """
    Delete all documents from a specific source file.
    Useful for re-ingesting updated documents.
    
    Args:
        source_file: Name of the source file to delete
        
    Returns:
        Number of documents deleted
    """
    with get_db_cursor() as cursor:
        cursor.execute(
            "DELETE FROM documents WHERE source_file = %s",
            (source_file,)
        )
        deleted = cursor.rowcount
        logger.info(f"Deleted {deleted} chunks from {source_file}")
        return deleted


def clear_all_documents() -> int:
    """
    Delete ALL documents from the database.
    Use with caution!
    
    Returns:
        Number of documents deleted
    """
    with get_db_cursor() as cursor:
        cursor.execute("DELETE FROM documents")
        deleted = cursor.rowcount
        logger.warning(f"Cleared all documents ({deleted} total)")
        return deleted


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Database Operations Test ===\n")
    
    # Check document count
    count = get_document_count()
    print(f"Current document count: {count}")
    
    # Check sources
    sources = get_unique_sources()
    print(f"Unique sources: {sources if sources else 'None yet'}")
    
    print("\n✓ Database operations module ready!")
