"""
Retrieval Module

PURPOSE:
Find relevant document chunks for a user query using vector similarity search.
This is the "R" in RAG - Retrieval!

HOW IT WORKS:
1. User asks: "What are pathogenic variants?"
2. We embed the query into a vector
3. Search the database for chunks with similar vectors
4. Return the most relevant chunks as context

USAGE:
    from src.rag.retrieval import retrieve_relevant_chunks
    
    results = retrieve_relevant_chunks("What is the treatment for BRCA1 mutations?")
    for chunk in results:
        print(chunk['content'])
"""

from typing import List, Dict, Any, Optional

from src.embedding.embed_model import embed_text
from src.database.db_operations import search_similar
from src.config.settings import settings
from src.utils.logger import logger


def retrieve_relevant_chunks(
    query: str,
    top_k: int = None,
    source_filter: Optional[str] = None,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Retrieve document chunks relevant to the query.
    
    Args:
        query: The user's question
        top_k: Number of chunks to retrieve (default from settings)
        source_filter: Only search within specific source file
        min_similarity: Minimum similarity score to include
        
    Returns:
        List of relevant chunks with content and metadata
    """
    top_k = top_k or settings.TOP_K_RESULTS
    
    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
    
    # Step 1: Embed the query
    query_embedding = embed_text(query)
    
    # Step 2: Search for similar documents
    results = search_similar(
        query_embedding=query_embedding,
        top_k=top_k,
        source_filter=source_filter
    )
    
    # Step 3: Filter by minimum similarity if specified
    if min_similarity > 0:
        original_count = len(results)
        results = [r for r in results if r.get('similarity', 0) >= min_similarity]
        if len(results) < original_count:
            logger.debug(f"Filtered {original_count - len(results)} results below similarity {min_similarity}")
    
    logger.info(f"Retrieved {len(results)} relevant chunks")
    
    return results


def format_context(chunks: List[Dict[str, Any]], include_sources: bool = True) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of retrieved chunks
        include_sources: Whether to include source citations
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant information found in the knowledge base."
    
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get('content', '')
        source = chunk.get('source_file', 'Unknown')
        similarity = chunk.get('similarity', 0)
        
        if include_sources:
            context_parts.append(
                f"[Source {i}: {source} (relevance: {similarity:.2f})]\n{content}"
            )
        else:
            context_parts.append(content)
    
    return "\n\n---\n\n".join(context_parts)


def get_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract source information from retrieved chunks for citation.
    
    Returns:
        List of source references with file, chunk index, and similarity
    """
    return [
        {
            "source_file": chunk.get('source_file', 'Unknown'),
            "chunk_index": chunk.get('chunk_index', 0),
            "similarity": round(chunk.get('similarity', 0), 3),
            "preview": chunk.get('content', '')[:100] + "..."
        }
        for chunk in chunks
    ]


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Retrieval Module Test ===\n")
    
    from src.database.db_operations import get_document_count
    
    count = get_document_count()
    print(f"Documents in database: {count}")
    
    if count == 0:
        print("\nNo documents found. Run ingest_documents.py first!")
    else:
        # Test retrieval
        test_query = "What are pathogenic variants?"
        print(f"\nTest query: '{test_query}'")
        
        results = retrieve_relevant_chunks(test_query, top_k=3)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Source: {result.get('source_file')} (similarity: {result.get('similarity', 0):.3f})")
            print(f"   Content: {result.get('content', '')[:150]}...")
    
    print("\n✓ Retrieval module ready!")
