"""
RAG Pipeline Module

PURPOSE:
The complete RAG pipeline that ties everything together.
Query → Retrieve → Generate → Response with sources

THIS IS THE MAIN INTERFACE for the CDSS!

USAGE:
    from src.rag.pipeline import ask, CDSSPipeline
    
    # Simple one-shot query
    result = ask("What are the treatment options for BRCA1 mutations?")
    print(result['answer'])
    print(result['sources'])
    
    # Or use the pipeline class for more control
    pipeline = CDSSPipeline()
    result = pipeline.query("What is a pathogenic variant?")
"""

from typing import Dict, Any, Optional, List, Generator
from dataclasses import dataclass

from src.rag.retrieval import retrieve_relevant_chunks, format_context, get_sources
from src.rag.generation import generate_answer, generate_answer_stream
from src.api.ollama_client import check_ollama_status
from src.database.db_operations import get_document_count
from src.config.settings import settings
from src.utils.logger import logger


@dataclass
class QueryResult:
    """Structured result from a RAG query."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    chunks_retrieved: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "context_used": self.context_used,
            "chunks_retrieved": self.chunks_retrieved
        }


class CDSSPipeline:
    """
    Clinical Decision Support System RAG Pipeline.
    
    Combines retrieval and generation into a complete workflow.
    """
    
    def __init__(
        self,
        top_k: int = None,
        temperature: float = 0.3,
        min_similarity: float = 0.0
    ):
        """
        Initialize the pipeline.
        
        Args:
            top_k: Number of chunks to retrieve
            temperature: LLM temperature (lower = more factual)
            min_similarity: Minimum similarity threshold
        """
        self.top_k = top_k or settings.TOP_K_RESULTS
        self.temperature = temperature
        self.min_similarity = min_similarity
        
        logger.info(f"CDSS Pipeline initialized (top_k={self.top_k}, temp={self.temperature})")
    
    def check_system_status(self) -> Dict[str, bool]:
        """Check if all components are ready."""
        status = {
            "ollama": check_ollama_status(),
            "database": get_document_count() > 0
        }
        status["ready"] = all(status.values())
        return status
    
    def query(
        self,
        question: str,
        source_filter: Optional[str] = None
    ) -> QueryResult:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: The user's clinical question
            source_filter: Optional filter to specific source
            
        Returns:
            QueryResult with answer and sources
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"CDSS Query: {question}")
        logger.info(f"{'='*50}")
        
        # Step 1: Retrieve relevant chunks
        logger.info("Step 1: Retrieving relevant context...")
        chunks = retrieve_relevant_chunks(
            query=question,
            top_k=self.top_k,
            source_filter=source_filter,
            min_similarity=self.min_similarity
        )
        
        if not chunks:
            logger.warning("No relevant chunks found")
            return QueryResult(
                query=question,
                answer="I couldn't find relevant information in my knowledge base to answer this question. Please try rephrasing or consult with a healthcare professional.",
                sources=[],
                context_used="",
                chunks_retrieved=0
            )
        
        # Step 2: Format context
        logger.info("Step 2: Formatting context...")
        context = format_context(chunks)
        sources = get_sources(chunks)
        
        # Step 3: Generate answer
        logger.info("Step 3: Generating answer...")
        answer = generate_answer(
            query=question,
            context=context,
            temperature=self.temperature
        )
        
        logger.info("✓ Query processed successfully")
        
        return QueryResult(
            query=question,
            answer=answer,
            sources=sources,
            context_used=context,
            chunks_retrieved=len(chunks)
        )
    
    def query_stream(
        self,
        question: str,
        source_filter: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a query response for real-time display.
        
        Yields:
            Dicts with 'type' and 'content' keys
        """
        # Retrieve chunks first
        chunks = retrieve_relevant_chunks(
            query=question,
            top_k=self.top_k,
            source_filter=source_filter
        )
        
        if not chunks:
            yield {"type": "error", "content": "No relevant information found."}
            return
        
        context = format_context(chunks)
        sources = get_sources(chunks)
        
        # Yield sources first
        yield {"type": "sources", "content": sources}
        
        # Stream the answer
        for chunk in generate_answer_stream(question, context, self.temperature):
            yield {"type": "answer_chunk", "content": chunk}
        
        yield {"type": "done", "content": None}


# === Convenience Function ===

def ask(question: str, **kwargs) -> Dict[str, Any]:
    """
    Simple one-shot query function.
    
    Args:
        question: Your clinical question
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        Dict with 'answer', 'sources', and metadata
    """
    pipeline = CDSSPipeline(**kwargs)
    result = pipeline.query(question)
    return result.to_dict()


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  CDSS RAG Pipeline Test")
    print("=" * 60)
    print()
    
    # Check system status
    pipeline = CDSSPipeline()
    status = pipeline.check_system_status()
    
    print("System Status:")
    print(f"  Ollama: {'✓' if status['ollama'] else '✗'}")
    print(f"  Database: {'✓' if status['database'] else '✗ (run ingest_documents.py first)'}")
    
    if not status['ready']:
        print("\nSystem not ready. Please:")
        if not status['ollama']:
            print("  1. Start Ollama: ollama serve")
        if not status['database']:
            print("  2. Ingest documents: python scripts/ingest_documents.py")
        exit(1)
    
    print("\n" + "-" * 60)
    
    # Test query
    test_question = "What are pathogenic variants?"
    print(f"Test Query: {test_question}")
    print("-" * 60)
    
    result = pipeline.query(test_question)
    
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nSources ({len(result.sources)}):")
    for src in result.sources:
        print(f"  - {src['source_file']} (similarity: {src['similarity']})")
    
    print("\n" + "=" * 60)
    print("  ✓ CDSS Pipeline Ready!")
    print("=" * 60)
