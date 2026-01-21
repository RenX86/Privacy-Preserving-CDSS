"""RAG module - exports the complete RAG pipeline"""
from .retrieval import retrieve_relevant_chunks, format_context, get_sources
from .generation import generate_answer, generate_answer_stream
from .pipeline import CDSSPipeline, ask, QueryResult

__all__ = [
    "retrieve_relevant_chunks",
    "format_context",
    "get_sources",
    "generate_answer",
    "generate_answer_stream",
    "CDSSPipeline",
    "ask",
    "QueryResult"
]
