"""Embedding module - exports embedding functions"""
from .embed_model import get_embedding_model, embed_text, embed_texts, compute_similarity
from .generate_embeddings import embed_document_chunks, embed_and_store

__all__ = [
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "compute_similarity",
    "embed_document_chunks",
    "embed_and_store"
]
