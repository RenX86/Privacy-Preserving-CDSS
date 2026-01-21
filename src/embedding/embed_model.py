"""
Embedding Model Module

PURPOSE:
Load and manage the sentence-transformers model for creating embeddings.
Embeddings convert text into vectors that capture semantic meaning.

HOW IT WORKS:
- Text like "heart disease symptoms" becomes a 384-dimensional vector
- Similar texts have vectors that are close in vector space
- This enables semantic search (find related content, not just keyword matches)

USAGE:
    from src.embedding.embed_model import get_embedding_model, embed_text
    
    # Embed a single text
    vector = embed_text("What are the symptoms of heart failure?")
    
    # Embed multiple texts (more efficient)
    vectors = embed_texts(["text1", "text2", "text3"])
"""

from typing import List, Optional
import numpy as np

from src.config.settings import settings
from src.utils.logger import logger

# Global model instance (loaded once, reused)
_model = None


def get_embedding_model():
    """
    Load the embedding model.
    
    Uses lazy loading - model is only loaded when first needed.
    Subsequent calls return the cached model.
    
    Returns:
        SentenceTransformer model instance
    """
    global _model
    
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        logger.info("This may take a few minutes for large models...")
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model with trust_remote_code for HuggingFace models
            _model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                trust_remote_code=True,
                device=device
            )
            
            # Log model info
            embedding_dim = _model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded successfully (dimension: {embedding_dim})")
            
            # Verify dimension matches settings
            if embedding_dim != settings.EMBEDDING_DIMENSION:
                logger.warning(
                    f"⚠ Dimension mismatch! Model: {embedding_dim}, Settings: {settings.EMBEDDING_DIMENSION}"
                )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    return _model


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string into an embedding vector.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats (384-dimensional vector for MiniLM)
    """
    model = get_embedding_model()
    
    # Generate embedding
    embedding = model.encode(text, convert_to_numpy=True)
    
    # Convert to list for database storage
    return embedding.tolist()


def embed_texts(texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
    """
    Convert multiple texts into embedding vectors.
    More efficient than calling embed_text() in a loop.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process at once
        show_progress: Show progress bar for large batches
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    model = get_embedding_model()
    
    logger.info(f"Embedding {len(texts)} texts (batch size: {batch_size})")
    
    # Generate embeddings in batches
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    logger.info(f"✓ Created {len(embeddings)} embeddings")
    
    # Convert to list of lists for database
    return [emb.tolist() for emb in embeddings]


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Returns:
        Similarity score between -1 and 1 (1 = identical, 0 = unrelated)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def get_model_info() -> dict:
    """Get information about the loaded embedding model."""
    model = get_embedding_model()
    
    return {
        "model_name": settings.EMBEDDING_MODEL,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": model.max_seq_length
    }


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Embedding Model Test ===\n")
    
    # Load model and get info
    info = get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Dimension: {info['embedding_dimension']}")
    print(f"Max length: {info['max_sequence_length']} tokens\n")
    
    # Test single embedding
    test_text = "What are the treatment options for breast cancer?"
    print(f"Test text: '{test_text}'")
    
    embedding = embed_text(test_text)
    print(f"Embedding shape: {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}\n")
    
    # Test similarity
    similar_text = "How is breast cancer treated?"
    different_text = "What is the weather like today?"
    
    emb_similar = embed_text(similar_text)
    emb_different = embed_text(different_text)
    
    sim_score = compute_similarity(embedding, emb_similar)
    diff_score = compute_similarity(embedding, emb_different)
    
    print("Similarity test:")
    print(f"  '{test_text[:40]}...'")
    print(f"  vs '{similar_text}' → {sim_score:.3f}")
    print(f"  vs '{different_text}' → {diff_score:.3f}")
    
    print("\n✓ Embedding model ready!")
