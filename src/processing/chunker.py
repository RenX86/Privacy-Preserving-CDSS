"""
Text Chunker Module

PURPOSE:
Split long documents into smaller chunks for embedding.
Each chunk becomes a searchable unit in our vector database.

WHY CHUNKING MATTERS:
- LLMs have context limits
- Smaller chunks = more precise retrieval
- Too small = lose context, too big = dilute relevance
- Overlap helps preserve context across chunk boundaries

USAGE:
    from src.processing.chunker import chunk_text
    
    chunks = chunk_text(long_text, chunk_size=500, overlap=50)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:100]}...")
"""

from typing import List
import re

from src.config.settings import settings
from src.utils.logger import logger


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    respect_sentences: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Target words per chunk (default from settings)
        chunk_overlap: Words to overlap between chunks (default from settings)
        respect_sentences: Try to break at sentence boundaries
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Use defaults from settings if not specified
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    if respect_sentences:
        chunks = chunk_by_sentences(text, chunk_size, chunk_overlap)
    else:
        chunks = chunk_by_words(text, chunk_size, chunk_overlap)
    
    logger.info(f"Created {len(chunks)} chunks (target: {chunk_size} words, overlap: {chunk_overlap})")
    
    return chunks


def chunk_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple chunking by word count.
    Fast but may break mid-sentence.
    """
    words = text.split()
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        # Move start position, accounting for overlap
        start = end - overlap
        
        # Prevent infinite loop if overlap >= chunk_size
        if start >= len(words) - overlap:
            break
    
    return chunks


def chunk_by_sentences(text: str, target_size: int, overlap: int) -> List[str]:
    """
    Chunk while respecting sentence boundaries.
    Produces more coherent chunks for medical text.
    """
    # Split into sentences (handles common abbreviations)
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_pattern.split(text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence exceeds target, save current chunk and start new one
        if current_word_count + sentence_words > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep some sentences for overlap
            overlap_sentences = []
            overlap_word_count = 0
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_word_count + s_words <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_word_count += s_words
                else:
                    break
            
            current_chunk = overlap_sentences
            current_word_count = overlap_word_count
        
        current_chunk.append(sentence)
        current_word_count += sentence_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def chunk_with_metadata(
    text: str,
    source_file: str,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[dict]:
    """
    Chunk text and return with metadata ready for database insertion.
    
    Returns:
        List of dicts with 'content', 'source_file', 'chunk_index', 'metadata'
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    return [
        {
            "content": chunk,
            "source_file": source_file,
            "chunk_index": i,
            "metadata": {
                "word_count": len(chunk.split()),
                "char_count": len(chunk)
            }
        }
        for i, chunk in enumerate(chunks)
    ]


def estimate_chunks(text: str, chunk_size: int = None) -> int:
    """
    Estimate how many chunks will be created without actually chunking.
    Useful for progress bars and planning.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    word_count = len(text.split())
    return max(1, word_count // (chunk_size // 2))  # Rough estimate with overlap


# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Text Chunker Test ===\n")
    
    # Sample medical text
    sample = """
    The American College of Medical Genetics and Genomics (ACMG) has developed 
    standards and guidelines for the interpretation of sequence variants. These 
    guidelines are designed to help clinical laboratories and healthcare providers 
    classify genetic variants systematically.
    
    Pathogenic variants are those that have strong evidence of causing disease. 
    These include variants that result in a null effect on the gene product, such 
    as nonsense mutations, frameshift mutations, or canonical splice site mutations.
    
    Likely pathogenic variants have sufficient evidence to support a disease-causing 
    role, but the evidence is not definitive. Additional supporting evidence may 
    include functional studies, segregation data, or computational predictions.
    
    Variants of uncertain significance (VUS) represent a significant challenge in 
    clinical genetics. These variants lack sufficient evidence to classify them as 
    either pathogenic or benign, requiring careful communication with patients.
    
    Benign variants are common in the population and do not cause disease. These 
    include synonymous variants that do not affect protein function and variants 
    found at high frequency in population databases.
    """
    
    print(f"Sample text: {len(sample.split())} words\n")
    
    # Test chunking
    chunks = chunk_text(sample, chunk_size=100, chunk_overlap=20)
    
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        print(f"Chunk {i} ({word_count} words):")
        print(f"  {chunk[:150]}...")
        print()
    
    print("✓ Chunker module ready!")
