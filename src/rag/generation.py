"""
Generation Module

PURPOSE:
Generate LLM responses grounded in retrieved context.
This is the "G" in RAG - Generation!

USAGE:
    from src.rag.generation import generate_answer
    
    answer = generate_answer(
        query="What are pathogenic variants?",
        context="Clinical guideline text..."
    )
"""

from typing import Optional, Generator

from src.api.ollama_client import generate_response, generate_response_stream
from src.api.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.utils.logger import logger


def generate_answer(
    query: str,
    context: str,
    temperature: float = 0.3,  # Lower = more factual
    max_tokens: int = 1024,
    model: Optional[str] = None
) -> str:
    """
    Generate an answer to the query based on the context.
    
    Args:
        query: User's question
        context: Retrieved document context
        temperature: LLM creativity (0.1-0.3 recommended for medical)
        max_tokens: Maximum response length
        model: Optional model override
        
    Returns:
        Generated answer text
    """
    logger.info(f"Generating answer for: '{query[:50]}...'")
    
    # Build the prompt
    prompt = build_rag_prompt(query, context)
    
    # Generate response
    response = generate_response(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model
    )
    
    return response


def generate_answer_stream(
    query: str,
    context: str,
    temperature: float = 0.3,
    model: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream an answer for real-time display.
    
    Yields:
        Text chunks as they're generated
    """
    logger.info(f"Streaming answer for: '{query[:50]}...'")
    
    prompt = build_rag_prompt(query, context)
    
    for chunk in generate_response_stream(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=temperature,
        model=model
    ):
        yield chunk


# Quick test
if __name__ == "__main__":
    print("=== Generation Module Test ===\n")
    
    from src.api.ollama_client import check_ollama_status
    
    if not check_ollama_status():
        print("Ollama not running. Start with: ollama serve")
        exit(1)
    
    # Test with sample context
    test_context = """[Source 1: acmg_guidelines.pdf]
Pathogenic variants are genetic changes with strong evidence of causing disease.
Examples include null variants such as nonsense, frameshift, and splice site mutations.

[Source 2: acmg_guidelines.pdf]
Classification requires evaluation of population frequency, functional data, and segregation."""
    
    test_query = "What are pathogenic variants and how are they classified?"
    
    print(f"Query: {test_query}\n")
    print("Generating answer...")
    print("-" * 50)
    
    answer = generate_answer(test_query, test_context)
    print(answer)
    print("-" * 50)
    
    print("\n✓ Generation module ready!")
