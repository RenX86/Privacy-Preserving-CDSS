"""
Prompt Templates Module

PURPOSE:
Define prompt templates for the RAG system.
Good prompts are CRITICAL for accurate, grounded responses.

KEY PRINCIPLES:
1. Ground responses in retrieved context
2. Acknowledge when information is unavailable
3. Cite sources for traceability
4. Medical accuracy and caution

USAGE:
    from src.api.prompts import build_rag_prompt, SYSTEM_PROMPT
"""


# === System Prompt ===
# This sets the behavior/persona for the LLM

SYSTEM_PROMPT = """You are a Clinical Decision Support Assistant specialized in genetic variant interpretation and clinical guidelines.

CRITICAL RULES:
1. ONLY use information from the provided context. Do NOT use prior knowledge.
2. If the context doesn't contain relevant information, say "I don't have information about this in my knowledge base."
3. Always cite your sources using [Source X] format.
4. Be precise and medically accurate.
5. When uncertain, acknowledge the uncertainty.
6. This is decision SUPPORT - always recommend consulting with qualified healthcare professionals.

RESPONSE FORMAT:
- Provide clear, structured answers
- Use bullet points for multiple items
- Include source citations inline
- End with any relevant caveats or recommendations for further consultation"""


# === RAG Prompt Template ===
# Combines context with user query

RAG_PROMPT_TEMPLATE = """Based on the following clinical guidelines and evidence, answer the user's question.

CONTEXT FROM KNOWLEDGE BASE:
{context}

---

USER QUESTION: {query}

Provide a comprehensive answer based ONLY on the context above. Cite sources using [Source X] format."""


# === Fallback Prompts ===

NO_CONTEXT_PROMPT = """I apologize, but I couldn't find relevant information in my knowledge base to answer your question about: "{query}"

This could mean:
1. The topic isn't covered in the currently loaded clinical guidelines
2. The question may need to be rephrased
3. Additional guidelines may need to be ingested

Please consult with a qualified healthcare professional or try rephrasing your question."""


# === Builder Functions ===

def build_rag_prompt(query: str, context: str) -> str:
    """
    Build the complete RAG prompt with context and query.
    
    Args:
        query: User's question
        context: Retrieved and formatted context from documents
        
    Returns:
        Complete prompt string
    """
    if not context or context == "No relevant information found in the knowledge base.":
        return NO_CONTEXT_PROMPT.format(query=query)
    
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        query=query
    )


def build_followup_prompt(query: str, context: str, previous_qa: list) -> str:
    """
    Build a prompt that includes conversation history.
    
    Args:
        query: Current question
        context: Retrieved context
        previous_qa: List of (question, answer) tuples
        
    Returns:
        Complete prompt with history
    """
    history = ""
    if previous_qa:
        history_parts = []
        for q, a in previous_qa[-3:]:  # Last 3 exchanges
            history_parts.append(f"Q: {q}\nA: {a}")
        history = "PREVIOUS CONVERSATION:\n" + "\n\n".join(history_parts) + "\n\n---\n\n"
    
    return f"""{history}Based on the following context, answer the follow-up question.

CONTEXT:
{context}

---

FOLLOW-UP QUESTION: {query}

Answer based on the context. Reference previous answers if relevant."""


def build_summary_prompt(chunks: list) -> str:
    """
    Build a prompt to summarize multiple document chunks.
    """
    combined = "\n\n---\n\n".join([c.get('content', '') for c in chunks])
    
    return f"""Summarize the following clinical guideline excerpts into a concise overview.

EXCERPTS:
{combined}

---

Provide a structured summary of the key points."""


# Quick test
if __name__ == "__main__":
    print("=== Prompt Templates Test ===\n")
    
    # Test RAG prompt
    test_context = """[Source 1: acmg_guidelines.pdf (relevance: 0.85)]
Pathogenic variants are those with strong evidence of disease causation.

[Source 2: acmg_guidelines.pdf (relevance: 0.78)]
Evidence includes null variants, functional studies, and segregation data."""
    
    test_query = "What defines a pathogenic variant?"
    
    prompt = build_rag_prompt(test_query, test_context)
    
    print("Generated Prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    print("\nSystem Prompt Preview:")
    print(SYSTEM_PROMPT[:200] + "...")
    
    print("\n✓ Prompt templates ready!")
