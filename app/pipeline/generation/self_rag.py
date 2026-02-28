import ollama
from app.config import settings
from app.pipeline.generation.guardrails import (
    build_system_prompt,
    build_context_block,
    SAFE_FAILURE_MESSAGE,
)
from app.pipeline.retrieval.reranker import RetrievedChunk

def generate_answer(query: str, verified_chunks: list[RetrievedChunk]) -> str:

    if not verified_chunks:
        return SAFE_FAILURE_MESSAGE

    system_prompt = build_system_prompt()
    context_block = build_context_block(verified_chunks)

    user_message = f"""CLINICAL QUERY: {query}

{context_block}

Answer the query using ONLY the context above. Cite every claim."""

    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.message.content

    except Exception as e:
        print(f"[LLM] Error calling Ollama: {e}")
        return SAFE_FAILURE_MESSAGE

def self_rag_critic(query: str, draft_answer: str, verified_chunks: list[RetrievedChunk]) -> str:

    context_block = build_context_block(verified_chunks)

    critic_prompt = f"""You are a clinical fact-checker.

Review this draft clinical answer and check: does EVERY factual claim have explicit support in the context below?

DRAFT ANSWER: 
{draft_answer}

{context_block}

If ALL claims are supported → respond with exactly: VALID
If ANY claim lacks support → rewrite the answer using only supported claims.
Begin your rewrite immediately without any preamble."""

    try:
        response = ollama.chat(
            model=settings.LOCAL_LLM_MODEL,
            messages=[
                {"role": "user", "content": critic_prompt}
            ]
        )
        result = response.message.content.strip()
        
        if result.upper().startswith("VALID"):
            return draft_answer
        else:
            return result

    except Exception as e:
        print(f"[Self-RAG] Critic error: {e}")
        return draft_answer