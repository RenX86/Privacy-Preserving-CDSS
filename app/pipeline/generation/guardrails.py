from app.pipeline.retrieval.reranker import RetrievedChunk

SAFE_FAILURE_MESSAGE = (
    "Insufficient clinical data to safely provide an assessment. "
    "Please consult primary clinical databases or a qualified clinician."
)

def build_system_prompt() -> str:

    return  """You are a clinical decision support assistant.
STRICT RULES — you must follow these without exception:
1. ONLY make claims that are explicitly supported by the provided context below.
2. If the context does not contain enough information to answer safely, respond ONLY with: "Insufficient clinical data to safely provide an assessment."
3. NEVER invent, guess, or infer clinical facts not present in the context.
4. ALWAYS cite your source for every factual claim using [Source: name, reference].
5. Use precise clinical language. Do not simplify medical terms.
6. Do not add disclaimers or padding — be direct and factual.
"""

def build_context_block(chunks: list[RetrievedChunk]) -> str:

    if not chunks:
        return "No verified clinical context available."

    lines = ["--- VERIFIED CLINICAL CONTEXT ---"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"\n[{i}] source: {chunk.source} | Reference: {chunk.reference}")
        lines.append(chunk.text)

    lines.append("\n--- END OF CONTEXT ---")
    return "\n".join(lines)