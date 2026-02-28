from sentence_transformers import CrossEncoder

_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class RetrievedChunk:

    def __init__(self, text: str, source: str, reference: str, score: float = 0.0):
        self.text = text
        self.source = source
        self.reference = reference
        self.score = score

    def __repr__(self):
        return f"RetrievedChunk(source={self.source}, score={self.score:.3f}"

def rerank_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    pairs = [(query, chunk.text) for chunk in chunks]
    scores = _cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)

    return ranked

def from_postgres_result(result: dict) -> RetrievedChunk:

    text = (
        f"Variant {result.get('rsid')} in gene {result.get('gene_symbol')} "
        f"is classified as {result.get('clinical_significance')} "
        f"Condition: {result.get('condition')}"
        f"Review Status: {result.get('review_status')} "

    )
    return RetrievedChunk(
        text=text,
        source="Clinvar",
        reference=result.get("rsid", "unknown")
    )

def from_vector_result(result: dict) -> RetrievedChunk:
    text = result.get("parent_text") or result.get("chunk_text", "")
    return RetrievedChunk(
        text=text,
        source=result.get("source", "VectorDB"),
        reference=result.get("source", "unknown")
    )

def from_clingen_result(result: dict) -> RetrievedChunk:

    classification = result.get("classification", {}).get("label", "Unknown")
    disease = result.get("disease", {}).get("label", "Unknown condition")
    text = (
        f"ClinGen expert panel classification for {gene}: "
        f"{classification}. Disease: {disease}."
    )
    return RetrievedChunk(
        text=text,
        source="ClinGen",
        reference=f"{gene}-EP"
    )