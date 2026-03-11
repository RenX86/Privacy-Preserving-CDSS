from sentence_transformers import CrossEncoder

_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class RetrievedChunk:

    def __init__(self, text: str, source: str, reference: str, score: float = 0.0):
        self.text = text
        self.source = source
        self.reference = reference
        self.score = score

    def __repr__(self):
        return f"RetrievedChunk(source={self.source}, score={self.score:.3f})"

def rerank_chunks(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    pairs = [(query, chunk.text) for chunk in chunks]
    scores = _cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)

    return ranked

def _clean_conditions(raw: str) -> str:
    """Clean ClinVar pipe/semicolon-separated conditions into readable text."""
    if not raw:
        return "Not provided"
    # Split on pipes and semicolons, deduplicate, remove empty
    parts = [p.strip() for p in raw.replace("|", ";").split(";") if p.strip()]
    seen = set()
    unique = []
    for p in parts:
        key = p.lower()
        if key not in seen and key != "not provided":
            seen.add(key)
            unique.append(p)
    return "; ".join(unique) if unique else "Not provided"

def from_postgres_result(result: dict) -> RetrievedChunk:

    conditions = _clean_conditions(result.get('condition', ''))
    text = (
        f"Variant {result.get('rsid')} in gene {result.get('gene_symbol')} "
        f"is classified as {result.get('clinical_significance')}.\n"
        f"Condition: {conditions}.\n"
        f"Review Status: {result.get('review_status')}."
    )
    return RetrievedChunk(
        text=text,
        source="Clinvar",
        reference=result.get("rsid", "unknown")
    )

def from_vector_result(result: dict) -> RetrievedChunk:
    text = result.get("parent_text") or result.get("chunk_text", "")
    
    metadata = result.get("metadata", {})
    reference = None
    if isinstance(metadata, dict):
        for i in range(6, 0, -1):
            key = f"Header_{i}"
            if key in metadata:
                reference = metadata[key].replace("\n", " ").strip()
                if len(reference) > 60:
                    reference = reference[:57] + "..."
                break
        if not reference and "page" in metadata:
            reference = f"page {metadata['page']}"
                
    if not reference:
        reference = result.get("source", "unknown")
        
    return RetrievedChunk(
        text=text,
        source=result.get("source", "VectorDB"),
        reference=reference
    )

def from_clingen_result(result: dict, gene: str) -> RetrievedChunk:
    """
    Build a chunk from a ClinGen /api/genes row.
    Fields: symbol, hgnc_id, has_validity, has_actionability, has_dosage,
            has_variant, date_last_curated, location, name
    """
    symbol   = result.get("symbol", gene)
    hgnc_id  = result.get("hgnc_id", "unknown")
    validity = result.get("has_validity")      # True/False or None
    action   = result.get("has_actionability") # True/False or None
    dosage   = result.get("has_dosage")        # True/False or None
    variant  = result.get("has_variant")       # True/False or None
    curated  = result.get("date_last_curated", "unknown date")
    location = result.get("location", "unknown locus")
    name     = result.get("name", "")

    # Build a readable clinical summary
    validity_str  = "YES - gene-disease validity curated by ClinGen expert panel" if validity else "Not curated"
    action_str    = "YES - actionability curations available" if action else "No actionability curation"
    variant_str   = "YES - variant curation expert panel active" if variant else "No variant curation"

    text = (
        f"ClinGen Gene Summary for {symbol} ({hgnc_id}): {name}\n"
        f"Chromosomal location: {location}\n"
        f"Gene-Disease Validity: {validity_str}\n"
        f"Actionability: {action_str}\n"
        f"Dosage Sensitivity: {'Curated' if dosage else 'Not curated'}\n"
        f"Variant Expert Panel: {variant_str}\n"
        f"Last curated: {curated}\n"
        f"Interpretation note: ClinGen has validated the gene-disease relationship for {symbol}. "
        f"Variants in this gene should be interpreted using ClinGen/ACMG joint guidelines."
    )

    return RetrievedChunk(
        text=text,
        source="ClinGen",
        reference=f"{symbol} ({hgnc_id})"
    )