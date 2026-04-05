"""
Self-Query Retriever — Planned Feature

Planned: extract metadata filters (source, category, gene) from natural
language queries to narrow vector search before embedding lookup.

Not yet implemented — vector_client.py category_filter handles this
manually via keyword-based routing in decomposition.py.
"""


def build_self_query_filter(query: str) -> dict:
    """
    Extract metadata filters from a natural language query.

    Returns a dict with optional keys: 'source', 'category', 'gene'.
    These would be passed to vector_client.search_documents() as
    pre-filters before the cosine similarity search.

    Currently not implemented — the pipeline uses deterministic
    keyword routing in decomposition.py to select the correct
    category_filter for each sub-query.
    """
    raise NotImplementedError(
        "Self-query metadata extraction is not yet implemented. "
        "Use decomposition.py keyword routing instead."
    )
