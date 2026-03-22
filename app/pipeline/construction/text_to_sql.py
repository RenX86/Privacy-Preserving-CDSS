"""
Text-to-SQL Query Builder — Planned Feature

Planned: translate natural language clinical questions into SQL queries
against the ClinVar variants table, enabling flexible variant lookups
beyond simple rsID or gene_symbol exact matches.

Not yet implemented — postgres_client.py provides dedicated functions
(get_variant_by_rsid, get_variant_by_gene) that handle the current
query patterns via parameterized SQL.
"""


def build_sql_query(natural_language_query: str, gene: str = None) -> str:
    """
    Convert a natural language query into a parameterized SQL query.

    Would support queries like:
    - "Show all pathogenic BRCA1 variants reviewed by expert panel"
    - "Find VUS variants in TP53 on chromosome 17"

    Currently not implemented — direct SQL functions in
    postgres_client.py handle rsID and gene lookups.
    """
    raise NotImplementedError(
        "Text-to-SQL query building is not yet implemented. "
        "Use postgres_client.get_variant_by_rsid() or "
        "get_variant_by_gene() instead."
    )
