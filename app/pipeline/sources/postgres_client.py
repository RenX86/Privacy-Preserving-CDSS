import psycopg2
import psycopg2.extras
from app.config import settings

def get_connection():
    return psycopg2.connect(settings.POSTGRES_URL)

def get_variant_by_rsid(rsid: str) -> dict | None:
    query = """
    SELECT
        rsid,
        gene_symbol,
        chromosome,
        position,
        ref_allele,
        alt_allele,
        clinical_significance,
        review_status,
        condition,
        last_evaluated
    FROM variants
    WHERE rsid = %s
    LIMIT 1;
    """

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, (rsid,))
            result = cursor.fetchone()
            return dict(result) if result else None
    finally:
        conn.close()

def get_variant_by_gene(gene_symbol: str) -> list[dict]:
    query = """
    SELECT
        rsid,
        gene_symbol,
        clinical_significance,
        condition,
        review_status
    FROM variants
    WHERE gene_symbol = %s
    ORDER BY clinical_significance;
    """

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(query, (gene_symbol,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
    finally:
        conn.close()