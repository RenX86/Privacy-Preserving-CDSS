from app.models.embeddings import embed_text
from app.config import settings
from app.db.pool import db_conn
import psycopg2.extras


def search_documents(query_text: str, top_k: int = 5, source_filter: str = None, category_filter: str = None) -> list[dict]:

    query_vector = embed_text(query_text)

    # Build WHERE clause and params matching SQL placeholder order:
    # 1: %s::vector in SELECT (similarity), 2: %s in WHERE, 3: %s::vector in ORDER BY, 4: %s in LIMIT
    if source_filter:
        where_clause = "WHERE source = %s"
        params = (query_vector, source_filter, query_vector, top_k)
    elif category_filter:
        where_clause = "WHERE category = %s"
        params = (query_vector, category_filter, query_vector, top_k)
    else:
        where_clause = ""
        params = (query_vector, query_vector, top_k)

    if where_clause:
        sql = f"""
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
    else:
        sql = """
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""

    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]