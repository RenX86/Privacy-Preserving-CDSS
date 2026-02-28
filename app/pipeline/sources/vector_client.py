from sentence_transformers import SentenceTransformer
from app.config import settings
import psycopg2
import psycopg2.extras

_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_text(text: str) -> list[float]:
    return _model.encode(text).tolist()

def search_documents(query_text: str, top_k: int = 5, source_filter: str = None, category_filter: str = None) -> list[dict]:

    query_vector = embed_text(query_text)

    if source_filter:
        where_clause = "WHERE source = %s"
        filter_param = source_filter
    elif category_filter:
        where_clause = "WHERE category = %s"
        filter_param = category_filter
    else:
        where_clause = ""
        filter_param = None

    if filter_param:
        sql = f""" 
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        params = (query_vector, filter_param, query_vector, top_k)
    else:
        sql = """
        SELECT id, source, category, gene, chunk_text, metadata, parent_text,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        params = (query_vector, query_vector, top_k)
        
    conn = psycopg2.connect(settings.POSTGRES_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
    finally:
        conn.close()