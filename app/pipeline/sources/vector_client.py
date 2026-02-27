from sentence_transformers import SentenceTransformer
from app.config import settings
import psycopg2
import psycopg2.extras

_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def embed_text(text: str) -> list[float]:
    return _model.encode(text).tolist()

def search_variants(query_text: str, top_k: int = 5, source_filter: str = None) -> list[dict]:

    query_vector = embed_text(query_text)

    if source_filter:
        sql = """ 
        SELECT id, source, category, gene, chunk_text, metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        WHERE source = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        params = (query_vector, source_filter, top_k)
    else:
        sql = """
        SELECT id, source, category, gene, chunk_text, metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM medical_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;"""
        params = (query_vector, top_k)
        
    conn = psycopg2.connect(settings.POSTGRES_URL)
    try:
        with conn.cursor(cursor_facotry=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
    finally:
        conn.close()