from sentence_transformers import SentenceTransformer
from app.config import settings

_embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def get_embedding() -> SentenceTransformer:
    return _embedding_model

def embed_text(text: str) -> list[float]:
    return _embedding_model.encode(text).tolist()