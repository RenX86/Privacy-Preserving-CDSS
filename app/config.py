from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    POSTGRES_URL: str

    LOCAL_LLM_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "llama3"

    EMBEDDING_MODEL: str = "NLP4Science/pubmedbert-base-embeddings"

    CLINGEN_API_URL: str = "https://search.clinicalgenome.org/kb"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()