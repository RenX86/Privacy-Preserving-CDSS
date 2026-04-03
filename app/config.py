from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    POSTGRES_URL: str 
    POSTGRES_USER: str 
    POSTGRES_PASSWORD: str 
    POSTGRES_PORT: str 
    POSTGRES_DB: str 
    LOCAL_LLM_URL: str 
    LOCAL_LLM_MODEL: str 
    EMBEDDING_MODEL: str 
    CLINGEN_API_URL: str

    # Set to false to disable live gnomAD lookups (privacy-sensitive environments).
    # When disabled, population frequency data will be omitted from responses.
    ENABLE_GNOMAD_LOOKUP: bool = True 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()