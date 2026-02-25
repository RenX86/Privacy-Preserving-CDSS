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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()